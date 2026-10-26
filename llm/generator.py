import base64
import hashlib
import os
import random
import time
from types import SimpleNamespace

import httpx
from google import genai
from google.genai import errors, types

from llm.evidence_builder import EventEvidenceBuilder
from llm.stage_cache import StageCache
from llm.response_parser import (
    parse_analysis_response,
    parse_commentary_response,
    parse_verification_response,
    verification_response_diagnostics,
)
from llm.video_prompt import VideoCommentaryPrompt
from vision.possession_verifier import (
    LocalPossessionVerifier,
    mark_team_mentions,
    reconcile_possession,
)


RETRYABLE_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}
MAX_MODEL_ATTEMPTS = 3
UPLOAD_CACHE_MAX_AGE_SECONDS = 20 * 60 * 60
STAGE_CACHE_SCHEMA_VERSION = 1

VERIFICATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "events": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "event_id": {"type": "INTEGER"},
                    "start_seconds": {"type": "NUMBER"},
                    "end_seconds": {"type": "NUMBER"},
                    "cue_seconds": {"type": "NUMBER"},
                    "action_type": {
                        "type": "STRING",
                        "enum": [
                            "carry", "pass_release", "reception", "back_pass",
                            "long_ball", "challenge", "clearance", "cross",
                            "shot", "save", "restart", "movement", "unclear",
                        ],
                    },
                    "description": {"type": "STRING"},
                    "ball_state": {
                        "type": "STRING",
                        "enum": ["controlled", "loose", "in_flight", "not_visible"],
                    },
                    "action_confidence": {
                        "type": "STRING",
                        "enum": ["low", "medium", "high"],
                    },
                    "possession_team": {
                        "type": "STRING",
                        "enum": ["white", "green", "unclear"],
                    },
                    "possession_confidence": {
                        "type": "STRING",
                        "enum": ["low", "medium", "high"],
                    },
                    "evidence_notes": {"type": "STRING"},
                },
                "required": [
                    "event_id", "start_seconds", "end_seconds", "cue_seconds",
                    "action_type", "description", "ball_state",
                    "action_confidence", "possession_team",
                    "possession_confidence", "evidence_notes",
                ],
            },
        },
    },
    "required": ["events"],
}
VERIFICATION_RESPONSE_SCHEMA_VERSION = 1


class CommentaryGenerator:

    def __init__(
        self,
        api_key,
        model="gemini-3.5-flash",
    ):

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env."
            )

        # Disable SDK retries so this module owns a bounded total request budget.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        self.model = model

        fallback_setting = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-2.5-flash",
        )

        self.fallback_models = [
            value.strip()
            for value in fallback_setting.split(",")
            if value.strip()
        ]

        self.prompt_builder = VideoCommentaryPrompt()
        self.evidence_builder = EventEvidenceBuilder(
            cell_width=int(
                os.getenv(
                    "EVIDENCE_CELL_WIDTH",
                    "800",
                )
            ),
            jpeg_quality=int(
                os.getenv(
                    "EVIDENCE_JPEG_QUALITY",
                    "80",
                )
            ),
        )

        self.local_verifier = LocalPossessionVerifier(
            image_size=int(
                os.getenv(
                    "LOCAL_VISION_IMAGE_SIZE",
                    "1280",
                )
            )
        )

    def _models_to_try(self):

        models = [self.model]

        for model in self.fallback_models:
            if model not in models:
                models.append(model)

        return models

    def _status_code(self, error):

        code = getattr(error, "code", None)

        if code is None:
            code = getattr(error, "status_code", None)

        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    def _is_retryable_api_error(self, error):

        return self._status_code(error) in RETRYABLE_CODES

    def _wait_before_retry(self, attempt):

        base_delay = min(8.0, 0.5 * (2 ** max(0, attempt - 1)))
        delay = base_delay + random.uniform(
            0.0,
            min(1.0, base_delay * 0.25),
        )

        print(f"Waiting {delay:.1f} seconds before retrying...")
        time.sleep(delay)

    def _is_transient_failure(self, error):

        current = error
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            if isinstance(current, errors.APIError):
                if self._is_retryable_api_error(current):
                    return True
            elif isinstance(
                current,
                (
                    httpx.TransportError,
                    TimeoutError,
                    ConnectionError,
                ),
            ):
                return True

            current = getattr(current, "__cause__", None)

        return False

    def _upload_video(
        self,
        video_path,
        attempts=MAX_MODEL_ATTEMPTS,
    ):

        last_error = None

        for attempt in range(1, attempts + 1):

            try:
                print(
                    "Uploading video to Gemini "
                    f"(attempt {attempt}/{attempts})..."
                )

                uploaded = self.client.files.upload(
                    file=video_path
                )

                print("Video upload completed.")
                return uploaded

            except (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_error = exc
                print(
                    "The connection was interrupted "
                    "during video upload."
                )

            except errors.APIError as exc:

                if not self._is_retryable_api_error(exc):
                    raise

                last_error = exc
                print(
                    "Gemini temporarily rejected "
                    "the video upload."
                )

            if attempt < attempts:
                self._wait_before_retry(attempt)

        raise RuntimeError(
            "The video could not be uploaded after "
            f"{attempts} attempts. Check the internet "
            "connection and try again."
        ) from last_error

    def _file_state(self, uploaded_video):

        state = getattr(uploaded_video, "state", None)
        name = getattr(state, "name", None)

        return str(name or state or "").upper()

    def _wait_until_active(
        self,
        uploaded_video,
        timeout_seconds=180,
    ):

        deadline = time.monotonic() + timeout_seconds
        current = uploaded_video
        status_failures = 0

        while time.monotonic() < deadline:

            state = self._file_state(current)

            if state.endswith("ACTIVE"):
                print(
                    "Gemini finished processing the video."
                )
                return current

            if state.endswith("FAILED"):
                raise RuntimeError(
                    "Gemini failed to process the uploaded video."
                )

            print("Gemini is processing the video...")
            time.sleep(3)

            try:
                current = self.client.files.get(
                    name=current.name
                )
            except errors.APIError as exc:
                if not self._is_retryable_api_error(exc):
                    raise
                status_failures += 1
                if status_failures >= MAX_MODEL_ATTEMPTS:
                    raise RuntimeError(
                        "Gemini upload status remained unavailable after "
                        "bounded retries."
                    ) from exc
                self._wait_before_retry(status_failures)
                continue
            except (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                status_failures += 1
                if status_failures >= MAX_MODEL_ATTEMPTS:
                    raise RuntimeError(
                        "Gemini upload status remained unavailable after "
                        "bounded retries."
                    ) from exc
                self._wait_before_retry(status_failures)
                continue
            else:
                status_failures = 0

        raise TimeoutError(
            "Gemini did not finish processing the "
            f"video within {timeout_seconds} seconds."
        )

    def _json_config(
        self,
        temperature,
        visual=False,
        response_schema=None,
    ):

        config_values = {
            "response_mime_type": "application/json",
            "temperature": float(temperature),
        }

        if response_schema is not None:
            config_values["response_schema"] = response_schema

        if visual:
            config_values["media_resolution"] = (
                types.MediaResolution
                .MEDIA_RESOLUTION_HIGH
            )

        return types.GenerateContentConfig(
            **config_values
        )

    def _generate_json_with_fallback(
        self,
        content,
        purpose,
        temperature,
        visual=False,
        response_validator=None,
        response_schema=None,
    ):

        last_error = None

        for model in self._models_to_try():

            for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):

                try:
                    print(
                        f"{purpose} with {model} "
                        f"(attempt {attempt}/{MAX_MODEL_ATTEMPTS})..."
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=content,
                        config=self._json_config(
                            temperature=temperature,
                            visual=visual,
                            response_schema=response_schema,
                        ),
                    )
                    response_text = response.text
                    if response_validator is not None:
                        try:
                            response_validator(response_text)
                        except (TypeError, ValueError, KeyError) as exc:
                            last_error = exc
                            print(
                                f"{model} returned an invalid {purpose} response; "
                                + verification_response_diagnostics(
                                    response_text,
                                    exc,
                                )
                                + ". Retrying within the configured limit."
                            )
                            if attempt < MAX_MODEL_ATTEMPTS:
                                self._wait_before_retry(attempt)
                            continue

                    print(f"{purpose} succeeded with {model}.")
                    return response_text, model

                except errors.APIError as exc:

                    if not self._is_retryable_api_error(exc):
                        raise

                    last_error = exc

                except (
                    httpx.TransportError,
                    TimeoutError,
                    ConnectionError,
                ) as exc:
                    last_error = exc

                if attempt < MAX_MODEL_ATTEMPTS:
                    print(
                        f"{model} had a transient Gemini failure; "
                        "retrying within the configured limit."
                    )
                    self._wait_before_retry(attempt)

            print(f"{model} exhausted its bounded retry budget.")

        raise RuntimeError(
            "Every configured Gemini model failed to produce a usable response "
            "after bounded retries. Please try again shortly."
        ) from last_error

    def _video_part(self, uploaded_video):

        return types.Part(
            file_data=types.FileData(
                file_uri=uploaded_video.uri,
                mime_type=uploaded_video.mime_type,
            ),
            video_metadata=types.VideoMetadata(
                fps=float(
                    os.getenv(
                        "GEMINI_VIDEO_FPS",
                        "8",
                    )
                )
            ),
        )

    def _draft_events(
        self,
        video_part,
        prompt,
        duration_seconds,
    ):

        content = types.Content(
            role="user",
            parts=[
                video_part,
                types.Part(text=prompt),
            ],
        )

        response_text, model = self._generate_json_with_fallback(
            content,
            purpose="Extracting neutral football actions",
            temperature=0.0,
            visual=True,
        )
        events = parse_analysis_response(
            response_text,
            duration_seconds=duration_seconds,
        )["events"]
        return events, model, response_text

    def _local_possession_signals(self, evidence):

        enabled = os.getenv(
            "ENABLE_LOCAL_POSSESSION_VERIFIER",
            "true",
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        if not enabled:
            print(
                "Local possession verification is disabled; "
                "team names will remain conservative."
            )
            return []

        try:
            print(
                "Running independent local ball-and-shirt "
                "verification..."
            )
            return self.local_verifier.verify(evidence)
        except Exception as exc:
            print(
                "Warning: local possession verification failed. "
                "Team colours will not be spoken without "
                f"independent agreement ({type(exc).__name__})."
            )
            return []

    def _verify_events(
        self,
        video_part,
        draft_events,
        evidence,
        duration_seconds,
    ):

        parts = [video_part]

        for item in evidence:
            parts.append(
                types.Part(
                    text=(
                        "Timestamped evidence sheet for "
                        f"event_id {item['event_id']}."
                    )
                )
            )
            parts.append(
                types.Part.from_bytes(
                    data=item["jpeg_bytes"],
                    mime_type="image/jpeg",
                )
            )

        parts.append(
            types.Part(
                text=self.prompt_builder.build_verification(
                    draft_events
                )
            )
        )

        content = types.Content(
            role="user",
            parts=parts,
        )

        response_text, model = self._generate_json_with_fallback(
            content,
            purpose="Fact-checking actions and possession",
            temperature=0.0,
            visual=True,
            response_schema=VERIFICATION_RESPONSE_SCHEMA,
            response_validator=lambda text: parse_verification_response(
                text,
                draft_events=draft_events,
                duration_seconds=duration_seconds,
            ),
        )
        events = parse_verification_response(
            response_text,
            draft_events=draft_events,
            duration_seconds=duration_seconds,
        )["events"]
        return events, model, response_text

    def _write_commentary(
        self,
        verified_events,
        duration_seconds,
    ):

        prompt = self.prompt_builder.build_commentary(
            verified_events
        )

        response_text, model = self._generate_json_with_fallback(
            prompt,
            purpose="Writing grounded bilingual commentary",
            temperature=0.35,
            visual=False,
        )
        result = parse_commentary_response(
            response_text,
            verified_events=verified_events,
            duration_seconds=duration_seconds,
        )
        return result, model, response_text

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

    def _cache_identity(self, prompt, duration_seconds):

        enabled = os.getenv(
            "ENABLE_LOCAL_POSSESSION_VERIFIER",
            "true",
        ).strip().lower() not in {"0", "false", "no", "off"}

        return {
            "schema_version": STAGE_CACHE_SCHEMA_VERSION,
            "models": self._models_to_try(),
            "duration_seconds": float(duration_seconds),
            "analysis_prompt": prompt,
            "video_fps": os.getenv("GEMINI_VIDEO_FPS", "8"),
            "json_config": {
                "analysis_temperature": 0.0,
                "verification_temperature": 0.0,
                "commentary_temperature": 0.35,
                "visual_media_resolution": "high",
                "response_mime_type": "application/json",
                "sdk_retry_attempts": 1,
                "outer_attempts_per_model": MAX_MODEL_ATTEMPTS,
            },
            "evidence_builder": {
                "cell_width": self.evidence_builder.cell_width,
                "jpeg_quality": self.evidence_builder.jpeg_quality,
            },
            "local_verifier": {
                "enabled": enabled,
                "image_size": getattr(self.local_verifier, "image_size", None),
                "version": 1,
            },
            "max_team_mentions": int(
                os.getenv("MAX_TEAM_MENTIONS", "4")
            ),
            "parser_version": 1,
        }

    def _serialize_evidence(self, evidence):

        return [
            {
                "event_id": int(item["event_id"]),
                "cue_seconds": float(item["cue_seconds"]),
                "timestamps": [
                    float(value) for value in item["timestamps"]
                ],
                "jpeg_base64": base64.b64encode(
                    item["jpeg_bytes"]
                ).decode("ascii"),
            }
            for item in evidence
        ]

    def _restore_evidence(self, cached):

        if not isinstance(cached, list):
            raise ValueError("Cached evidence must be a list.")

        restored = []
        for item in cached:
            if not isinstance(item, dict):
                raise ValueError("Cached evidence item is malformed.")
            encoded = item.get("jpeg_base64")
            if not isinstance(encoded, str):
                raise ValueError("Cached evidence image is missing.")
            restored.append({
                "event_id": int(item["event_id"]),
                "cue_seconds": float(item["cue_seconds"]),
                "timestamps": [
                    float(value) for value in item["timestamps"]
                ],
                "jpeg_bytes": base64.b64decode(encoded, validate=True),
            })

        return restored

    def _evidence_digest(self, evidence):

        digest = hashlib.sha256()
        for item in evidence:
            digest.update(str(int(item["event_id"])).encode("ascii"))
            digest.update(b"\\0")
            digest.update(item["jpeg_bytes"])
            digest.update(b"\\0")
        return digest.hexdigest()

    def _cached_response(self, cache, stage, context):

        value = cache.get_stage(stage, context)
        if not isinstance(value, dict):
            return None

        model = value.get("model")
        response_text = value.get("response_text")
        if (
            model not in self._models_to_try()
            or not isinstance(response_text, str)
            or not response_text.strip()
        ):
            return None

        return response_text, model

    def _delete_uploaded_video(self, uploaded_video):

        name = getattr(uploaded_video, "name", None)
        if not name:
            return True

        try:
            self.client.files.delete(name=name)
            return True
        except errors.APIError as exc:
            if self._status_code(exc) == 404:
                return True
            print(
                "Warning: could not delete Gemini upload "
                f"({type(exc).__name__})."
            )
            return False
        except Exception as exc:
            # Provider exception text may contain request metadata; keep logs safe.
            print(
                "Warning: could not delete Gemini upload "
                f"({type(exc).__name__})."
            )
            return False

    def _get_or_upload_video(self, cache, video_path):

        saved = cache.get_upload()
        if saved:
            age = time.time() - float(saved.get("saved_at", 0.0))
            if age >= 0 and age < UPLOAD_CACHE_MAX_AGE_SECONDS:
                try:
                    existing = self.client.files.get(name=saved["name"])
                    state = self._file_state(existing)
                    if state.endswith("ACTIVE"):
                        print("Reusing the active Gemini video upload.")
                        return existing
                    if not state.endswith("FAILED"):
                        return self._wait_until_active(existing)
                    if not self._delete_uploaded_video(existing):
                        raise RuntimeError(
                            "Could not clean up a failed Gemini upload; "
                            "retry after provider access is restored."
                        )
                except errors.APIError as exc:
                    if self._is_retryable_api_error(exc):
                        raise
                    if self._status_code(exc) != 404:
                        raise
                except (
                    httpx.TransportError,
                    TimeoutError,
                    ConnectionError,
                ):
                    # Keep the cached pointer and retry the lookup later.
                    raise
            else:
                deleted = self._delete_uploaded_video(
                    SimpleNamespace(name=saved.get("name"))
                )
                if not deleted:
                    raise RuntimeError(
                        "Could not clean up an expired Gemini upload; "
                        "retry after provider access is restored."
                    )
            cache.clear_upload()

        uploaded = self._upload_video(video_path)
        try:
            cache.set_upload({
                "name": getattr(uploaded, "name", ""),
                "uri": getattr(uploaded, "uri", ""),
                "mime_type": getattr(uploaded, "mime_type", ""),
                "saved_at": time.time(),
            })
            active = self._wait_until_active(uploaded)
            cache.set_upload({
                "name": getattr(active, "name", ""),
                "uri": getattr(active, "uri", ""),
                "mime_type": getattr(active, "mime_type", ""),
                "saved_at": time.time(),
            })
            return active
        except Exception:
            if self._delete_uploaded_video(uploaded):
                cache.clear_upload()
            raise

    def generate_from_video(
        self,
        video_path,
        prompt,
        duration_seconds=None,
    ):

        if duration_seconds is None:
            raise ValueError(
                "duration_seconds is required for video commentary."
            )

        cache = StageCache(
            video_path,
            identity=self._cache_identity(prompt, duration_seconds),
        )
        uploaded_video = None
        failure = None
        verification_completed = False

        try:
            draft_context = {
                "prompt": prompt,
                "temperature": 0.0,
                "visual": True,
            }
            cached_draft = self._cached_response(
                cache,
                "draft",
                draft_context,
            )

            if cached_draft is None:
                uploaded_video = self._get_or_upload_video(cache, video_path)
                draft_events, draft_model, draft_text = self._draft_events(
                    self._video_part(uploaded_video),
                    prompt,
                    duration_seconds,
                )
                cache.set_stage(
                    "draft",
                    draft_context,
                    {"model": draft_model, "response_text": draft_text},
                )
            else:
                draft_text, draft_model = cached_draft
                try:
                    draft_events = parse_analysis_response(
                        draft_text,
                        duration_seconds=duration_seconds,
                    )["events"]
                except (TypeError, ValueError, KeyError):
                    cache.clear_stage("draft")
                    uploaded_video = self._get_or_upload_video(cache, video_path)
                    draft_events, draft_model, draft_text = self._draft_events(
                        self._video_part(uploaded_video),
                        prompt,
                        duration_seconds,
                    )
                    cache.set_stage(
                        "draft",
                        draft_context,
                        {
                            "model": draft_model,
                            "response_text": draft_text,
                        },
                    )

            evidence_context = {
                "draft_events": draft_events,
                "builder": {
                    "cell_width": self.evidence_builder.cell_width,
                    "jpeg_quality": self.evidence_builder.jpeg_quality,
                },
            }
            try:
                evidence = self._restore_evidence(
                    cache.get_stage("evidence", evidence_context)
                )
            except (TypeError, ValueError, KeyError):
                evidence = None

            if evidence is None:
                evidence = self.evidence_builder.build(
                    video_path,
                    draft_events,
                    duration_seconds,
                )
                cache.set_stage(
                    "evidence",
                    evidence_context,
                    self._serialize_evidence(evidence),
                )

            evidence_digest = self._evidence_digest(evidence)
            local_context = {
                "evidence_digest": evidence_digest,
                "enabled": os.getenv(
                    "ENABLE_LOCAL_POSSESSION_VERIFIER",
                    "true",
                ).strip().lower() not in {"0", "false", "no", "off"},
                "image_size": getattr(self.local_verifier, "image_size", None),
                "version": 1,
            }
            local_signals = cache.get_stage("local_signals", local_context)
            if not isinstance(local_signals, list):
                local_signals = self._local_possession_signals(evidence)
                cache.set_stage(
                    "local_signals",
                    local_context,
                    local_signals,
                )

            verification_prompt = self.prompt_builder.build_verification(
                draft_events
            )
            verification_context = {
                "prompt": verification_prompt,
                "evidence_digest": evidence_digest,
                "temperature": 0.0,
                "visual": True,
                "response_schema_version": VERIFICATION_RESPONSE_SCHEMA_VERSION,
                "parser_version": 2,
            }
            cached_verification = self._cached_response(
                cache,
                "verification",
                verification_context,
            )

            if cached_verification is None:
                if uploaded_video is None:
                    uploaded_video = self._get_or_upload_video(cache, video_path)
                verified_events, verifier_model, verification_text = (
                    self._verify_events(
                        self._video_part(uploaded_video),
                        draft_events,
                        evidence,
                        duration_seconds,
                    )
                )
                cache.set_stage(
                    "verification",
                    verification_context,
                    {
                        "model": verifier_model,
                        "response_text": verification_text,
                    },
                )
            else:
                verification_text, verifier_model = cached_verification
                try:
                    verified_events = parse_verification_response(
                        verification_text,
                        draft_events=draft_events,
                        duration_seconds=duration_seconds,
                    )["events"]
                except (TypeError, ValueError, KeyError) as exc:
                    print(
                        "Cached Gemini verification response rejected: "
                        + verification_response_diagnostics(
                            verification_text,
                            exc,
                        )
                    )
                    cache.clear_stage("verification")
                    if uploaded_video is None:
                        uploaded_video = self._get_or_upload_video(cache, video_path)
                    verified_events, verifier_model, verification_text = (
                        self._verify_events(
                            self._video_part(uploaded_video),
                            draft_events,
                            evidence,
                            duration_seconds,
                        )
                    )
                    cache.set_stage(
                        "verification",
                        verification_context,
                        {
                            "model": verifier_model,
                            "response_text": verification_text,
                        },
                    )

            verification_completed = True
            verified_events = reconcile_possession(
                verified_events,
                local_signals,
            )
            verified_events = mark_team_mentions(
                verified_events,
                max_mentions=int(os.getenv("MAX_TEAM_MENTIONS", "4")),
            )

            for item in evidence:
                item.pop("frames", None)

            commentary_prompt = self.prompt_builder.build_commentary(
                verified_events
            )
            commentary_context = {
                "prompt": commentary_prompt,
                "temperature": 0.35,
                "visual": False,
            }
            cached_commentary = self._cached_response(
                cache,
                "commentary",
                commentary_context,
            )

            if cached_commentary is None:
                commentary_result, writer_model, commentary_text = (
                    self._write_commentary(
                        verified_events,
                        duration_seconds,
                    )
                )
                cache.set_stage(
                    "commentary",
                    commentary_context,
                    {
                        "model": writer_model,
                        "response_text": commentary_text,
                    },
                )
            else:
                commentary_text, writer_model = cached_commentary
                try:
                    commentary_result = parse_commentary_response(
                        commentary_text,
                        verified_events=verified_events,
                        duration_seconds=duration_seconds,
                    )
                except (TypeError, ValueError, KeyError):
                    cache.clear_stage("commentary")
                    commentary_result, writer_model, commentary_text = (
                        self._write_commentary(
                            verified_events,
                            duration_seconds,
                        )
                    )
                    cache.set_stage(
                        "commentary",
                        commentary_context,
                        {
                            "model": writer_model,
                            "response_text": commentary_text,
                        },
                    )

            return commentary_result

        except Exception as exc:
            failure = exc
            raise

        finally:
            if uploaded_video is not None:
                if (
                    failure is not None
                    and self._is_transient_failure(failure)
                    and not verification_completed
                ):
                    # Keep a completed upload only for an unfinished provider
                    # stage; the next run revalidates that the upload is active.
                    pass
                else:
                    if self._delete_uploaded_video(uploaded_video):
                        cache.clear_upload()
            elif (
                not (
                    failure is not None
                    and self._is_transient_failure(failure)
                    and not verification_completed
                )
                and cache.get_upload()
            ):
                saved_upload = cache.get_upload()
                if self._delete_uploaded_video(
                    SimpleNamespace(name=saved_upload.get("name"))
                ):
                    cache.clear_upload()
