import os
import random
import time

import httpx
from google import genai
from google.genai import errors, types

from llm.evidence_builder import EventEvidenceBuilder
from llm.response_parser import (
    parse_analysis_response,
    parse_commentary_response,
    parse_verification_response,
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

        self.client = genai.Client(api_key=api_key)
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

        delay = min(30, 2 ** attempt)
        delay += random.uniform(0.0, 1.0)

        print(
            f"Waiting {delay:.1f} seconds before retrying..."
        )

        time.sleep(delay)

    def _upload_video(
        self,
        video_path,
        attempts=5,
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
            except (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ):
                print(
                    "Status check was interrupted; trying again..."
                )

        raise TimeoutError(
            "Gemini did not finish processing the "
            f"video within {timeout_seconds} seconds."
        )

    def _json_config(
        self,
        temperature,
        visual=False,
    ):

        config_values = {
            "response_mime_type": "application/json",
            "temperature": float(temperature),
        }

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
    ):

        last_error = None

        for model in self._models_to_try():

            try:
                print(f"{purpose} with {model}...")

                response = self.client.models.generate_content(
                    model=model,
                    contents=content,
                    config=self._json_config(
                        temperature=temperature,
                        visual=visual,
                    ),
                )

                print(f"{purpose} succeeded with {model}.")
                return response.text

            except errors.APIError as exc:

                if not self._is_retryable_api_error(exc):
                    raise

                last_error = exc
                print(f"{model} is temporarily unavailable.")

            except (
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_error = exc
                print(
                    f"The connection to {model} was interrupted."
                )

        raise RuntimeError(
            "Every configured Gemini model is currently "
            "unavailable. Please try again shortly."
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

        response_text = self._generate_json_with_fallback(
            content,
            purpose="Extracting neutral football actions",
            temperature=0.0,
            visual=True,
        )

        return parse_analysis_response(
            response_text,
            duration_seconds=duration_seconds,
        )["events"]

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
                f"independent agreement. Details: {exc}"
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

        response_text = self._generate_json_with_fallback(
            content,
            purpose="Fact-checking actions and possession",
            temperature=0.0,
            visual=True,
        )

        return parse_verification_response(
            response_text,
            draft_events=draft_events,
            duration_seconds=duration_seconds,
        )["events"]

    def _write_commentary(
        self,
        verified_events,
        duration_seconds,
    ):

        prompt = self.prompt_builder.build_commentary(
            verified_events
        )

        response_text = self._generate_json_with_fallback(
            prompt,
            purpose="Writing grounded bilingual commentary",
            temperature=0.35,
            visual=False,
        )

        return parse_commentary_response(
            response_text,
            verified_events=verified_events,
            duration_seconds=duration_seconds,
        )

    def generate(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text

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

        uploaded_video = None

        try:
            uploaded_video = self._upload_video(video_path)
            uploaded_video = self._wait_until_active(
                uploaded_video
            )

            video_part = self._video_part(uploaded_video)

            draft_events = self._draft_events(
                video_part,
                prompt,
                duration_seconds,
            )

            evidence = self.evidence_builder.build(
                video_path,
                draft_events,
                duration_seconds,
            )

            local_signals = (
                self._local_possession_signals(evidence)
            )

            verified_events = self._verify_events(
                video_part,
                draft_events,
                evidence,
                duration_seconds,
            )

            verified_events = reconcile_possession(
                verified_events,
                local_signals,
            )

            verified_events = mark_team_mentions(
                verified_events,
                max_mentions=int(
                    os.getenv(
                        "MAX_TEAM_MENTIONS",
                        "4",
                    )
                ),
            )

            # Release the OpenCV images before TTS starts.
            for item in evidence:
                item.pop("frames", None)

            return self._write_commentary(
                verified_events,
                duration_seconds,
            )

        finally:
            if (
                uploaded_video is not None
                and getattr(uploaded_video, "name", None)
            ):
                try:
                    self.client.files.delete(
                        name=uploaded_video.name
                    )
                except Exception as exc:
                    print(
                        "Warning: could not delete Gemini upload: "
                        f"{exc}"
                    )
