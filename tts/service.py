import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from llm.response_parser import grounded_fallback_lines
from tts.generator import TTSGenerator
from utils.video_metadata import get_media_duration, has_audio_stream
from video.merger import VideoMerger


DELIVERY_TAGS = {
    "focused": "[focused]",
    "building": "[building excitement]",
    "excited": "[excited]",
    "shouting": "[shouts]",
}


CACHE_VERSION = 1
MAX_FIT_CANDIDATES = 2


def build_tts_text(event, language):

    line = event[f"{language}_tts"].strip()

    delivery = event.get(
        "delivery",
        "focused",
    ).lower()

    tag = DELIVERY_TAGS.get(
        delivery,
        "[focused]",
    )

    return f"{tag} {line}"


def calculate_scheduled_starts(events, video_duration=None):

    starts = []

    for event in events:

        try:
            cue = float(event["cue_seconds"])
            event_end = float(event["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Event {len(starts)} has an invalid cue or end timestamp."
            ) from exc

        if not math.isfinite(cue) or cue < 0:
            raise ValueError(
                f"Event {len(starts)} has invalid cue_seconds={cue!r}; "
                "cues must be finite and non-negative."
            )
        if not math.isfinite(event_end) or event_end <= cue:
            raise ValueError(
                f"Event {len(starts)} has invalid end_seconds={event_end!r}; "
                "event end must be finite and after its cue."
            )
        if video_duration is not None and cue > video_duration:
            raise ValueError(
                f"Event {len(starts)} cue {cue:.3f}s exceeds video duration "
                f"{video_duration:.3f}s."
            )
        if starts and cue <= starts[-1]:
            raise ValueError(
                f"Event {len(starts)} cue {cue:.3f}s is not after the "
                f"previous cue {starts[-1]:.3f}s."
            )
        starts.append(cue)

    return starts


def _fit_candidates(event, language):
    primary = event[f"{language}_tts"].strip()
    candidates = [primary]

    # Writer-provided neutral lines are not semantically checked by the parser.
    # Use only the parser's action-specific fallback after verified-context checks.
    fallback_lines = grounded_fallback_lines(event)
    if fallback_lines is not None:
        fallback = fallback_lines[0 if language == "english" else 1]
        is_shorter = (
            len(fallback.split()) < len(primary.split())
            or len(fallback) < len(primary)
        )
        if fallback != primary and is_shorter:
            candidates.append(fallback)

    return candidates[:MAX_FIT_CANDIDATES]


def _cache_paths(segment_directory, language, index, identity):
    encoded_identity = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(encoded_identity.encode("utf-8")).hexdigest()
    stem = f"{language}_{index:03d}_{fingerprint[:20]}"
    return (
        segment_directory / f"{stem}.mp3",
        segment_directory / f"{stem}.json",
    )


def _invalidate_cache(audio_path, metadata_path):
    for path in (audio_path, metadata_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _audio_sha256(audio_path):
    digest = hashlib.sha256()
    with audio_path.open("rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cached_audio(audio_path, metadata_path, identity):
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("cache_version") != CACHE_VERSION
            or metadata.get("identity") != identity
            or not audio_path.is_file()
            or audio_path.stat().st_size <= 0
            or metadata.get("size_bytes") != audio_path.stat().st_size
            or metadata.get("audio_sha256") != _audio_sha256(audio_path)
        ):
            raise ValueError("cache metadata mismatch")
        if not has_audio_stream(str(audio_path)):
            raise ValueError("cached file has no audio stream")
        duration = get_media_duration(str(audio_path))
        cached_duration = metadata.get("duration_seconds")
        if (
            isinstance(cached_duration, bool)
            or not isinstance(cached_duration, (int, float))
            or not math.isfinite(float(cached_duration))
            or not math.isfinite(duration)
            or duration <= 0
            or abs(duration - float(cached_duration)) > 0.03
        ):
            raise ValueError("cached audio duration mismatch")
        return duration
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        _invalidate_cache(audio_path, metadata_path)
        return None


def _write_cache_metadata(metadata_path, identity, duration, audio_path):
    metadata = {
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "duration_seconds": duration,
        "size_bytes": audio_path.stat().st_size,
        "audio_sha256": _audio_sha256(audio_path),
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=metadata_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
            json.dump(metadata, temporary_file, ensure_ascii=False, sort_keys=True)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, metadata_path)
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


class TTSService:

    def generate(self, events, video_duration=None):

        generator = TTSGenerator()

        segment_directory = Path(
            "media/output/commentary_segments"
        )

        segment_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if video_duration is not None:
            try:
                video_duration = float(video_duration)
            except (TypeError, ValueError) as exc:
                raise ValueError("video_duration must be a finite positive number.") from exc
            if not math.isfinite(video_duration) or video_duration <= 0:
                raise ValueError("video_duration must be a finite positive number.")

        scheduled_starts = calculate_scheduled_starts(events, video_duration)

        generated = {
            "english": [],
            "arabic": [],
        }

        for language in ("english", "arabic"):

            plain_lines = [
                event[f"{language}_tts"].strip()
                for event in events
            ]

            for index, event in enumerate(events):

                previous_text = (
                    plain_lines[index - 1]
                    if index > 0
                    else None
                )

                next_text = (
                    plain_lines[index + 1]
                    if index + 1 < len(events)
                    else None
                )

                start = scheduled_starts[index]

                if index + 1 < len(events):
                    end = scheduled_starts[index + 1]
                else:
                    end = float(
                        event["end_seconds"]
                    )

                if video_duration is not None:
                    end = min(end, video_duration)

                available = end - start
                if not math.isfinite(end) or available <= 0:
                    raise ValueError(
                        f"{language} event {index} has no valid narration "
                        f"window: {start:.3f}s to {end!r}s."
                    )
                fit_available = available - VideoMerger.END_MARGIN_SECONDS
                if fit_available <= 0:
                    raise ValueError(
                        f"{language} event {index} has no usable narration "
                        f"time after the {VideoMerger.END_MARGIN_SECONDS:.3f}s "
                        "rendering margin."
                    )

                selected_path = None
                selected_duration = None
                selected_line = None
                last_fit_error = None
                attempt_count = 0

                for line in _fit_candidates(event, language):
                    attempt_count += 1
                    speech_text = build_tts_text(
                        {**event, f"{language}_tts": line},
                        language,
                    )
                    cache_identity = getattr(generator, "cache_identity", None)
                    identity = (
                        cache_identity(
                            speech_text,
                            language,
                            previous_text=previous_text,
                            next_text=next_text,
                        )
                        if callable(cache_identity)
                        else None
                    )

                    if identity is not None:
                        output_path, metadata_path = _cache_paths(
                            segment_directory,
                            language,
                            index,
                            identity,
                        )
                        duration = _load_cached_audio(
                            output_path,
                            metadata_path,
                            identity,
                        )
                    else:
                        output_path = segment_directory / f"{language}_{index:03d}.mp3"
                        metadata_path = None
                        duration = None

                    if duration is None:
                        temporary_path = segment_directory / (
                            f".{language}_{index:03d}.{uuid.uuid4().hex}.tmp.mp3"
                        )
                        try:
                            reported_duration = generator.generate(
                                speech_text,
                                str(temporary_path),
                                language,
                                previous_text=previous_text,
                                next_text=next_text,
                            )
                            if (
                                isinstance(reported_duration, bool)
                                or not isinstance(reported_duration, (int, float))
                                or not math.isfinite(float(reported_duration))
                                or reported_duration <= 0
                            ):
                                raise ValueError(
                                    f"{language} event {index} returned invalid audio duration."
                                )
                            if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
                                raise ValueError(
                                    f"{language} event {index} did not produce a complete MP3 file."
                                )
                            if not has_audio_stream(str(temporary_path)):
                                raise ValueError(
                                    f"{language} event {index} MP3 has no audio stream."
                                )
                            duration = get_media_duration(str(temporary_path))
                            if not math.isfinite(duration) or duration <= 0:
                                raise ValueError(
                                    f"{language} event {index} MP3 has an invalid probed duration."
                                )
                            if identity is None:
                                output_path = segment_directory / f"{language}_{index:03d}.mp3"
                            os.replace(temporary_path, output_path)
                            if identity is not None:
                                _write_cache_metadata(
                                    metadata_path,
                                    identity,
                                    duration,
                                    output_path,
                                )
                        finally:
                            try:
                                temporary_path.unlink(missing_ok=True)
                            except OSError:
                                pass

                    max_fittable_duration = fit_available * VideoMerger.MAX_TEMPO
                    if duration > max_fittable_duration:
                        last_fit_error = (
                            f"duration={duration:.3f}s, "
                            f"available={fit_available:.3f}s, "
                            f"required_tempo={duration / fit_available:.3f}, "
                            f"maximum_tempo={VideoMerger.MAX_TEMPO:.2f}"
                        )
                        continue

                    selected_path = output_path
                    selected_duration = duration
                    selected_line = line
                    break

                if selected_path is None:
                    raise ValueError(
                        f"{language} event {index} could not fit after "
                        f"{attempt_count} bounded phrasing attempts "
                        f"(limit {MAX_FIT_CANDIDATES}): "
                        f"window={start:.3f}s-{end:.3f}s, "
                        f"{last_fit_error or 'no grounded shorter phrasing was available'}."
                    )

                # Keep the written script identical to the text chosen for audio.
                event[f"{language}_tts"] = selected_line
                plain_lines[index] = selected_line

                generated[language].append({
                    "path": str(selected_path),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration": selected_duration,
                    "text": build_tts_text(
                        {**event, f"{language}_tts": selected_line},
                        language,
                    ),
                })

                print(
                    f"{language} segment {index}: "
                    f"cue={event['cue_seconds']:.2f}s, "
                    f"play={start:.2f}s, "
                    f"window_end={end:.2f}s"
                )

        return generated
