"""Re-merge validated, locally cached commentary audio without provider calls."""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from llm.response_parser import (
    parse_analysis_response,
    parse_commentary_response,
    parse_verification_response,
)
from tts.service import (
    CACHE_VERSION as AUDIO_CACHE_VERSION,
    DELIVERY_TAGS,
    _fit_candidates,
    build_tts_text,
    calculate_scheduled_starts,
)
from tts.text import strip_audio_tags
from utils.video_metadata import get_media_duration, has_audio_stream
from video.merger import VideoMerger
from vision.possession_verifier import mark_team_mentions, reconcile_possession


STAGE_NAMES = ("draft", "evidence", "local_signals", "verification", "commentary")
IDENTITY_KEYS = {
    "context", "delivery_tags", "model", "provider", "schema_version",
    "settings", "synthesis_text", "text", "voice",
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_PREFIX_PATTERN = re.compile(r"^(english|arabic)_(\d{3})_([0-9a-f]{20})$")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def _read_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} as valid JSON: {path}") from exc


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _load_stage_cache(video_path, cache_dir):
    fingerprint = _sha256(video_path)
    cache_path = Path(cache_dir) / f"{fingerprint}.json"
    cache = _read_json(cache_path, "Gemini stage cache")
    if (
        not isinstance(cache, dict)
        or type(cache.get("schema_version")) is not int
        or cache.get("schema_version") != 1
    ):
        raise ValueError("Gemini cache root or schema version is invalid.")
    if cache.get("video_fingerprint") != fingerprint:
        raise ValueError("Gemini cache does not belong to the selected video.")
    if not isinstance(cache.get("identity_digest"), str) or not HASH_PATTERN.fullmatch(
        cache["identity_digest"]
    ):
        raise ValueError("Gemini cache identity digest is invalid.")

    stages = cache.get("stages")
    if not isinstance(stages, dict) or any(name not in stages for name in STAGE_NAMES):
        raise ValueError("Gemini cache is missing one or more required stages.")

    values = {}
    for name in STAGE_NAMES:
        record = stages[name]
        if not isinstance(record, dict):
            raise ValueError(f"Gemini stage {name} is malformed.")
        digest = record.get("context_digest")
        value = record.get("value")
        if not isinstance(digest, str) or not HASH_PATTERN.fullmatch(digest):
            raise ValueError(f"Gemini stage {name} context digest is invalid.")
        if not isinstance(value, (dict, list)):
            raise ValueError(f"Gemini stage {name} value has an unsupported shape.")
        values[name] = value
    return cache_path, values


def _stage_response(stages, name):
    value = stages[name]
    if not isinstance(value, dict):
        raise ValueError(f"Gemini stage {name} must contain a response record.")
    model = value.get("model")
    response_text = value.get("response_text")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Gemini stage {name} model is missing.")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError(f"Gemini stage {name} response text is missing.")
    return response_text


def _validate_event_ids(events, label):
    if not isinstance(events, list) or not events:
        raise ValueError(f"{label} must contain at least one event.")
    ids = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or type(event.get("event_id")) is not int:
            raise ValueError(f"{label} event {index} has no integer event_id.")
        ids.append(event["event_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate event IDs.")
    if ids != list(range(len(events))):
        raise ValueError(f"{label} event IDs must be ordered and contiguous from zero.")
    return ids


def _validate_evidence(stage_value, draft_events):
    if not isinstance(stage_value, list):
        raise ValueError("Gemini evidence stage must be a list.")
    expected_ids = [event["event_id"] for event in draft_events]
    found = {}
    for index, evidence in enumerate(stage_value):
        if not isinstance(evidence, dict) or type(evidence.get("event_id")) is not int:
            raise ValueError(f"Gemini evidence item {index} is malformed.")
        event_id = evidence["event_id"]
        if event_id not in expected_ids or event_id in found:
            raise ValueError("Gemini evidence has an unexpected or duplicate event ID.")
        cue = _finite_number(evidence.get("cue_seconds"), "Evidence cue_seconds")
        timestamps = evidence.get("timestamps")
        encoded = evidence.get("jpeg_base64")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError(f"Gemini evidence event {event_id} has no timestamps.")
        if any(not math.isfinite(_finite_number(t, "Evidence timestamp")) for t in timestamps):
            raise ValueError(f"Gemini evidence event {event_id} has an invalid timestamp.")
        try:
            image_bytes = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else b""
        except ValueError as exc:
            raise ValueError(f"Gemini evidence event {event_id} has invalid image encoding.") from exc
        if not image_bytes.startswith(b"\xff\xd8") or not image_bytes.endswith(b"\xff\xd9"):
            raise ValueError(f"Gemini evidence event {event_id} has no valid cached image.")
        found[event_id] = cue
    if set(found) != set(expected_ids):
        raise ValueError("Gemini evidence does not cover every draft event exactly once.")
    for event in draft_events:
        if not math.isclose(found[event["event_id"]], event["cue_seconds"], abs_tol=1e-9):
            raise ValueError(f"Gemini evidence cue differs from draft event {event['event_id']}.")


def _validate_local_signals(stage_value, event_ids):
    if not isinstance(stage_value, list):
        raise ValueError("Gemini local_signals stage must be a list.")
    if not stage_value:
        return
    found = set()
    for index, signal in enumerate(stage_value):
        if not isinstance(signal, dict) or type(signal.get("event_id")) is not int:
            raise ValueError(f"Local signal {index} is malformed.")
        event_id = signal["event_id"]
        if event_id not in event_ids or event_id in found:
            raise ValueError("Local signals contain an unexpected or duplicate event ID.")
        if signal.get("possession_team") not in {"white", "green", "unclear"}:
            raise ValueError(f"Local signal {event_id} has an invalid possession team.")
        if signal.get("confidence") not in {"low", "medium", "high"}:
            raise ValueError(f"Local signal {event_id} has an invalid confidence.")
        found.add(event_id)
    if found != set(event_ids):
        raise ValueError("Local signals must cover every event or be entirely empty.")


def _strict_commentary_events(response_text, expected_ids):
    try:
        payload = json.loads(response_text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Cached commentary response is not valid JSON.") from exc
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise ValueError("Cached commentary response must be an object with only events.")
    raw_events = payload["events"]
    if not isinstance(raw_events, list):
        raise ValueError("Cached commentary events must be a list.")
    ids = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, dict) or type(event.get("event_id")) is not int:
            raise ValueError(f"Commentary event {index} has no integer event_id.")
        ids.append(event["event_id"])
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Commentary event IDs are duplicate, missing, or unexpected.")
    return raw_events


def _read_script(path, language, events):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read the {language} TTS script: {path}") from exc
    lines = text.splitlines()
    if len(lines) != len(events) or any(not line.strip() for line in lines):
        raise ValueError(f"The {language} TTS script must have exactly one non-empty line per event.")

    scripts = {}
    event_by_id = {event["event_id"]: event for event in events}
    for index, line in enumerate(lines):
        event = event_by_id[index]
        valid_lines = {
            build_tts_text({**event, f"{language}_tts": candidate}, language)
            for candidate in _fit_candidates(event, language)
        }
        if line not in valid_lines:
            raise ValueError(
                f"The {language} script line for event {index} does not match "
                "a validated commentary phrasing and delivery tag."
            )
        scripts[index] = line
    return scripts


def _validate_identity(identity, expected_text, filename_language, filename_index):
    if not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS:
        raise ValueError("Cached audio identity fields are missing or unsupported.")
    if (
        type(identity.get("schema_version")) is not int
        or identity.get("schema_version") != 1
        or identity.get("provider") != "elevenlabs"
    ):
        raise ValueError("Cached audio is not a supported ElevenLabs identity.")
    for field in ("voice", "model"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"Cached audio identity has no valid {field}.")
    settings = identity.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Cached audio synthesis settings are missing.")
    if settings.get("output_format") != "mp3_44100_128":
        raise ValueError("Cached audio output format is unsupported.")
    voice_settings = settings.get("voice_settings")
    if not isinstance(voice_settings, dict):
        raise ValueError("Cached audio voice settings are missing.")
    expected_voice_keys = (
        {"stability"} if identity["model"] == "eleven_v3"
        else {"stability", "similarity_boost", "style", "use_speaker_boost", "speed"}
    )
    if set(settings) != {"output_format", "voice_settings"} or set(voice_settings) != expected_voice_keys:
        raise ValueError("Cached audio synthesis settings have an unsupported shape.")
    for key, value in voice_settings.items():
        if key == "use_speaker_boost":
            if type(value) is not bool:
                raise ValueError("Cached audio speaker boost setting is invalid.")
        else:
            number = _finite_number(value, f"Cached voice setting {key}")
            if number < 0 or number > 1.5:
                raise ValueError(f"Cached voice setting {key} is out of range.")
    context = identity.get("context")
    if not isinstance(context, dict) or set(context) != {
        "previous_text", "next_text",
    }:
        raise ValueError("Cached audio context identity is malformed.")
    if any(value is not None and not isinstance(value, str) for value in context.values()):
        raise ValueError("Cached audio context text is malformed.")
    if identity["model"] == "eleven_v3" and identity["context"] != {
        "previous_text": None, "next_text": None,
    }:
        raise ValueError("Eleven v3 cache context does not match generator semantics.")
    if identity.get("text") != expected_text:
        raise ValueError("Cached audio text does not exactly match the saved script line.")
    match = re.match(r"^(\[[^\[\]\r\n]+\]) ", expected_text)
    if not match or match.group(1) not in DELIVERY_TAGS.values():
        raise ValueError("Saved TTS script line has an unsupported delivery tag.")
    tag_matches = re.findall(r"\[[^\[\]\r\n]+\]", expected_text)
    if identity.get("delivery_tags") != tag_matches or tag_matches != [match.group(1)]:
        raise ValueError("Cached audio delivery tags do not exactly match the script.")
    expected_synthesis = (
        expected_text if identity["model"] == "eleven_v3"
        else strip_audio_tags(expected_text)
    )
    if identity.get("synthesis_text") != expected_synthesis:
        raise ValueError("Cached audio synthesis text does not match generator semantics.")
    return match.group(1)


def _validate_audio_metadata(
    metadata_path, audio_path, expected_text, language, index, expected_context,
):
    metadata = _read_json(metadata_path, "audio cache metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Audio metadata is malformed: {metadata_path}")
    if (
        type(metadata.get("cache_version")) is not int
        or metadata.get("cache_version") != AUDIO_CACHE_VERSION
    ):
        raise ValueError(f"Unsupported audio cache version: {metadata_path}")
    identity = metadata.get("identity")
    tag = _validate_identity(identity, expected_text, language, index)
    if identity["model"] != "eleven_v3" and identity["context"] != expected_context:
        raise ValueError("Cached audio context does not match adjacent narration text.")

    encoded_identity = _canonical_json(identity).encode("utf-8")
    identity_digest = hashlib.sha256(encoded_identity).hexdigest()
    expected_stem = f"{language}_{index:03d}_{identity_digest[:20]}"
    if Path(metadata_path).stem != expected_stem or Path(audio_path).stem != expected_stem:
        raise ValueError(f"Audio cache filename does not match its identity: {audio_path}")
    if not Path(audio_path).is_file() or Path(audio_path).stat().st_size <= 0:
        raise ValueError(f"Cached audio is missing or empty: {audio_path}")
    size = Path(audio_path).stat().st_size
    if type(metadata.get("size_bytes")) is not int or metadata["size_bytes"] != size:
        raise ValueError(f"Cached audio size does not match metadata: {audio_path}")
    audio_hash = metadata.get("audio_sha256")
    if not isinstance(audio_hash, str) or not HASH_PATTERN.fullmatch(audio_hash):
        raise ValueError(f"Cached audio hash metadata is malformed: {metadata_path}")
    actual_hash = _sha256(audio_path)
    if actual_hash != audio_hash:
        raise ValueError(f"Cached audio hash does not match metadata: {audio_path}")
    cached_duration = _finite_number(metadata.get("duration_seconds"), "Cached duration")
    if cached_duration <= 0:
        raise ValueError(f"Cached audio duration is not positive: {audio_path}")
    if not has_audio_stream(str(audio_path)):
        raise ValueError(f"Cached file has no valid audio stream: {audio_path}")
    duration = get_media_duration(str(audio_path))
    if not math.isfinite(duration) or duration <= 0 or abs(duration - cached_duration) > 0.03:
        raise ValueError(f"Cached audio duration does not match metadata: {audio_path}")
    return {
        "path": str(Path(audio_path).resolve()),
        "metadata_path": str(Path(metadata_path).resolve()),
        "identity": identity,
        "identity_sha256": identity_digest,
        "audio_sha256": actual_hash,
        "size_bytes": size,
        "duration_seconds": duration,
        "delivery_tag": tag,
    }


def _profile_key(candidate):
    identity = candidate["identity"]
    return _canonical_json({
        "provider": identity["provider"],
        "voice": identity["voice"],
        "model": identity["model"],
        "settings": identity["settings"],
        "schema_version": identity["schema_version"],
    })


def _select_audio(language, index, expected_text, expected_context, cache_dir):
    prefix = f"{language}_{index:03d}_"
    metadata_files = sorted(Path(cache_dir).glob(f"{prefix}*.json"))
    audio_files = sorted(Path(cache_dir).glob(f"{prefix}*.mp3"))
    audio_by_stem = {path.stem: path for path in audio_files}
    if not metadata_files or set(audio_by_stem) != {path.stem for path in metadata_files}:
        raise ValueError(f"{language} event {index} has missing or orphaned audio cache files.")

    candidates = []
    for metadata_path in metadata_files:
        match = IDENTITY_PREFIX_PATTERN.fullmatch(metadata_path.stem)
        if not match or match.group(1) != language or int(match.group(2)) != index:
            raise ValueError(f"Unexpected audio cache filename: {metadata_path}")
        audio_path = audio_by_stem[metadata_path.stem]
        metadata = _read_json(metadata_path, "audio cache metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("identity"), dict):
            raise ValueError(f"Audio metadata is malformed: {metadata_path}")
        identity = metadata["identity"]
        # Validate every variant's identity; only exact script-text matches qualify.
        candidate_text = identity.get("text")
        if not isinstance(candidate_text, str):
            raise ValueError(f"Audio metadata lacks exact text identity: {metadata_path}")
        _validate_identity(identity, candidate_text, language, index)
        digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
        if match.group(3) != digest[:20]:
            raise ValueError(f"Audio cache filename fingerprint is invalid: {metadata_path}")
        context_matches = (
            identity["model"] == "eleven_v3"
            or identity["context"] == expected_context
        )
        if candidate_text == expected_text and context_matches:
            candidates.append(_validate_audio_metadata(
                metadata_path, audio_path, expected_text, language, index,
                expected_context,
            ))
    if not candidates:
        raise ValueError(f"No validated {language} audio variant matches event {index} script text.")
    return candidates


def _match_audio(language, scripts, commentary_events, cache_dir):
    initial_lines = {
        event["event_id"]: event[f"{language}_tts"].strip()
        for event in commentary_events
    }
    final_lines = {
        index: strip_audio_tags(line)
        for index, line in scripts.items()
    }
    expected_contexts = {}
    for index in scripts:
        expected_contexts[index] = {
            "previous_text": final_lines[index - 1] if index > 0 else None,
            "next_text": initial_lines[index + 1] if index + 1 in initial_lines else None,
        }
    by_event = {
        index: _select_audio(language, index, text, expected_contexts[index], cache_dir)
        for index, text in scripts.items()
    }
    common_profiles = None
    for candidates in by_event.values():
        profiles = {_profile_key(candidate) for candidate in candidates}
        common_profiles = profiles if common_profiles is None else common_profiles & profiles
    if len(common_profiles or ()) != 1:
        raise ValueError(f"{language} audio identity profile is missing or ambiguous across events.")
    profile = next(iter(common_profiles))
    selected = {}
    for index, candidates in by_event.items():
        matching = [candidate for candidate in candidates if _profile_key(candidate) == profile]
        unique = {candidate["identity_sha256"]: candidate for candidate in matching}
        if len(unique) != 1:
            raise ValueError(f"{language} event {index} has ambiguous cached audio identities.")
        selected[index] = next(iter(unique.values()))
    return selected


def reconstruct_schedule(root, video_path):
    """Validate local evidence and return selected, integrity-checked segments."""
    root = Path(root).resolve()
    video_path = Path(video_path)
    if not video_path.is_absolute():
        video_path = root / video_path
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise ValueError(f"Source video is missing: {video_path}")

    output_dir = root / "media" / "output"
    gemini_dir = output_dir / ".gemini_cache"
    segment_dir = output_dir / "commentary_segments"
    cache_path, stages = _load_stage_cache(video_path, gemini_dir)
    duration = get_media_duration(str(video_path))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source video duration is invalid.")

    draft_text = _stage_response(stages, "draft")
    verification_text = _stage_response(stages, "verification")
    commentary_text = _stage_response(stages, "commentary")
    draft_events = parse_analysis_response(draft_text, duration_seconds=duration)["events"]
    event_ids = _validate_event_ids(draft_events, "Parsed draft")
    _validate_evidence(stages["evidence"], draft_events)
    _validate_local_signals(stages["local_signals"], event_ids)
    verified_events = parse_verification_response(
        verification_text, draft_events=draft_events, duration_seconds=duration,
    )["events"]
    if _validate_event_ids(verified_events, "Verified stage") != event_ids:
        raise ValueError("Draft and verified event IDs do not match.")

    reconciled = reconcile_possession(verified_events, stages["local_signals"])
    grounded = mark_team_mentions(reconciled, max_mentions=4)
    commentary_ids = [event["event_id"] for event in _strict_commentary_events(commentary_text, event_ids)]
    if commentary_ids != event_ids:
        raise ValueError("Cached commentary order does not match the current event IDs.")
    commentary_events = parse_commentary_response(
        commentary_text, verified_events=grounded, duration_seconds=duration,
    )["events"]
    if _validate_event_ids(commentary_events, "Parsed commentary") != event_ids:
        raise ValueError("Parsed commentary event IDs do not match verified events.")

    scripts = {
        language: _read_script(
            output_dir / f"{language}_tts_script.txt", language, commentary_events,
        )
        for language in ("english", "arabic")
    }
    selected_audio = {
        language: _match_audio(
            language, scripts[language], commentary_events, segment_dir,
        )
        for language in ("english", "arabic")
    }
    verified_by_id = {event["event_id"]: event for event in verified_events}
    starts = calculate_scheduled_starts(verified_events, video_duration=duration)
    schedules = {"english": [], "arabic": []}
    for language in ("english", "arabic"):
        for index, start in enumerate(starts):
            event = verified_by_id[index]
            end = (
                starts[index + 1]
                if index + 1 < len(starts)
                else min(float(event["end_seconds"]), duration)
            )
            audio = selected_audio[language][index]
            available = end - start - VideoMerger.END_MARGIN_SECONDS
            if available <= 0 or audio["duration_seconds"] > available * VideoMerger.MAX_TEMPO:
                raise ValueError(
                    f"{language} event {index} cached audio does not fit the verified window: "
                    f"cue={start:.3f}s, end={end:.3f}s, duration="
                    f"{audio['duration_seconds']:.3f}s."
                )
            schedules[language].append({
                "event_id": index,
                "path": audio["path"],
                "start_seconds": start,
                "end_seconds": end,
                "duration": audio["duration_seconds"],
                "text": scripts[language][index],
                "audio_sha256": audio["audio_sha256"],
                "metadata_path": audio["metadata_path"],
                "identity_sha256": audio["identity_sha256"],
                "size_bytes": audio["size_bytes"],
                "delivery_tag": audio["delivery_tag"],
                "matching_evidence": (
                    "verified cue + next verified cue/final verified end; script text, "
                    "delivery tags, provider/voice/model/settings/context, file hash, "
                    "size and probed duration matched"
                ),
            })
    return {
        "source_video": str(video_path),
        "source_video_sha256": _sha256(video_path),
        "source_duration_seconds": duration,
        "gemini_cache": str(cache_path.resolve()),
        "audio_cache_directory": str(segment_dir.resolve()),
        "schedule_basis": "verification event cue_seconds; end is next verified cue, or final verified end clamped to video duration",
        "languages": schedules,
    }


def _write_json_temp(directory, name, value):
    fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        return Path(temp_name)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _human_report(manifest):
    lines = [
        "Offline continuity re-merge validation",
        f"Source: {manifest['source_video']}",
        f"Source SHA-256: {manifest['source_video_sha256']}",
        f"Video duration: {manifest['source_duration_seconds']:.3f}s",
        f"Gemini cache: {manifest['gemini_cache']}",
        "Schedule: " + manifest["schedule_basis"],
    ]
    for language, segments in manifest["languages"].items():
        lines.extend(("", language.title() + " segments:"))
        for segment in segments:
            lines.append(
                f"  event {segment['event_id']}: cue={segment['start_seconds']:.3f}s, "
                f"end={segment['end_seconds']:.3f}s, "
                f"duration={segment['duration']:.3f}s, "
                f"tag={segment['delivery_tag']}, audio={segment['path']}"
            )
            lines.append(
                f"    identity_sha256={segment['identity_sha256']}; "
                f"audio_sha256={segment['audio_sha256']}; "
                f"size={segment['size_bytes']} bytes"
            )
    return "\n".join(lines) + "\n"


def run_offline_remerge(root=".", video_path="media/input/video.mp4"):
    root = Path(root).resolve()
    output_dir = root / "media" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "english": "english_continuity_comparison.mp4",
        "arabic": "arabic_continuity_comparison.mp4",
    }
    manifest_name = "continuity_remerge_manifest.json"
    report_name = "continuity_remerge_report.txt"
    final_paths = [output_dir / value for value in names.values()]
    final_paths += [output_dir / manifest_name, output_dir / report_name]
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing comparison artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    manifest = reconstruct_schedule(root, video_path)
    temp_paths = []
    try:
        temp_video_paths = {}
        for language, filename in names.items():
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".mp4", dir=output_dir,
            )
            os.close(fd)
            os.unlink(temp_name)
            temp_path = Path(temp_name)
            temp_paths.append(temp_path)
            temp_video_paths[language] = temp_path
            VideoMerger().merge(
                manifest["source_video"], manifest["languages"][language], str(temp_path),
            )

        manifest["outputs"] = {language: filename for language, filename in names.items()}
        temp_manifest = _write_json_temp(output_dir, manifest_name, manifest)
        temp_paths.append(temp_manifest)
        fd, report_temp_name = tempfile.mkstemp(
            prefix=f".{report_name}.", suffix=".tmp", dir=output_dir,
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as report:
            report.write(_human_report(manifest))
            report.flush()
            os.fsync(report.fileno())
        temp_report = Path(report_temp_name)
        temp_paths.append(temp_report)

        for language, filename in names.items():
            os.replace(temp_video_paths[language], output_dir / filename)
        os.replace(temp_manifest, output_dir / manifest_name)
        os.replace(temp_report, output_dir / report_name)
        return [str(path) for path in final_paths]
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Re-merge only validated local Gemini/ElevenLabs caches; no provider calls are made."
    )
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    parser.add_argument("--video", default="media/input/video.mp4", help="Source video path, relative to --root unless absolute.")
    args = parser.parse_args(argv)
    try:
        outputs = run_offline_remerge(args.root, args.video)
    except Exception as exc:
        print(f"Offline re-merge refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(f"Created: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
