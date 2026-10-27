"""Offline acoustic-boundary diagnostics for continuity re-merge outputs.

This report measures waveform activity only. It cannot determine whether a
spoken word or sentence is semantically complete; that needs alignment, ASR, or
human listening.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

import imageio_ffmpeg

from utils.process import run_bounded
from video.merger import VideoMerger


SAMPLE_RATE = 16000
FRAME_MILLISECONDS = 10
ACTIVITY_FLOOR_DBFS = -55.0
RELATIVE_ACTIVITY_DB = 35.0
ANALYSIS_TIMEOUT_SECONDS = 120


def _dbfs(value):
    if value <= 0:
        return None
    return 20.0 * math.log10(value)


def _activity_interval(samples, sample_rate=SAMPLE_RATE):
    """Return one thresholded activity interval; no speech semantics implied."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return None
    frame_size = max(1, round(sample_rate * FRAME_MILLISECONDS / 1000))
    frame_count = math.ceil(samples.size / frame_size)
    padded = np.pad(samples, (0, frame_count * frame_size - samples.size))
    frames = padded.reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak_rms = float(np.max(rms))
    threshold = max(
        10.0 ** (ACTIVITY_FLOOR_DBFS / 20.0),
        peak_rms * 10.0 ** (-RELATIVE_ACTIVITY_DB / 20.0),
    )
    active = rms >= threshold
    if not np.any(active):
        return None

    # Bridge gaps no longer than 30 ms to avoid splitting a phrase on brief
    # consonant closures; retain longer pauses as separate inactivity.
    max_gap_frames = 3
    indices = np.flatnonzero(active)
    for left, right in zip(indices, indices[1:]):
        if right - left - 1 <= max_gap_frames:
            active[left:right + 1] = True
    active_indices = np.flatnonzero(active)
    start_sample = int(active_indices[0] * frame_size)
    end_sample = min(samples.size, int((active_indices[-1] + 1) * frame_size))
    return {
        "start_seconds": start_sample / sample_rate,
        "end_seconds": end_sample / sample_rate,
        "peak_rms": peak_rms,
        "peak_dbfs": _dbfs(peak_rms),
        "activity_threshold_dbfs": _dbfs(threshold),
    }


def _decode_audio(
    path, executable, *, filter_graph=None,
    start_seconds=None, duration_seconds=None,
):
    command = [executable, "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if start_seconds is not None:
        command += ["-ss", f"{float(start_seconds):.6f}"]
    if duration_seconds is not None:
        command += ["-t", f"{float(duration_seconds):.6f}"]
    if filter_graph:
        command += ["-af", filter_graph]
    command += [
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1",
    ]
    result = run_bounded(
        command,
        timeout=ANALYSIS_TIMEOUT_SECONDS,
        capture_output=True,
    )
    if result.returncode:
        detail = (result.stderr or b"").decode("utf-8", errors="replace")[:3000]
        raise RuntimeError(f"FFmpeg decode failed for {path}: {detail}")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def _measure_peak_dbfs(path, executable):
    result = run_bounded(
        [
            executable, "-hide_banner", "-i", str(path), "-vn",
            "-af", "volumedetect", "-f", "null", "-",
        ],
        timeout=ANALYSIS_TIMEOUT_SECONDS,
        capture_output=True,
    )
    diagnostics = (result.stderr or b"").decode("utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"FFmpeg peak measurement failed for {path}: {diagnostics[:3000]}")
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", diagnostics)
    if not match:
        raise RuntimeError(f"FFmpeg did not report a decoded AAC peak for {path}.")
    return float(match.group(1))


def _segment_filters(segment):
    start = float(segment["start_seconds"])
    end = float(segment["end_seconds"])
    duration = float(segment["duration"])
    available = end - start - VideoMerger.END_MARGIN_SECONDS
    if not all(math.isfinite(value) for value in (start, end, duration, available)):
        raise ValueError("Manifest contains non-finite audio timing values.")
    if start < 0 or end <= start or duration <= 0 or available <= 0:
        raise ValueError("Manifest contains an invalid audio playback window.")
    tempo = max(1.0, duration / available)
    if tempo > VideoMerger.MAX_TEMPO:
        raise ValueError("Manifest segment exceeds the approved tempo limit.")
    chain = [
        "silenceremove=start_periods=1:start_duration=0.02:"
        "start_threshold=-50dB:start_silence=0.02:timestamp=write",
    ]
    if abs(tempo - 1.0) > 0.001:
        chain.append(f"atempo={round(tempo, 6):.6f}")
    chain.extend((
        "volume=1.08",
        f"adelay=delays={round(start * 1000)}:all=1",
        "asetpts=N/SR/TB",
        "aresample=16000",
        "aformat=sample_fmts=flt:channel_layouts=mono",
    ))
    return ",".join(chain), tempo


def _window_stats(samples, start, end):
    left = max(0, round(start * SAMPLE_RATE))
    right = min(samples.size, round(end * SAMPLE_RATE))
    window = samples[left:right]
    if window.size == 0:
        return {"rms": 0.0, "rms_dbfs": None, "peak": 0.0}
    rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
    peak = float(np.max(np.abs(window)))
    return {"rms": rms, "rms_dbfs": _dbfs(rms), "peak": peak}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_manifest(root, manifest_path):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("languages"), dict):
        raise ValueError("Offline re-merge manifest has an unsupported structure.")

    output_names = manifest.get("outputs")
    if not isinstance(output_names, dict):
        raise ValueError("Manifest is missing rendered comparison output names.")
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    report = {
        "manifest": str(manifest_path.resolve()),
        "activity_method": (
            f"{FRAME_MILLISECONDS}ms RMS frames; floor {ACTIVITY_FLOOR_DBFS} dBFS; "
            f"{RELATIVE_ACTIVITY_DB} dB below per-file peak"
        ),
        "overlap_method": (
            "adjacent interval comparison of each MP3 after the merger's per-segment filters; "
            "final mixed AAC cannot attribute activity to individual speakers"
        ),
        "semantic_sentence_completeness": (
            "NOT ASSESSED: waveform activity cannot prove words or sentence meaning; "
            "use provider alignment, ASR, or human listening"
        ),
        "languages": {},
    }

    for language, segments in manifest["languages"].items():
        output_name = output_names.get(language)
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            raise ValueError(f"Manifest has an unsafe {language} output filename.")
        output_path = root / "media" / "output" / output_name
        if not output_path.is_file():
            raise FileNotFoundError(f"Rendered {language} comparison video is missing: {output_path}")
        event_reports = []
        for index, segment in enumerate(segments):
            audio_path = Path(segment["path"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"Cached audio segment is missing: {audio_path}")
            if (
                type(segment.get("size_bytes")) is not int
                or audio_path.stat().st_size != segment["size_bytes"]
                or segment.get("audio_sha256") != _file_sha256(audio_path)
            ):
                raise ValueError(f"Cached audio no longer matches manifest integrity data: {audio_path}")
            native = _decode_audio(audio_path, executable)
            native_activity = _activity_interval(native)
            filter_graph, tempo = _segment_filters(segment)
            processed = _decode_audio(audio_path, executable, filter_graph=filter_graph)
            processed_activity = _activity_interval(processed)
            start = float(segment["start_seconds"])
            end = float(segment["end_seconds"])
            native_duration = native.size / SAMPLE_RATE
            processed_end = (
                processed_activity["end_seconds"] if processed_activity else None
            )
            next_audible_start = None
            if index + 1 < len(segments):
                next_graph, _ = _segment_filters(segments[index + 1])
                next_audio = _decode_audio(
                    segments[index + 1]["path"], executable, filter_graph=next_graph,
                )
                next_activity = _activity_interval(next_audio)
                next_audible_start = (
                    next_activity["start_seconds"] if next_activity else None
                )
            overlap = (
                processed_end is not None
                and next_audible_start is not None
                and processed_end > next_audible_start + 0.01
            )
            event_reports.append({
                "event_id": segment.get("event_id"),
                "cue_seconds": start,
                "window_end_seconds": end,
                "cached_mp3": str(audio_path.resolve()),
                "manifest_duration_seconds": float(segment["duration"]),
                "decoded_mp3_duration_seconds": native_duration,
                "decoded_duration_delta_seconds": native_duration - float(segment["duration"]),
                "native_mp3_activity": native_activity,
                "tempo_applied": tempo,
                "processed_activity_absolute": processed_activity,
                "next_segment_activity_start_seconds": next_audible_start,
                "adjacent_activity_overlap": overlap,
                "window_overrun": processed_end is not None and processed_end > end + 0.01,
                "first_last_phoneme_check": (
                    "NOT DETERMINABLE FROM AMPLITUDE; compare alignment or listen to source and output"
                ),
            })

        gain_windows = []
        for index, segment in enumerate(segments):
            activity = event_reports[index]["processed_activity_absolute"]
            active_start = (
                activity["start_seconds"]
                if activity else float(segment["start_seconds"])
            )
            active_end = (
                activity["end_seconds"]
                if activity else active_start + 0.01
            )
            window_audio = _decode_audio(
                output_path,
                executable,
                start_seconds=active_start,
                duration_seconds=max(0.01, active_end - active_start),
            )
            stats = _window_stats(
                window_audio, 0.0, max(0.01, active_end - active_start),
            )
            cue = float(segment["start_seconds"])
            window_end = (
                float(segments[index + 1]["start_seconds"])
                if index + 1 < len(segments)
                else float(segment["end_seconds"])
            )
            rendered_window = _decode_audio(
                output_path,
                executable,
                start_seconds=cue,
                duration_seconds=window_end - cue,
            )
            rendered_activity = _activity_interval(rendered_window)
            if rendered_activity is not None:
                rendered_activity = {
                    "start_seconds": cue + rendered_activity["start_seconds"],
                    "end_seconds": cue + rendered_activity["end_seconds"],
                    "peak_dbfs": rendered_activity["peak_dbfs"],
                }
            gain_windows.append({
                "event_id": segment.get("event_id"),
                "active_start_seconds": active_start,
                "active_end_seconds": active_end,
                "rendered_aac_window_activity": rendered_activity,
                "activity_touches_window_end": (
                    rendered_activity is not None
                    and rendered_activity["end_seconds"] >= window_end - 0.02
                ),
                **stats,
            })

        peak_dbfs = _measure_peak_dbfs(output_path, executable)
        report["languages"][language] = {
            "rendered_aac_video": str(output_path.resolve()),
            "decoded_aac_peak_dbfs": peak_dbfs,
            "gain_by_event_window": gain_windows,
            "segments": event_reports,
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure local cached-audio boundaries and rendered AAC levels without provider calls."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--manifest", default="media/output/continuity_remerge_manifest.json",
        help="Offline re-merge manifest path, relative to --root unless absolute.",
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(
            analyze_manifest(args.root, args.manifest),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ))
    except Exception as exc:
        print(f"Audio integrity analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
