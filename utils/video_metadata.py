import re
import subprocess
from pathlib import Path

import imageio_ffmpeg


DURATION_PATTERN = re.compile(
    r"Duration:\s*"
    r"(?P<hours>\d+):"
    r"(?P<minutes>\d{2}):"
    r"(?P<seconds>\d+(?:\.\d+)?)"
)

AUDIO_STREAM_PATTERN = re.compile(
    r"Stream #\d+:\d+[^\n]*Audio:"
)


def _probe_media(video_path: str) -> str:

    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    ffmpeg_executable = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    result = subprocess.run(
        [
            ffmpeg_executable,
            "-hide_banner",
            "-i",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    return result.stderr


def get_media_duration(media_path: str) -> float:

    probe_output = _probe_media(media_path)

    match = DURATION_PATTERN.search(
        probe_output
    )

    if match is None:
        raise ValueError(
            f"Could not read media duration: {media_path}"
        )

    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))

    duration = (
        hours * 3600
        + minutes * 60
        + seconds
    )

    if duration <= 0:
        raise ValueError(
            f"Invalid video duration: {duration}"
        )

    return duration


def get_video_duration(video_path: str) -> float:

    return get_media_duration(video_path)


def has_audio_stream(video_path: str) -> bool:

    probe_output = _probe_media(video_path)

    return bool(
        AUDIO_STREAM_PATTERN.search(
            probe_output
        )
    )
