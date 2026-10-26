import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from video.merger import VideoMerger
from utils.process import run_bounded


class BoundedSubprocessTests(unittest.TestCase):

    def test_hanging_subprocess_fails_within_configured_timeout(self):
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                timeout=0.25,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        self.assertLess(time.monotonic() - started, 4.0)

    def test_video_merger_ffmpeg_timeout_is_reported_and_bounded(self):
        class FakeAudio:
            def filter(self, *_args, **_kwargs):
                return self

        class FakeOutput:
            def overwrite_output(self):
                return self

            def get_args(self):
                return ["-c", "import time; time.sleep(30)"]

        merger = VideoMerger()
        merger.FFMPEG_TIMEOUT_SECONDS = 0.25
        started = time.monotonic()
        with (
            patch("video.merger.get_video_duration", return_value=30.0),
            patch("video.merger.has_audio_stream", return_value=False),
            patch("video.merger.imageio_ffmpeg.get_ffmpeg_exe", return_value=sys.executable),
            patch("video.merger.ffmpeg.input", return_value=SimpleNamespace(video=object())),
            patch.object(merger, "_mix_segments", return_value=FakeAudio()),
            patch("video.merger.ffmpeg.output", return_value=FakeOutput()),
        ):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg timed out") as raised:
                merger.merge("source.mp4", [], "out.mp4")
        self.assertLess(time.monotonic() - started, 4.0)
        self.assertIn("time.sleep(30)", str(raised.exception))
        self.assertIn("Diagnostics:", str(raised.exception))


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "real FFmpeg/ffprobe binaries are required",
)
class FFmpegTimingIntegrationTests(unittest.TestCase):

    DURATION = 30.0
    FPS = 25
    CUES = {
        "english": (2.0, 2.6, 15.0, 28.0),
        "arabic": (3.0, 16.0, 29.0),
    }

    def _run(self, command):
        try:
            result = run_bounded(
                command,
                timeout=120,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                f"External process timed out after 120s: {command!r}; "
                f"stdout={self._bounded(exc.stdout)!r}; "
                f"stderr={self._bounded(exc.stderr)!r}"
            )
        if result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )

    def _probe(self, path, selector=None):
        command = ["ffprobe", "-v", "error"]
        if selector:
            command += ["-select_streams", selector]
        command += ["-show_streams", "-show_format", "-show_frames", "-of", "json", str(path)]
        try:
            result = run_bounded(
                command, timeout=60, check=True,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(
                f"ffprobe timed out after 60s: {command!r}; "
                f"stderr={self._bounded(exc.stderr)!r}"
            )
        return json.loads(result.stdout)

    @staticmethod
    def _bounded(value, limit=2000):
        if value is None:
            return "<none>"
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return value[:limit] + ("… [truncated]" if len(value) > limit else "")

    def _make_source(self, directory):
        source = directory / "timestamp-markers.mp4"
        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x204020:s=640x360:r=25:d=30",
            "-vf", (
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                "text='SOURCE %{pts\\:hms}':fontsize=36:fontcolor=white:"
                "x=20:y=20:box=1:boxcolor=black@0.7,"
                "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                "text='KNOWN EVENT':fontsize=42:fontcolor=yellow:x=20:y=100:"
                "enable='between(t,2,2.2)+between(t,2.6,2.8)+"
                "between(t,15,15.2)+between(t,28,28.2)'"
            ),
            "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", str(source),
        ])
        return source

    def _make_marker_audio(self, directory):
        marker = directory / "short-tone.mp3"
        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            "sine=frequency=1000:sample_rate=48000:duration=0.5",
            "-c:a", "libmp3lame", "-q:a", "4", str(marker),
        ])
        return marker

    def _render(self, source, marker, output, cues):
        segments = []
        for index, cue in enumerate(cues):
            audio_path = marker.with_name(f"{output.stem}_{index}.mp3")
            shutil.copyfile(marker, audio_path)
            segments.append({
                "path": str(audio_path),
                "start_seconds": cue,
                "end_seconds": cue + 1.0,
                "duration": 0.5,
            })
        VideoMerger().merge(str(source), segments, str(output))

    def _decoded_audio(self, path, directory):
        pcm = directory / f"{Path(path).stem}.f32"
        self._run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "48000",
            "-f", "f32le", str(pcm),
        ])
        return np.fromfile(pcm, dtype="<f4")

    def _onset(self, samples, expected_seconds):
        expected = round(expected_seconds * 48000)
        lower = max(0, expected - 2400)
        upper = min(len(samples), expected + 2400)
        for offset in range(lower, upper, 24):
            if np.max(np.abs(samples[offset:offset + 24])) > 0.03:
                return offset / 48000
        self.fail(f"No audio marker near {expected_seconds:.3f}s")

    def test_pts_cues_and_audio_transitions_are_stable_in_real_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = self._make_source(directory)
            marker = self._make_marker_audio(directory)
            source_probe = self._probe(source, "v:0")
            source_video = source_probe["streams"][0]
            self.assertEqual(source_video["avg_frame_rate"], "25/1")
            self.assertEqual(float(source_probe["format"]["duration"]), 30.0)
            self.assertEqual(float(source_video["start_time"]), 0.0)
            self.assertEqual(len(source_probe["frames"]), 750)

            measured = {}
            for language, cues in self.CUES.items():
                output = directory / f"{language}.mp4"
                self._render(source, marker, output, cues)
                rendered_probe = self._probe(output, "v:0")
                rendered_video = next(
                    stream for stream in rendered_probe["streams"]
                    if stream["codec_type"] == "video"
                )
                self.assertEqual(rendered_video["avg_frame_rate"], "25/1")
                self.assertEqual(rendered_video["time_base"], source_video["time_base"])
                self.assertEqual(float(rendered_video["start_time"]), 0.0)
                self.assertEqual(len(rendered_probe["frames"]), 750)
                self.assertEqual(
                    [rendered_probe["frames"][i]["best_effort_timestamp_time"]
                     for i in (0, 375, 749)],
                    [source_probe["frames"][i]["best_effort_timestamp_time"]
                     for i in (0, 375, 749)],
                )
                self.assertEqual(float(rendered_probe["format"]["duration"]), 30.0)

                decoded = self._decoded_audio(output, directory)
                onsets = [self._onset(decoded, cue) for cue in cues]
                errors = [onset - cue for onset, cue in zip(onsets, cues)]
                measured[language] = onsets
                self.assertTrue(all(abs(error) <= 0.02 for error in errors), errors)
                self.assertLessEqual(abs(errors[-1] - errors[0]), 0.02, errors)

                if language == "english":
                    first_start = round(cues[0] * 48000)
                    next_start = round(cues[1] * 48000)
                    active = np.flatnonzero(
                        np.abs(decoded[first_start:next_start]) > 0.03
                    )
                    self.assertGreater(len(active), 0)
                    first_end = cues[0] + active[-1] / 48000
                    self.assertLessEqual(cues[1] - first_end, 0.14)

                # There is no per-line fade: a phrase marker keeps its native
                # level through the final 80 ms instead of tapering abruptly.
                last_cue = cues[-1]
                center = round((last_cue + 0.25) * 48000)
                tail = round((last_cue + 0.45) * 48000)
                mid_rms = np.sqrt(np.mean(decoded[center:center + 1200] ** 2))
                tail_rms = np.sqrt(np.mean(decoded[tail:tail + 1200] ** 2))
                self.assertGreater(tail_rms, mid_rms * 0.65)

            self.assertAlmostEqual(
                measured["arabic"][0] - measured["english"][0],
                1.0,
                delta=0.02,
            )
