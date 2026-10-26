from pathlib import Path

import ffmpeg
import imageio_ffmpeg

from utils.video_metadata import (
    get_video_duration,
    has_audio_stream,
)


class VideoMerger:

    END_MARGIN_SECONDS = 0.08
    FADE_SECONDS = 0.08
    MAX_TEMPO = 1.35

    def _change_tempo(
        self,
        audio,
        tempo: float,
    ):

        if tempo <= 0:
            raise ValueError(
                "Audio tempo must be greater than zero."
            )

        while tempo > 2.0:
            audio = audio.filter(
                "atempo",
                2.0,
            )
            tempo /= 2.0

        while tempo < 0.5:
            audio = audio.filter(
                "atempo",
                0.5,
            )
            tempo /= 0.5

        if abs(tempo - 1.0) > 0.001:
            audio = audio.filter(
                "atempo",
                round(tempo, 6),
            )

        return audio

    def _prepare_segment(self, segment):

        start = max(
            0.0,
            float(segment["start_seconds"]),
        )

        end = max(
            start + 0.25,
            float(segment["end_seconds"]),
        )

        duration = float(segment["duration"])

        available_time = max(
            0.25,
            end - start - self.END_MARGIN_SECONDS,
        )

        audio = ffmpeg.input(
            segment["path"]
        ).audio

        audio = audio.filter(
            "silenceremove",
            start_periods=1,
            start_duration=0.02,
            start_threshold="-50dB",
            start_silence=0.02,
            timestamp="write",
        )

        # Only speed up segments that are too long.
        # Short segments remain natural instead of being stretched.
        tempo = max(
            1.0,
            duration / available_time,
        )

        if tempo > self.MAX_TEMPO:
            raise ValueError(
                "Commentary segment cannot fit within its scheduled "
                "window without exceeding the maximum tempo: "
                f"path={segment['path']}, "
                f"window={start:.2f}s-{end:.2f}s "
                f"(available={available_time:.2f}s), "
                f"duration={duration:.2f}s, "
                f"required_tempo={tempo:.2f}, "
                f"maximum_tempo={self.MAX_TEMPO:.2f}."
            )

        audio = self._change_tempo(
            audio,
            tempo,
        )

        audio = audio.filter(
            "volume",
            1.08,
        )

        fade_start = max(
            0.0,
            min(duration / tempo, available_time)
            - self.FADE_SECONDS,
        )

        audio = audio.filter(
            "afade",
            t="out",
            st=fade_start,
            d=self.FADE_SECONDS,
        )

        delay_ms = round(start * 1000)

        audio = audio.filter(
            "adelay",
            **{
                "delays": str(delay_ms),
                "all": 1,
            },
        )

        audio = audio.filter(
            "asetpts",
            "N/SR/TB",
        )

        return audio

    def _mix_segments(self, segments):

        prepared = []

        for segment in segments:

            audio = self._prepare_segment(
                segment
            )

            if audio is None:
                raise RuntimeError(
                    "Commentary segment preparation returned no audio for "
                    f"{segment.get('path', '<unknown path>')}."
                )
            prepared.append(audio)

        if not prepared:
            raise ValueError(
                "No commentary segments were provided."
            )

        if len(prepared) == 1:
            return prepared[0]

        return ffmpeg.filter(
            prepared,
            "amix",
            inputs=len(prepared),
            duration="longest",
            dropout_transition=0,
        )

    def merge(
        self,
        video_path: str,
        segments,
        output_path: str,
    ):

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        video_duration = get_video_duration(
            video_path
        )

        ffmpeg_executable = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        source_has_audio = has_audio_stream(
            video_path
        )

        video = ffmpeg.input(video_path)

        commentary = self._mix_segments(
            segments
        )

        if source_has_audio:

            background = video.audio.filter(
                "volume",
                0.20,
            )

            mixed_audio = ffmpeg.filter(
                [background, commentary],
                "amix",
                inputs=2,
                duration="longest",
                dropout_transition=0,
            ).filter("apad")

        else:

            mixed_audio = commentary.filter(
                "apad"
            )

        output = ffmpeg.output(
            video.video,
            mixed_audio,
            output_path,
            vcodec="copy",
            acodec="aac",
            t=video_duration,
            movflags="+faststart",
        )

        try:

            output.overwrite_output().run(
                cmd=ffmpeg_executable,
                capture_stdout=True,
                capture_stderr=True,
            )

        except ffmpeg.Error as exc:

            details = (
                exc.stderr.decode(
                    "utf-8",
                    errors="replace",
                )
                if exc.stderr
                else str(exc)
            )

            raise RuntimeError(
                "FFmpeg failed to create the "
                f"commentary video:\n{details}"
            ) from exc
