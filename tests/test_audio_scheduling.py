import base64
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from services.pipeline_service import PipelineService
from tts.generator import TTSGenerator
from tts.service import TTSService, calculate_scheduled_starts
from video.merger import VideoMerger


def event(cue=1.0, end=4.0):
    return {
        "cue_seconds": cue,
        "end_seconds": end,
        "english_tts": "Back it goes.",
        "arabic_tts": "الكرة إلى الخلف.",
    }


class FakeTTS:
    durations = {"english": 1.0, "arabic": 1.0}

    def generate(self, text, path, language, **kwargs):
        return self.durations[language]


class AudioScheduleTests(unittest.TestCase):

    def test_rejects_invalid_negative_and_nonfinite_cues(self):
        for cue in ("bad", -0.1, float("nan"), float("inf"), float("-inf")):
            with self.subTest(cue=cue), self.assertRaises(ValueError):
                calculate_scheduled_starts([event(cue=cue)])

    def test_rejects_nonincreasing_cues(self):
        with self.assertRaisesRegex(ValueError, "not after"):
            calculate_scheduled_starts([event(1.0), event(1.0)])
        with self.assertRaisesRegex(ValueError, "not after"):
            calculate_scheduled_starts([event(2.0), event(1.0)])

    def test_rejects_cue_beyond_video_duration(self):
        with self.assertRaisesRegex(ValueError, "exceeds video duration"):
            calculate_scheduled_starts([event(cue=4.01)], video_duration=4.0)

    def test_accepts_cue_at_video_duration_but_no_window_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    with self.assertRaisesRegex(ValueError, "no valid narration window|event end"):
                        TTSService().generate([event(cue=4.0, end=5.0)], video_duration=4.0)
            finally:
                os.chdir(old)

    def test_rejects_nonfinite_event_end_and_video_duration(self):
        with self.assertRaisesRegex(ValueError, "end_seconds"):
            calculate_scheduled_starts([event(end=float("nan"))])
        for duration in (float("nan"), float("inf"), -1):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as directory:
                old = os.getcwd()
                os.chdir(directory)
                try:
                    with patch("tts.service.TTSGenerator", FakeTTS):
                        with self.assertRaisesRegex(ValueError, "video_duration"):
                            TTSService().generate([event()], video_duration=duration)
                finally:
                    os.chdir(old)

    def test_accepts_tempo_adjustment_up_to_merger_limit_for_both_languages(self):
        # Available media time is 4.0 - 1.0 - end margin; duration fits at 1.30x.
        FakeTTS.durations = {"english": 3.796, "arabic": 2.5}
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    result = TTSService().generate([event()], video_duration=4.0)
                self.assertEqual(result["english"][0]["start_seconds"], 1.0)
                self.assertEqual(result["arabic"][0]["start_seconds"], 1.0)
                self.assertEqual(result["english"][0]["duration"], 3.796)
            finally:
                os.chdir(old)

    def test_rejects_audio_exceeding_max_tempo_with_language_diagnostic(self):
        FakeTTS.durations = {"english": 4.0, "arabic": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    with self.assertRaisesRegex(ValueError, "english event 0.*maximum_tempo"):
                        TTSService().generate([event()], video_duration=4.0)
            finally:
                os.chdir(old)

    def test_english_and_arabic_durations_are_checked_independently(self):
        FakeTTS.durations = {"english": 1.0, "arabic": 4.0}
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    with self.assertRaisesRegex(ValueError, "arabic event 0"):
                        TTSService().generate([event()], video_duration=4.0)
            finally:
                os.chdir(old)


    def test_window_is_clamped_to_video_duration(self):
        FakeTTS.durations = {"english": 1.0, "arabic": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    result = TTSService().generate(
                        [event(end=8.0)], video_duration=4.0
                    )
                self.assertEqual(result["english"][0]["end_seconds"], 4.0)
                self.assertEqual(result["arabic"][0]["end_seconds"], 4.0)
            finally:
                os.chdir(old)

    def test_rejects_cues_too_close_to_form_nonoverlapping_windows(self):
        FakeTTS.durations = {"english": 0.01, "arabic": 0.01}
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", FakeTTS):
                    with self.assertRaisesRegex(ValueError, "no valid narration window"):
                        TTSService().generate(
                            [event(1.0, 2.0), event(1.05, 2.0)],
                            video_duration=3.0,
                        )
            finally:
                os.chdir(old)

    def test_saved_elevenlabs_media_duration_is_authoritative(self):
        generator = TTSGenerator.__new__(TTSGenerator)
        generator.provider = "elevenlabs"
        generator.model_id = "test-model"
        generator.default_voice_id = "voice"
        generator.client = SimpleNamespace(
            text_to_speech=SimpleNamespace(
                convert_with_timestamps=lambda **kwargs: SimpleNamespace(
                    audio_base_64=base64.b64encode(b"synthetic-audio").decode(),
                    alignment=SimpleNamespace(character_end_times_seconds=[9.0]),
                )
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "speech.mp3")
            with patch.object(generator, "_voice_id", return_value="voice"), patch(
                "tts.generator.get_media_duration", return_value=1.25
            ) as probe:
                duration = generator.generate("Hello", path, "english")
            self.assertEqual(duration, 1.25)
            probe.assert_called_once_with(path)

    def test_merger_rejects_negative_start_instead_of_clamping(self):
        merger = VideoMerger()
        with self.assertRaisesRegex(ValueError, "Invalid commentary window"):
            merger._prepare_segment({
                "path": "synthetic.mp3",
                "start_seconds": -0.1,
                "end_seconds": 2.0,
                "duration": 0.5,
            })

    def test_merger_explicitly_fails_when_tempo_exceeds_limit(self):
        merger = VideoMerger()

        class FakeAudio:
            def filter(self, *args, **kwargs):
                return self

        with patch("video.merger.ffmpeg.input", return_value=SimpleNamespace(audio=FakeAudio())):
            with self.assertRaisesRegex(ValueError, "required_tempo=.*maximum_tempo"):
                merger._prepare_segment({
                    "path": "synthetic.mp3",
                    "start_seconds": 1.0,
                    "end_seconds": 1.2,
                    "duration": 1.0,
                })


class PipelineScheduleIntegrationTests(unittest.TestCase):

    def test_pipeline_forwards_actual_video_duration_to_tts(self):
        service = PipelineService.__new__(PipelineService)
        service.summary_builder = SimpleNamespace(build_events=lambda events: "summary")
        service.generator = SimpleNamespace(
            generate_from_video=lambda *args, **kwargs: {"events": [{
                "cue_seconds": 1.0,
                "end_seconds": 3.0,
                "english_tts": "Back it goes.",
                "arabic_tts": "الكرة إلى الخلف.",
            }]}
        )
        captured = {}
        service.tts = SimpleNamespace(
            generate=lambda events, **kwargs: captured.update(kwargs) or {
                "english": [], "arabic": []
            }
        )
        service.merger = SimpleNamespace(merge=lambda *args: None)
        with patch("services.pipeline_service.get_video_duration", return_value=5.0), patch(
            "services.pipeline_service.save_text"
        ), patch("services.pipeline_service.ENABLE_TTS", True), patch(
            "services.pipeline_service.ENABLE_VIDEO_MERGE", False
        ):
            service.run("clip.mp4")
        self.assertEqual(captured["video_duration"], 5.0)


if __name__ == "__main__":
    unittest.main()
