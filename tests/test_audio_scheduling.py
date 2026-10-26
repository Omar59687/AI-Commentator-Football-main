import base64
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm.response_parser import grounded_fallback_lines
from services.pipeline_service import PipelineService
from tts.generator import TTSGenerator
from tts.service import TTSService, calculate_scheduled_starts
from video.merger import VideoMerger


def event(cue=1.0, end=4.0):
    return {
        "cue_seconds": cue,
        "end_seconds": end,
        "action_type": "back_pass",
        "description": "A player releases a backward pass.",
        "ball_state": "controlled",
        "action_confidence": "high",
        "english_tts": "Back it goes.",
        "arabic_tts": "الكرة إلى الخلف.",
    }


class FakeTTS:
    durations = {"english": 1.0, "arabic": 1.0}
    calls = []

    def generate(self, text, path, language, **kwargs):
        duration = self.durations[language]
        self.calls.append((text, path, language))
        Path(path).write_text(str(duration), encoding="utf-8")
        return duration


class CacheableFakeTTS(FakeTTS):
    duration_by_text = {}
    reported_duration_by_text = {}

    def cache_identity(self, text, language, previous_text=None, next_text=None):
        return {
            "provider": "mock",
            "voice": language,
            "model": "mock-model",
            "settings": {"rate": "default"},
            "text": text,
            "delivery_tags": [text.split()[0]],
            "context": {
                "previous_text": previous_text,
                "next_text": next_text,
            },
        }

    def generate(self, text, path, language, **kwargs):
        duration = self.duration_by_text.get(text, self.durations[language])
        self.calls.append((text, path, language))
        Path(path).write_text(str(duration), encoding="utf-8")
        return self.reported_duration_by_text.get(text, duration)


class AudioScheduleTests(unittest.TestCase):

    def setUp(self):
        FakeTTS.durations = {"english": 1.0, "arabic": 1.0}
        FakeTTS.calls = []
        CacheableFakeTTS.durations = {"english": 1.0, "arabic": 1.0}
        CacheableFakeTTS.calls = []
        CacheableFakeTTS.duration_by_text = {}
        CacheableFakeTTS.reported_duration_by_text = {}
        self.probe_patch = patch(
            "tts.service.get_media_duration",
            side_effect=lambda path: float(Path(path).read_text(encoding="utf-8")),
        )
        self.probe_patch.start()
        self.addCleanup(self.probe_patch.stop)
        self.audio_stream_patch = patch("tts.service.has_audio_stream", return_value=True)
        self.audio_stream_patch.start()
        self.addCleanup(self.audio_stream_patch.stop)

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
            calculate_scheduled_starts([event(cue=4.01, end=5.0)], video_duration=4.0)

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

    def test_cache_reuses_only_complete_matching_audio(self):
        generator = CacheableFakeTTS()
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", return_value=generator):
                    first = TTSService().generate([event()], video_duration=4.0)
                    TTSService().generate([event()], video_duration=4.0)
                self.assertEqual(len(generator.calls), 2)
                self.assertEqual(first["english"][0]["duration"], 1.0)

                cached_path = Path(first["english"][0]["path"])
                cached_path.write_bytes(b"9.9")
                with patch("tts.service.TTSGenerator", return_value=generator):
                    TTSService().generate([event()], video_duration=4.0)
                self.assertEqual(len(generator.calls), 3)
            finally:
                os.chdir(old)

    def test_fitting_uses_verified_action_line_and_actual_probe_duration(self):
        generator = CacheableFakeTTS()
        source = event(cue=1.0, end=4.0)
        source.update({
            "action_type": "pass_release",
            "description": "A player releases a pass into midfield.",
            "english_tts": "The player releases a pass into midfield.",
            "english_neutral_tts": "Pass into the middle.",
            "arabic_tts": "اللاعب يرسل تمريرة نحو منتصف الملعب.",
        })
        generator.duration_by_text = {
            "[focused] The player releases a pass into midfield.": 4.2,
            "[focused] A pass is released.": 1.25,
            "[focused] اللاعب يرسل تمريرة نحو منتصف الملعب.": 1.5,
            "[focused] تُرسل الكرة بتمريرة.": 1.5,
        }
        generator.reported_duration_by_text = {
            "[focused] The player releases a pass into midfield.": 0.5,
            "[focused] A pass is released.": 0.4,
        }
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", return_value=generator):
                    result = TTSService().generate([source], video_duration=4.0)
                self.assertEqual(source["english_tts"], "A pass is released.")
                self.assertEqual(result["english"][0]["text"], "[focused] A pass is released.")
                self.assertEqual(result["english"][0]["duration"], 1.25)
                self.assertEqual(result["english"][0]["start_seconds"], 1.0)
                self.assertEqual(result["english"][0]["end_seconds"], 4.0)
            finally:
                os.chdir(old)

    def test_bounded_grounded_fallback_fails_without_dropping_event(self):
        generator = CacheableFakeTTS()
        source = event(cue=1.0, end=4.0)
        source.update({
            "action_type": "pass_release",
            "description": "A player releases a pass inside.",
            "english_tts": "The player releases a pass inside.",
            "english_neutral_tts": "A pass inside.",
        })
        generator.duration_by_text = {
            "[focused] The player releases a pass inside.": 5.0,
            "[focused] A pass is released.": 5.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", return_value=generator):
                    with self.assertRaisesRegex(ValueError, "2 bounded phrasing attempts"):
                        TTSService().generate([source], video_duration=4.0)
                self.assertEqual(len(generator.calls), 2)
                self.assertEqual(source["english_tts"], "The player releases a pass inside.")
            finally:
                os.chdir(old)

    def test_event_8_recovers_from_one_candidate_with_verified_short_line_and_cache(self):
        generator = CacheableFakeTTS()
        source = event(cue=24.5, end=26.0)
        source.update({
            "event_id": 8,
            "action_type": "pass_release",
            "description": "A player releases a pass toward midfield.",
            "ball_state": "in_flight",
            "english_tts": "The player releases a pass toward midfield.",
            "english_neutral_tts": "The player releases a pass toward midfield.",
            "arabic_tts": "اللاعب يرسل تمريرة نحو منتصف الملعب.",
            "arabic_neutral_tts": "اللاعب يرسل تمريرة نحو منتصف الملعب.",
        })
        generator.duration_by_text = {
            "[focused] The player releases a pass toward midfield.": 2.04,
            "[focused] A pass is released.": 1.6,
            "[focused] اللاعب يرسل تمريرة نحو منتصف الملعب.": 1.3,
            "[focused] تُرسل الكرة بتمريرة.": 1.3,
        }
        original = dict(source)
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            os.chdir(directory)
            try:
                with patch("tts.service.TTSGenerator", return_value=generator):
                    result = TTSService().generate([source], video_duration=30.0)
                    second_result = TTSService().generate([original], video_duration=30.0)
                self.assertEqual(result["english"][0]["start_seconds"], 24.5)
                self.assertEqual(result["arabic"][0]["start_seconds"], 24.5)
                self.assertEqual(result["english"][0]["end_seconds"], 26.0)
                self.assertEqual(result["english"][0]["duration"], 1.6)
                self.assertEqual(result["arabic"][0]["duration"], 1.3)
                self.assertEqual(result["english"][0]["text"], "[focused] A pass is released.")
                self.assertEqual(source["english_tts"], "A pass is released.")
                self.assertEqual(second_result["english"][0]["path"], result["english"][0]["path"])
                self.assertEqual(len(generator.calls), 3)
            finally:
                os.chdir(old)

    def test_generator_cache_identity_covers_voice_model_settings_text_and_tags(self):
        generator = TTSGenerator.__new__(TTSGenerator)
        generator.provider = "elevenlabs"
        generator.model_id = "eleven_v3"
        generator.default_voice_id = "default-voice"
        with patch.dict(os.environ, {
            "ELEVENLABS_ENGLISH_VOICE_ID": "english-voice",
            "ELEVENLABS_STABILITY": "0.30",
        }):
            identity = generator.cache_identity(
                "[excited] A shot is taken!",
                "english",
            )
            changed_text = generator.cache_identity(
                "[focused] A shot is taken!",
                "english",
            )
        self.assertEqual(identity["provider"], "elevenlabs")
        self.assertEqual(identity["voice"], "english-voice")
        self.assertEqual(identity["model"], "eleven_v3")
        self.assertEqual(identity["delivery_tags"], ["[excited]"])
        self.assertEqual(identity["text"], "[excited] A shot is taken!")
        self.assertNotEqual(identity, changed_text)

        with patch.dict(os.environ, {"ELEVENLABS_STABILITY": "0.55"}):
            changed_settings = generator.cache_identity(
                "[excited] A shot is taken!",
                "english",
            )
        self.assertNotEqual(identity, changed_settings)

    def test_action_fallback_requires_matching_verified_context(self):
        verified = event()
        self.assertEqual(
            grounded_fallback_lines(verified),
            ("A backward pass is played.", "تمريرة إلى الخلف."),
        )

        wrong_description = {**verified, "description": "A player takes a shot."}
        self.assertIsNone(grounded_fallback_lines(wrong_description))
        low_confidence = {**verified, "action_confidence": "low"}
        self.assertIsNone(grounded_fallback_lines(low_confidence))
        hidden_ball = {**verified, "ball_state": "not_visible"}
        self.assertIsNone(grounded_fallback_lines(hidden_ball))


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

    def test_pipeline_writes_the_scripts_selected_for_audio(self):
        service = PipelineService.__new__(PipelineService)
        service.summary_builder = SimpleNamespace(build_events=lambda events: "summary")
        events = [{
            "cue_seconds": 1.0,
            "end_seconds": 3.0,
            "english_tts": "Now, a quick pass.",
            "english_neutral_tts": "A pass.",
            "arabic_tts": "تمريرة سريعة.",
            "arabic_neutral_tts": "تمريرة.",
            "delivery": "focused",
        }]
        service.generator = SimpleNamespace(
            generate_from_video=lambda *args, **kwargs: {"events": events}
        )

        def fit_events(actual_events, **kwargs):
            actual_events[0]["english_tts"] = "A pass."
            actual_events[0]["arabic_tts"] = "تمريرة."
            return {"english": [], "arabic": []}

        service.tts = SimpleNamespace(generate=fit_events)
        service.merger = SimpleNamespace(merge=lambda *args: None)
        saved = {}
        with patch("services.pipeline_service.get_video_duration", return_value=5.0), patch(
            "services.pipeline_service.save_text",
            side_effect=lambda content, path: saved.update({path: content}),
        ), patch("services.pipeline_service.ENABLE_TTS", True), patch(
            "services.pipeline_service.ENABLE_VIDEO_MERGE", False
        ):
            result = service.run("clip.mp4")

        self.assertEqual(result["english"], "A pass.")
        self.assertEqual(result["arabic"], "تمريرة.")
        self.assertEqual(saved["media/output/english_tts_script.txt"], "[focused] A pass.")
        self.assertEqual(saved["media/output/arabic_tts_script.txt"], "[focused] تمريرة.")


if __name__ == "__main__":
    unittest.main()
