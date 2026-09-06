import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm.response_parser import (
    parse_analysis_response,
    parse_commentary_response,
    parse_verification_response,
)
from llm.video_prompt import VideoCommentaryPrompt
from services.summary_builder import SummaryBuilder
from tts.service import (
    TTSService,
    _diagnostic,
    calculate_scheduled_starts,
)


def analysis_event(**overrides):
    event = {
        "event_id": 7,
        "start_seconds": 2.0,
        "end_seconds": 5.0,
        "cue_seconds": 3.0,
        "action_type": "carry",
        "description": "A player carries the ball.",
        "ball_state": "controlled",
        "action_confidence": "high",
    }
    event.update(overrides)
    return event


def verified_event(**overrides):
    event = analysis_event(
        event_id=0,
        possession_team="unclear",
        possession_confidence="low",
        action_peak_seconds=3.0,
        commentary_start_seconds=3.0,
        mode="live",
    )
    event.update(overrides)
    return event


def writer_event(**overrides):
    event = {
        "event_id": 0,
        "english_tts": "The ball moves forward.",
        "arabic_tts": "الكرة تتقدم إلى الأمام.",
        "english_neutral_tts": "The ball moves forward.",
        "arabic_neutral_tts": "الكرة تتقدم إلى الأمام.",
        "intensity": "low",
        "mode": "live",
        "delivery": "focused",
    }
    event.update(overrides)
    return event


class ParserTimingAndModeTests(unittest.TestCase):
    def test_legacy_payload_defaults_and_peak_fields_are_clamped(self):
        result = parse_analysis_response(json.dumps({
            "events": [analysis_event(
                start_seconds=2,
                end_seconds=5,
                cue_seconds=99,
                action_peak_seconds=-4,
                commentary_start_seconds=99,
            )]
        }))
        event = result["events"][0]
        self.assertEqual(event["cue_seconds"], 5.0)
        self.assertEqual(event["action_peak_seconds"], 2.0)
        self.assertEqual(event["commentary_start_seconds"], 5.0)
        self.assertEqual(event["mode"], "live")

    def test_all_supported_modes_normalize_and_unknown_mode_falls_back(self):
        for mode in ("live", "reaction", "build_up", "result"):
            result = parse_analysis_response(json.dumps({
                "events": [analysis_event(mode=mode)]
            }))
            self.assertEqual(result["events"][0]["mode"], mode)

        result = parse_analysis_response(json.dumps({
            "events": [analysis_event(mode="unexpected")]
        }))
        self.assertEqual(result["events"][0]["mode"], "live")

    def test_verification_clamps_new_timing_fields_and_preserves_legacy_draft(self):
        draft = [analysis_event()]
        result = parse_verification_response(json.dumps({
            "events": [{
                "event_id": 7,
                "start_seconds": 2,
                "end_seconds": 5,
                "cue_seconds": 3,
                "action_peak_seconds": 99,
                "commentary_start_seconds": -10,
                "mode": "result",
                "action_type": "carry",
                "description": "A player carries the ball.",
                "ball_state": "controlled",
                "action_confidence": "high",
                "possession_team": "green",
                "possession_confidence": "high",
                "evidence_notes": "Controller and ball remain visible.",
            }]
        }), draft)
        event = result["events"][0]
        self.assertEqual(event["action_peak_seconds"], 5.0)
        self.assertEqual(event["commentary_start_seconds"], 2.0)
        self.assertEqual(event["mode"], "result")

        omitted = parse_verification_response(
            json.dumps({"events": []}),
            [analysis_event()],
        )["events"][0]
        self.assertEqual(omitted["possession_team"], "unclear")
        self.assertEqual(omitted["possession_confidence"], "low")

    def test_commentary_mode_is_normalized_with_verified_event_fallback(self):
        verified = [verified_event(mode="reaction")]
        result = parse_commentary_response(
            json.dumps({"events": [writer_event(mode="not-a-mode")]}),
            verified,
        )
        self.assertEqual(result["events"][0]["mode"], "live")

        result = parse_commentary_response(
            json.dumps({"events": [writer_event(mode="build_up")]}),
            [verified_event(mode="reaction")],
        )
        self.assertEqual(result["events"][0]["mode"], "build_up")


class SchedulingAndDiagnosticsTests(unittest.TestCase):
    def test_scheduled_starts_use_peak_and_mode_without_negative_times(self):
        events = [
            verified_event(action_peak_seconds=0.02, mode="build_up"),
            verified_event(event_id=1, action_peak_seconds=1.0, mode="reaction"),
            verified_event(event_id=2, action_peak_seconds=1.1, mode="result"),
        ]
        starts = calculate_scheduled_starts(events, provider="edge")
        self.assertEqual(len(starts), 3)
        self.assertGreaterEqual(min(starts), 0.0)
        self.assertEqual(starts[0], 0.0)
        self.assertGreaterEqual(starts[1], starts[0])
        self.assertGreaterEqual(starts[2], starts[1])

    def test_tts_windows_stay_valid_for_close_events_offline(self):
        class FakeGenerator:
            provider = "edge"

            def generate(self, *args, **kwargs):
                return 0.4

        events = [
            verified_event(
                start_seconds=0.0, end_seconds=0.35,
                action_peak_seconds=0.02, cue_seconds=0.02,
                english_tts="One.", arabic_tts="واحد.",
            ),
            verified_event(
                event_id=1, start_seconds=0.1, end_seconds=0.3,
                action_peak_seconds=0.1, cue_seconds=0.1,
                english_tts="Two.", arabic_tts="اثنان.",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"TTS_PROVIDER": "edge"}):
                with patch("tts.service.TTSGenerator", FakeGenerator):
                    with patch("tts.service.Path", Path):
                        old = Path.cwd()
                        os.chdir(directory)
                        try:
                            generated = TTSService().generate(events)
                        finally:
                            os.chdir(old)

        for language_segments in generated.values():
            for segment in language_segments:
                self.assertGreaterEqual(segment["start_seconds"], 0.0)
                self.assertGreaterEqual(
                    segment["end_seconds"],
                    segment["start_seconds"] + 0.25,
                )

    def test_diagnostics_and_summary_expose_grounding_and_drift(self):
        event = verified_event(action_peak_seconds=4.25)
        diagnostics = _diagnostic(
            event, start=3.9, duration=1.2,
            window_end=5.1, provider="edge",
        )
        self.assertEqual(diagnostics["action_peak_seconds"], 4.25)
        self.assertEqual(diagnostics["scheduled_start_seconds"], 3.9)
        self.assertEqual(diagnostics["audio_duration_seconds"], 1.2)
        self.assertEqual(diagnostics["window_end_seconds"], 5.1)
        self.assertEqual(diagnostics["semantic_drift_seconds"], -0.35)
        self.assertIn("provider", diagnostics)
        summary = SummaryBuilder().build_events([dict(
            event,
            scheduled_start_seconds=3.9,
        )])
        self.assertIn("action peak 004.2s", summary)
        self.assertIn("scheduled start 003.9s", summary)


class PromptContractTests(unittest.TestCase):
    def test_prompt_represents_provider_aware_word_limits_and_timing_schema(self):
        prompt = VideoCommentaryPrompt().build_commentary([verified_event()])
        self.assertIn("Edge <= 10 English / <= 9 Arabic", prompt)
        self.assertIn("ElevenLabs <= 14 English / <= 12 Arabic", prompt)
        self.assertIn("Never exceed the event window", prompt)
        self.assertIn('"action_peak_seconds"', VideoCommentaryPrompt().build_analysis(10))
        self.assertIn('"commentary_start_seconds"', prompt)


if __name__ == "__main__":
    unittest.main()
