import json
import unittest

from llm.response_parser import (
    parse_analysis_response,
    parse_commentary_response,
    parse_verification_response,
)
from llm.video_prompt import VideoCommentaryPrompt
from services.summary_builder import SummaryBuilder
from tts.text import strip_audio_tags
from vision.possession_verifier import (
    mark_team_mentions,
    reconcile_possession,
)


def make_analysis_event(**overrides):

    event = {
        "event_id": 0,
        "start_seconds": 1.0,
        "end_seconds": 3.5,
        "cue_seconds": 1.5,
        "action_type": "back_pass",
        "description": "A player releases a backward pass.",
        "ball_state": "controlled",
        "action_confidence": "high",
    }

    event.update(overrides)
    return event


def make_verified_event(**overrides):

    event = make_analysis_event()

    event.update({
        "possession_team": "green",
        "possession_confidence": "high",
        "evidence_notes": (
            "The green controller and ball remain visible."
        ),
        "safe_to_name_team": True,
        "mention_team": True,
    })

    event.update(overrides)
    return event


def make_writer_event(**overrides):

    event = {
        "event_id": 0,
        "english_tts": "Green play it back.",
        "arabic_tts": "الأخضر يعيد الكرة للخلف.",
        "english_neutral_tts": "Back it goes.",
        "arabic_neutral_tts": "الكرة إلى الخلف.",
        "intensity": "low",
        "delivery": "focused",
    }

    event.update(overrides)
    return event


class AnalysisParserTests(unittest.TestCase):

    def test_accepts_json_code_fence(self):

        payload = {
            "events": [make_analysis_event()]
        }

        result = parse_analysis_response(
            "```json\n"
            + json.dumps(payload, ensure_ascii=False)
            + "\n```"
        )

        self.assertEqual(
            result["events"][0]["action_type"],
            "back_pass",
        )

    def test_clamps_analysis_to_video_duration(self):

        payload = {
            "events": [
                make_analysis_event(
                    start_seconds=8.0,
                    end_seconds=12.0,
                    cue_seconds=9.0,
                ),
                make_analysis_event(
                    start_seconds=11.0,
                    end_seconds=13.0,
                    cue_seconds=11.5,
                ),
            ]
        }

        result = parse_analysis_response(
            json.dumps(payload),
            duration_seconds=10.0,
        )

        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(
            result["events"][0]["end_seconds"],
            10.0,
        )


class VerificationParserTests(unittest.TestCase):


    def test_rejects_missing_verification_event(self):
        draft = [make_analysis_event(event_id=0), make_analysis_event(event_id=1)]
        with self.assertRaisesRegex(ValueError, "omitted event IDs: 1"):
            parse_verification_response(json.dumps({"events": [make_analysis_event(event_id=0)]}), draft)

    def test_rejects_duplicate_verification_event_id(self):
        draft = [make_analysis_event(event_id=0)]
        with self.assertRaisesRegex(ValueError, "duplicate event ID 0"):
            parse_verification_response(json.dumps({"events": [make_analysis_event(event_id=0), make_analysis_event(event_id=0)]}), draft)

    def test_rejects_malformed_verification_event_id(self):
        draft = [make_analysis_event(event_id=0)]
        malformed = make_analysis_event(event_id="0")
        with self.assertRaisesRegex(ValueError, "integer event ID"):
            parse_verification_response(json.dumps({"events": [malformed]}), draft)

    def test_rejects_non_object_verification_event(self):
        draft = [make_analysis_event(event_id=0)]
        with self.assertRaisesRegex(ValueError, "must be an object"):
            parse_verification_response(json.dumps({"events": [None]}), draft)

    def test_rejects_unexpected_verification_event_id(self):
        draft = [make_analysis_event(event_id=0)]
        with self.assertRaisesRegex(ValueError, "unexpected event ID 2"):
            parse_verification_response(json.dumps({"events": [make_analysis_event(event_id=0), make_analysis_event(event_id=2)]}), draft)

    def test_preserves_corrected_verified_event(self):
        draft = [make_analysis_event(event_id=0)]
        corrected = make_analysis_event(event_id=0, action_type="shot", description="A shot is taken.", ball_state="in_flight")
        corrected.update({
            "possession_team": "white",
            "possession_confidence": "high",
            "evidence_notes": "The ball is travelling after the shot.",
        })
        result = parse_verification_response(json.dumps({"events": [corrected]}), draft)["events"][0]
        self.assertEqual(result["action_type"], "shot")
        self.assertEqual(result["description"], "A shot is taken.")
        self.assertEqual(result["possession_team"], "unclear")
        self.assertEqual(result["possession_confidence"], "low")

    def test_removes_possession_from_travelling_ball(self):

        draft = [make_analysis_event()]
        verification = {
            "events": [
                {
                    **make_analysis_event(),
                    "ball_state": "in_flight",
                    "possession_team": "white",
                    "possession_confidence": "high",
                    "evidence_notes": "The ball is travelling.",
                }
            ]
        }

        result = parse_verification_response(
            json.dumps(verification),
            draft_events=draft,
        )

        event = result["events"][0]
        self.assertEqual(
            event["possession_team"],
            "unclear",
        )
        self.assertEqual(
            event["possession_confidence"],
            "low",
        )

    def test_removes_outfield_color_from_keeper_action(self):

        draft = [make_analysis_event()]
        verification = {
            "events": [
                {
                    **make_analysis_event(),
                    "description": (
                        "The goalkeeper controls the ball."
                    ),
                    "possession_team": "white",
                    "possession_confidence": "high",
                    "evidence_notes": "Keeper on the ball.",
                }
            ]
        }

        result = parse_verification_response(
            json.dumps(verification),
            draft_events=draft,
        )

        self.assertEqual(
            result["events"][0]["possession_team"],
            "unclear",
        )


class PossessionReconciliationTests(unittest.TestCase):

    def test_independent_agreement_allows_team_name(self):

        events = [make_verified_event()]
        signals = [{
            "event_id": 0,
            "possession_team": "green",
            "confidence": "high",
            "evidence": "Four green votes.",
        }]

        result = reconcile_possession(events, signals)[0]

        self.assertEqual(result["possession_team"], "green")
        self.assertTrue(result["safe_to_name_team"])
        self.assertEqual(
            result["verification_source"],
            "gemini+local-agreement",
        )

    def test_conflict_suppresses_team_name(self):

        events = [make_verified_event(
            possession_team="white"
        )]
        signals = [{
            "event_id": 0,
            "possession_team": "green",
            "confidence": "high",
            "evidence": "Four green votes.",
        }]

        result = reconcile_possession(events, signals)[0]

        self.assertEqual(result["possession_team"], "unclear")
        self.assertFalse(result["safe_to_name_team"])
        self.assertEqual(
            result["verification_source"],
            "conflict",
        )

    def test_gemini_alone_cannot_authorize_spoken_color(self):

        events = [make_verified_event(
            possession_team="white"
        )]

        result = reconcile_possession(events, [])[0]

        self.assertEqual(result["possession_team"], "white")
        self.assertFalse(result["safe_to_name_team"])
        self.assertEqual(
            result["verification_source"],
            "gemini-only",
        )

    def test_names_team_once_per_verified_phase(self):

        events = [
            make_verified_event(event_id=0),
            make_verified_event(event_id=1),
            make_verified_event(
                event_id=2,
                possession_team="white",
            ),
        ]

        result = mark_team_mentions(events)

        self.assertEqual(
            [event["mention_team"] for event in result],
            [True, False, True],
        )


class CommentaryParserTests(unittest.TestCase):

    def test_keeps_matching_verified_team(self):

        writer = {"events": [make_writer_event()]}

        result = parse_commentary_response(
            json.dumps(writer, ensure_ascii=False),
            verified_events=[make_verified_event()],
        )

        self.assertEqual(
            result["events"][0]["english_tts"],
            "Green play it back.",
        )

    def test_uses_neutral_line_when_team_not_approved(self):

        writer = {"events": [make_writer_event()]}

        result = parse_commentary_response(
            json.dumps(writer, ensure_ascii=False),
            verified_events=[
                make_verified_event(
                    mention_team=False,
                    safe_to_name_team=False,
                )
            ],
        )

        self.assertEqual(
            result["events"][0]["english_tts"],
            "Back it goes.",
        )
        self.assertEqual(
            result["events"][0]["arabic_tts"],
            "الكرة إلى الخلف.",
        )

    def test_wrong_color_is_replaced(self):

        writer = {
            "events": [
                make_writer_event(
                    english_tts="White play it back.",
                    arabic_tts="الأبيض يعيد الكرة للخلف.",
                )
            ]
        }

        result = parse_commentary_response(
            json.dumps(writer, ensure_ascii=False),
            verified_events=[make_verified_event()],
        )

        self.assertEqual(
            result["events"][0]["english_tts"],
            "Back it goes.",
        )


class PromptTests(unittest.TestCase):

    def test_prompts_separate_observation_and_writing(self):

        builder = VideoCommentaryPrompt()
        analysis = builder.build(30)
        verification = builder.build_verification([
            make_analysis_event()
        ])
        writing = builder.build_commentary([
            make_verified_event()
        ])

        self.assertIn("30-second", analysis)
        self.assertIn("Do not write commentary yet", analysis)
        self.assertIn("draft below is untrusted", verification)
        self.assertIn("four panels", verification)
        self.assertIn("writing-only pass", writing)
        self.assertIn("mention_team", writing)


class TextTests(unittest.TestCase):

    def test_strips_performance_tags_for_display(self):

        text = strip_audio_tags(
            "[focused] Patient build-up. "
            "[excited] Now forward!"
        )

        self.assertEqual(
            text,
            "Patient build-up. Now forward!",
        )


class SummaryTests(unittest.TestCase):

    def test_builds_auditable_event_timeline(self):

        event = make_verified_event(
            start_seconds=2.0,
            end_seconds=7.5,
            cue_seconds=3.0,
            intensity="medium",
            verification_source="gemini+local-agreement",
            gemini_possession_team="green",
            local_possession_team="green",
            local_evidence="Three green votes.",
        )

        summary = SummaryBuilder().build_events([event])

        self.assertIn("002.0s-007.5s", summary)
        self.assertIn("commentary cue 003.0s", summary)
        self.assertIn("possession=green", summary)
        self.assertIn("source=gemini+local-agreement", summary)
        self.assertIn("Gemini=green", summary)
        self.assertIn("local=green", summary)


if __name__ == "__main__":
    unittest.main()
