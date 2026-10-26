import json
import os
import stat
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm.generator import CommentaryGenerator, MAX_MODEL_ATTEMPTS
from llm.stage_cache import StageCache
from llm.video_prompt import VideoCommentaryPrompt


class FakeAPIError(Exception):

    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def make_generator(side_effects, models=None):

    generator = CommentaryGenerator.__new__(CommentaryGenerator)
    generator.model = "primary"
    generator.fallback_models = models or []
    generator.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=side_effects))
    )
    return generator


class GeminiRetryTests(unittest.TestCase):

    def test_sdk_retry_budget_is_one_attempt(self):

        with patch("llm.generator.genai.Client") as client, patch(
            "llm.generator.EventEvidenceBuilder"
        ), patch("llm.generator.LocalPossessionVerifier"):
            CommentaryGenerator(api_key="test-key")
        options = client.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)

    def test_503_recovers_with_bounded_exponential_retry(self):

        response = SimpleNamespace(text='{"events": []}')
        generator = make_generator([FakeAPIError(503), response])

        with patch("llm.generator.errors.APIError", FakeAPIError), patch(
            "llm.generator.time.sleep"
        ) as sleep, patch("llm.generator.random.uniform", return_value=0.0):
            result = generator._generate_json_with_fallback(
                "prompt",
                "test stage",
                temperature=0.0,
            )

        self.assertEqual(result, ('{"events": []}', "primary"))
        self.assertEqual(
            generator.client.models.generate_content.call_count,
            2,
        )
        sleep.assert_called_once_with(0.5)

    def test_permanent_authentication_error_is_not_retried(self):

        generator = make_generator([FakeAPIError(401)])

        with patch("llm.generator.errors.APIError", FakeAPIError), patch(
            "llm.generator.time.sleep"
        ) as sleep:
            with self.assertRaises(FakeAPIError):
                generator._generate_json_with_fallback(
                    "prompt",
                    "test stage",
                    temperature=0.0,
                )

        self.assertEqual(
            generator.client.models.generate_content.call_count,
            1,
        )
        sleep.assert_not_called()

    def test_retries_are_bounded_across_all_models(self):

        attempts = MAX_MODEL_ATTEMPTS * 2
        generator = make_generator(
            [FakeAPIError(503) for _ in range(attempts)],
            models=["fallback"],
        )

        with patch("llm.generator.errors.APIError", FakeAPIError), patch(
            "llm.generator.time.sleep"
        ) as sleep, patch("llm.generator.random.uniform", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "bounded retries"):
                generator._generate_json_with_fallback(
                    "prompt",
                    "test stage",
                    temperature=0.0,
                )

        self.assertEqual(
            generator.client.models.generate_content.call_count,
            attempts,
        )
        self.assertEqual(sleep.call_count, attempts - 2)


class StageCacheTests(unittest.TestCase):

    def test_reuses_matching_stage_and_rejects_prompt_or_video_changes(self):

        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, "clip.mp4")
            cache_dir = os.path.join(directory, "cache")
            with open(video, "wb") as target:
                target.write(b"video-one")

            cache = StageCache(video, {"model": "primary"}, cache_dir)
            cache.set_stage(
                "draft",
                {"prompt": "v1"},
                {"model": "primary", "response_text": "draft"},
            )

            same = StageCache(video, {"model": "primary"}, cache_dir)
            self.assertIsNotNone(same.get_stage("draft", {"prompt": "v1"}))
            self.assertIsNone(same.get_stage("draft", {"prompt": "v2"}))

            changed_identity = StageCache(video, {"model": "other"}, cache_dir)
            self.assertIsNone(
                changed_identity.get_stage("draft", {"prompt": "v1"})
            )

            with open(video, "wb") as target:
                target.write(b"video-two")
            changed_video = StageCache(video, {"model": "primary"}, cache_dir)
            self.assertIsNone(
                changed_video.get_stage("draft", {"prompt": "v1"})
            )

    def test_cache_file_is_private_and_does_not_store_credentials(self):

        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, "clip.mp4")
            with open(video, "wb") as target:
                target.write(b"video")

            cache = StageCache(
                video,
                {"model": "primary"},
                os.path.join(directory, "cache"),
            )
            cache.set_stage("draft", {"prompt": "p"}, {
                "model": "primary",
                "response_text": '{"events":[]}',
            })

            mode = stat.S_IMODE(os.stat(cache.path).st_mode)
            self.assertEqual(mode & 0o077, 0)
            with open(cache.path, encoding="utf-8") as source:
                stored = source.read()
            self.assertNotIn("test-key", stored)
            self.assertNotIn("api_key", stored)

    def test_prompt_identity_invalidates_stages_but_keeps_same_video_upload(self):

        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, "clip.mp4")
            with open(video, "wb") as target:
                target.write(b"video")

            cache_dir = os.path.join(directory, "cache")
            cache = StageCache(video, {"prompt": "old"}, cache_dir)
            cache.set_stage("draft", {"prompt": "old"}, {"text": "old"})
            cache.set_upload({
                "name": "files/123",
                "uri": "https://files.example/video",
                "mime_type": "video/mp4",
                "saved_at": 1.0,
            })

            changed = StageCache(video, {"prompt": "new"}, cache_dir)
            self.assertIsNone(changed.get_stage("draft", {"prompt": "old"}))
            self.assertEqual(changed.get_upload()["name"], "files/123")


class GeminiSuccessfulContinuationTests(unittest.TestCase):

    def test_retry_recovers_verification_then_cached_result_continues(self):

        analysis = {
            "events": [{
                "event_id": 0,
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "cue_seconds": 1.5,
                "action_type": "back_pass",
                "description": "A player releases a backward pass.",
                "ball_state": "controlled",
                "action_confidence": "high",
            }]
        }
        verification = {
            "events": [{
                **analysis["events"][0],
                "possession_team": "unclear",
                "possession_confidence": "low",
                "evidence_notes": "The shirt is unclear.",
            }]
        }
        commentary = {
            "events": [{
                "event_id": 0,
                "english_tts": "Back it goes.",
                "arabic_tts": "الكرة إلى الخلف.",
                "english_neutral_tts": "Back it goes.",
                "arabic_neutral_tts": "الكرة إلى الخلف.",
                "intensity": "low",
                "delivery": "focused",
            }]
        }
        active_upload = SimpleNamespace(
            name="files/video-1",
            uri="https://files.example/video-1",
            mime_type="video/mp4",
            state=SimpleNamespace(name="ACTIVE"),
        )
        model_call = Mock(side_effect=[
            SimpleNamespace(text=json.dumps(analysis)),
            *[FakeAPIError(503) for _ in range(MAX_MODEL_ATTEMPTS * 2)],
            SimpleNamespace(text=json.dumps(verification)),
            SimpleNamespace(text=json.dumps(commentary)),
        ])
        generator = CommentaryGenerator.__new__(CommentaryGenerator)
        generator.model = "primary"
        generator.fallback_models = ["fallback"]
        generator.client = SimpleNamespace(
            files=SimpleNamespace(
                upload=Mock(return_value=active_upload),
                get=Mock(return_value=active_upload),
                delete=Mock(),
            ),
            models=SimpleNamespace(generate_content=model_call),
        )
        generator.prompt_builder = VideoCommentaryPrompt()
        evidence_build = Mock(return_value=[{
            "event_id": 0,
            "cue_seconds": 1.5,
            "timestamps": [1.0, 1.25, 1.5, 2.0],
            "frames": [],
            "jpeg_bytes": b"jpeg-evidence",
        }])
        local_verify = Mock(return_value=[])
        generator.evidence_builder = SimpleNamespace(
            cell_width=800,
            jpeg_quality=80,
            build=evidence_build,
        )
        generator.local_verifier = SimpleNamespace(
            image_size=1280,
            verify=local_verify,
        )

        with tempfile.TemporaryDirectory() as directory, patch(
            "llm.generator.os.getenv",
            side_effect=lambda key, default=None: (
                directory if key == "GEMINI_CACHE_DIR" else default
            ),
        ), patch("llm.generator.errors.APIError", FakeAPIError), patch(
            "llm.generator.time.sleep"
        ), patch("llm.generator.random.uniform", return_value=0.0):
            video_path = os.path.join(directory, "clip.mp4")
            with open(video_path, "wb") as target:
                target.write(b"synthetic video")

            with self.assertRaisesRegex(RuntimeError, "bounded retries"):
                generator.generate_from_video(
                    video_path,
                    "analysis prompt",
                    duration_seconds=5.0,
                )

            self.assertEqual(model_call.call_count, 1 + MAX_MODEL_ATTEMPTS * 2)
            generator.client.files.delete.assert_not_called()
            self.assertEqual(generator.client.files.upload.call_count, 1)

            result = generator.generate_from_video(
                video_path,
                "analysis prompt",
                duration_seconds=5.0,
            )
            self.assertEqual(
                result["events"][0]["english_tts"],
                "Back it goes.",
            )
            self.assertEqual(
                model_call.call_count,
                1 + MAX_MODEL_ATTEMPTS * 2 + 2,
            )
            generator.client.files.get.assert_called_once_with(
                name="files/video-1"
            )
            generator.client.files.delete.assert_called_once_with(
                name="files/video-1"
            )
            evidence_build.assert_called_once()
            local_verify.assert_called_once()

            second = generator.generate_from_video(
                video_path,
                "analysis prompt",
                duration_seconds=5.0,
            )
            self.assertEqual(second, result)
            self.assertEqual(
                model_call.call_count,
                1 + MAX_MODEL_ATTEMPTS * 2 + 2,
            )
            self.assertEqual(generator.client.files.upload.call_count, 1)


if __name__ == "__main__":
    unittest.main()
