import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.offline_remerge import reconstruct_schedule, run_offline_remerge


CUES = (2.0, 4.5, 7.5, 10.0, 12.8, 15.2, 17.5, 20.5, 24.5)
JPEG_B64 = base64.b64encode(b"\xff\xd8mock-jpeg\xff\xd9").decode("ascii")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class OfflineRemergeTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "media" / "output"
        self.cache_dir = self.output / "commentary_segments"
        self.gemini_dir = self.output / ".gemini_cache"
        self.cache_dir.mkdir(parents=True)
        self.gemini_dir.mkdir(parents=True)
        self.video = self.root / "media" / "input" / "video.mp4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"local source video marker")
        (self.output / "english_commentary.mp4").write_bytes(b"preserve english original")
        (self.output / "arabic_commentary.mp4").write_bytes(b"preserve arabic original")
        self._write_stage_cache()
        self.audio_paths = {"english": [], "arabic": []}
        self.metadata_paths = {"english": [], "arabic": []}
        for language in ("english", "arabic"):
            for index in range(len(CUES)):
                self._write_audio_variant(language, index, selected=True)
                # Alternative text is present for each event and must not be
                # selected merely because it sorts first in the cache folder.
                self._write_audio_variant(language, index, selected=False)
        self._write_scripts()

    def _write_stage_cache(self):
        draft = []
        verified = []
        commentary = []
        evidence = []
        local_signals = []
        for index, cue in enumerate(CUES):
            start = max(0.0, cue - 0.5)
            end = cue + 0.7 if index + 1 < len(CUES) else 26.0
            draft.append({
                "start_seconds": start,
                "end_seconds": end,
                "cue_seconds": cue,
                "action_type": "back_pass",
                "description": "A player plays a backward pass.",
                "ball_state": "controlled",
                "action_confidence": "high",
            })
            verified.append({
                "event_id": index,
                "start_seconds": start,
                "end_seconds": end,
                "cue_seconds": cue,
                "action_type": "back_pass",
                "description": "A player plays a backward pass.",
                "ball_state": "controlled",
                "action_confidence": "high",
                "possession_team": "unclear",
                "possession_confidence": "low",
                "evidence_notes": "Ball moves backward from the player.",
            })
            commentary.append({
                "event_id": index,
                "english_tts": f"Back it goes {index}.",
                "arabic_tts": f"الكرة إلى الخلف {index}.",
                "english_neutral_tts": f"Back it goes {index}.",
                "arabic_neutral_tts": f"الكرة إلى الخلف {index}.",
                "intensity": "low",
                "delivery": "focused",
            })
            evidence.append({
                "event_id": index,
                "cue_seconds": cue,
                "timestamps": [cue - 0.2, cue, cue + 0.2, cue + 0.4],
                "jpeg_base64": JPEG_B64,
            })
            local_signals.append({
                "event_id": index,
                "possession_team": "unclear",
                "confidence": "low",
                "evidence": "No confident local vote.",
            })
        video_hash = hashlib.sha256(self.video.read_bytes()).hexdigest()
        cache = {
            "schema_version": 1,
            "video_fingerprint": video_hash,
            "identity_digest": "a" * 64,
            "upload": None,
            "stages": {
                "draft": self._stage_value({"events": draft}),
                "evidence": self._stage_value(evidence),
                "local_signals": self._stage_value(local_signals),
                "verification": self._stage_value({"events": verified}),
                "commentary": self._stage_value({"events": commentary}),
            },
        }
        (self.gemini_dir / f"{video_hash}.json").write_text(
            json.dumps(cache), encoding="utf-8",
        )

    @staticmethod
    def _stage_value(value):
        response = isinstance(value, dict) and "events" in value
        return {
            "context_digest": hashlib.sha256(b"ctx").hexdigest(),
            "value": (
                {"model": "gemini-3.5-flash", "response_text": json.dumps(value, ensure_ascii=False)}
                if response else value
            ),
        }

    def _identity(self, language, index, selected, voice=None):
        spoken = (
            f"Back it goes {index}." if selected else f"The backward pass continues {index}."
        ) if language == "english" else (
            f"الكرة إلى الخلف {index}." if selected else f"تمريرة خلفية مستمرة {index}."
        )
        return {
            "context": {"previous_text": None, "next_text": None},
            "delivery_tags": ["[focused]"],
            "model": "eleven_v3",
            "provider": "elevenlabs",
            "schema_version": 1,
            "settings": {
                "output_format": "mp3_44100_128",
                "voice_settings": {"stability": 0.3},
            },
            "synthesis_text": f"[focused] {spoken}",
            "text": f"[focused] {spoken}",
            "voice": voice or f"{language}-voice",
        }

    def _write_audio_variant(self, language, index, selected, voice=None):
        identity = self._identity(language, index, selected, voice)
        identity_digest = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
        stem = f"{language}_{index:03d}_{identity_digest[:20]}"
        audio = self.cache_dir / f"{stem}.mp3"
        metadata_path = self.cache_dir / f"{stem}.json"
        audio.write_bytes(f"mock-mp3-{language}-{index}-{selected}-{voice}".encode())
        metadata = {
            "cache_version": 1,
            "identity": identity,
            "duration_seconds": 0.5,
            "size_bytes": audio.stat().st_size,
            "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        if selected:
            self.audio_paths[language].append(audio)
            self.metadata_paths[language].append(metadata_path)

    def _write_scripts(self):
        for language in ("english", "arabic"):
            lines = []
            for index in range(len(CUES)):
                text = (
                    f"Back it goes {index}." if language == "english"
                    else f"الكرة إلى الخلف {index}."
                )
                lines.append(f"[focused] {text}")
            (self.output / f"{language}_tts_script.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8",
            )

    def _mock_probes(self):
        return (
            patch("tools.offline_remerge.get_media_duration", side_effect=lambda path: 30.0 if str(path).endswith("video.mp4") else 0.5),
            patch("tools.offline_remerge.has_audio_stream", return_value=True),
        )

    def test_valid_cache_selects_exact_script_variant_and_remerges_offline(self):
        duration_patch, stream_patch = self._mock_probes()
        with duration_patch, stream_patch:
            manifest = reconstruct_schedule(self.root, self.video)
            self.assertEqual(
                [segment["start_seconds"] for segment in manifest["languages"]["english"]],
                list(CUES),
            )
            self.assertEqual(
                Path(manifest["languages"]["english"][0]["path"]),
                self.audio_paths["english"][0].resolve(),
            )

            def merge(_source, _segments, output_path):
                Path(output_path).write_bytes(b"comparison render")

            with patch("tools.offline_remerge.VideoMerger") as merger:
                merger.MAX_TEMPO = 1.35
                merger.END_MARGIN_SECONDS = 0.08
                merger.return_value.merge.side_effect = merge
                created = run_offline_remerge(self.root, self.video)
            self.assertEqual(merger.return_value.merge.call_count, 2)
        self.assertTrue((self.output / "english_continuity_comparison.mp4").is_file())
        self.assertTrue((self.output / "arabic_continuity_comparison.mp4").is_file())
        self.assertTrue((self.output / "continuity_remerge_manifest.json").is_file())
        self.assertTrue((self.output / "continuity_remerge_report.txt").is_file())
        self.assertEqual((self.output / "english_commentary.mp4").read_bytes(), b"preserve english original")
        self.assertEqual((self.output / "arabic_commentary.mp4").read_bytes(), b"preserve arabic original")
        self.assertEqual(len(created), 4)

    def test_schedule_uses_verified_cue_when_it_differs_from_draft(self):
        cache_path = next(self.gemini_dir.glob("*.json"))
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        record = cache["stages"]["verification"]["value"]
        payload = json.loads(record["response_text"])
        payload["events"][0]["cue_seconds"] = 2.25
        record["response_text"] = json.dumps(payload)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        duration_patch, stream_patch = self._mock_probes()
        with duration_patch, stream_patch:
            manifest = reconstruct_schedule(self.root, self.video)
        self.assertEqual(manifest["languages"]["english"][0]["start_seconds"], 2.25)

    def test_multiple_matching_profiles_are_rejected_as_ambiguous(self):
        for index in range(len(CUES)):
            self._write_audio_variant("english", index, selected=True, voice="alternate-en-voice")
        duration_patch, stream_patch = self._mock_probes()
        with duration_patch, stream_patch, self.assertRaisesRegex(ValueError, "ambiguous"):
            reconstruct_schedule(self.root, self.video)

    def test_invalid_audio_hash_metadata_is_rejected(self):
        metadata_path = self.metadata_paths["english"][0]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["audio_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        duration_patch, stream_patch = self._mock_probes()
        with duration_patch, stream_patch, self.assertRaisesRegex(ValueError, "hash"):
            reconstruct_schedule(self.root, self.video)

    def test_missing_gemini_evidence_stage_fails_closed(self):
        cache_path = next(self.gemini_dir.glob("*.json"))
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        del cache["stages"]["evidence"]
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        duration_patch, stream_patch = self._mock_probes()
        with duration_patch, stream_patch, self.assertRaisesRegex(ValueError, "missing"):
            reconstruct_schedule(self.root, self.video)


if __name__ == "__main__":
    unittest.main()
