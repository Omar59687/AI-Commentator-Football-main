import base64
import asyncio
import os
import re
import threading
from pathlib import Path

from tts.text import strip_audio_tags
from utils.video_metadata import get_media_duration


DELIVERY_TAG_PATTERN = re.compile(r"\[[^\[\]\r\n]+\]")


class TTSGenerator:

    def __init__(self):

        self.provider = os.getenv(
            "TTS_PROVIDER",
            "elevenlabs",
        ).strip().lower()

        if self.provider not in {
            "elevenlabs",
            "edge",
        }:
            raise ValueError(
                "TTS_PROVIDER must be 'elevenlabs' or 'edge'."
            )

        self.client = None
        self.default_voice_id = None
        self.model_id = None

        if self.provider == "edge":
            return

        api_key = os.getenv("ELEVENLABS_API_KEY")

        if not api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY is missing from the .env file."
            )

        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "elevenlabs is not installed. Run: "
                "pip install -r requirements.txt"
            ) from exc

        self.client = ElevenLabs(api_key=api_key)

        self.default_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.model_id = os.getenv(
            "ELEVENLABS_MODEL_ID",
            "eleven_v3"
        )

    def _edge_voice(self, language: str) -> str:

        defaults = {
            "english": "en-GB-RyanNeural",
            "arabic": "ar-SA-HamedNeural",
        }

        return os.getenv(
            f"EDGE_TTS_{language.upper()}_VOICE",
            defaults[language],
        )

    def _edge_option(
        self,
        language: str,
        option: str,
        default: str,
    ) -> str:

        return os.getenv(
            f"EDGE_TTS_{language.upper()}_{option}",
            os.getenv(f"EDGE_TTS_{option}", default),
        )

    async def _save_edge_audio(
        self,
        text: str,
        output_path: str,
        language: str,
    ):

        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run: "
                "pip install -r requirements.txt"
            ) from exc

        communication = edge_tts.Communicate(
            text=text,
            voice=self._edge_voice(language),
            rate=self._edge_option(
                language,
                "RATE",
                "+8%",
            ),
            volume=self._edge_option(
                language,
                "VOLUME",
                "+0%",
            ),
            pitch=self._edge_option(
                language,
                "PITCH",
                "+0Hz",
            ),
        )

        await communication.save(output_path)

    def _run_async(self, coroutine):

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result = {}
        failure = {}

        def run_in_thread():
            try:
                result["value"] = asyncio.run(coroutine)
            except BaseException as exc:
                failure["error"] = exc

        worker = threading.Thread(
            target=run_in_thread,
            daemon=True,
        )
        worker.start()
        worker.join()

        if "error" in failure:
            raise failure["error"]

        return result.get("value")

    def _generate_edge(
        self,
        text: str,
        output_path: str,
        language: str,
    ) -> float:

        speech_text = strip_audio_tags(text)

        if not speech_text:
            raise ValueError(
                "Cannot generate speech from empty text."
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._run_async(
                self._save_edge_audio(
                    speech_text,
                    str(output),
                    language,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"Edge TTS failed for {language}: {exc}"
            ) from exc

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                "Edge TTS returned no audio data."
            )

        duration = get_media_duration(
            str(output)
        )

        print(
            f"Generated {language} narration with Edge TTS: "
            f"{duration:.2f} seconds"
        )

        return duration

    def _voice_id(self, language: str) -> str:

        language_voice = os.getenv(
            f"ELEVENLABS_{language.upper()}_VOICE_ID"
        )

        voice_id = language_voice or self.default_voice_id

        if not voice_id:
            raise ValueError(
                f"No ElevenLabs voice configured for {language}. "
                f"Set ELEVENLABS_{language.upper()}_VOICE_ID "
                f"or ELEVENLABS_VOICE_ID."
            )

        return voice_id

    def _speech_text(self, text: str) -> str:
        if self.provider == "edge":
            return strip_audio_tags(text)
        return text if self.model_id == "eleven_v3" else strip_audio_tags(text)

    def _elevenlabs_voice_settings(self):
        if self.model_id == "eleven_v3":
            return {
                "stability": float(
                    os.getenv("ELEVENLABS_STABILITY", "0.30")
                ),
            }
        return {
            "stability": float(
                os.getenv("ELEVENLABS_STABILITY", "0.32")
            ),
            "similarity_boost": float(
                os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.78")
            ),
            "style": float(os.getenv("ELEVENLABS_STYLE", "0.45")),
            "use_speaker_boost": True,
            "speed": float(os.getenv("ELEVENLABS_SPEED", "1.05")),
        }

    def cache_identity(
        self,
        text: str,
        language: str,
        previous_text: str = None,
        next_text: str = None,
    ):
        """Return all inputs that can change synthesized audio."""
        if self.provider == "edge":
            voice = self._edge_voice(language)
            model = "edge-tts"
            settings = {
                "rate": self._edge_option(language, "RATE", "+8%"),
                "volume": self._edge_option(language, "VOLUME", "+0%"),
                "pitch": self._edge_option(language, "PITCH", "+0Hz"),
            }
            context = {"previous_text": None, "next_text": None}
        else:
            voice = self._voice_id(language)
            model = self.model_id
            settings = {
                "output_format": "mp3_44100_128",
                "voice_settings": self._elevenlabs_voice_settings(),
            }
            context = {
                "previous_text": previous_text if self.model_id != "eleven_v3" else None,
                "next_text": next_text if self.model_id != "eleven_v3" else None,
            }

        return {
            "schema_version": 1,
            "provider": self.provider,
            "voice": voice,
            "model": model,
            "settings": settings,
            "text": text,
            "delivery_tags": DELIVERY_TAG_PATTERN.findall(text),
            "synthesis_text": self._speech_text(text),
            "context": context,
        }

    def generate(
        self,
        text: str,
        output_path: str,
        language: str,
        previous_text: str = None,
        next_text: str = None,
    ) -> float:

        if self.provider == "edge":
            return self._generate_edge(
                text,
                output_path,
                language,
            )

        speech_text = self._speech_text(text)

        options = {
            "voice_id": self._voice_id(language),
            "text": speech_text,
            "model_id": self.model_id,
            "output_format": "mp3_44100_128",
        }

        if self.model_id != "eleven_v3":
            if previous_text:
                options["previous_text"] = previous_text

            if next_text:
                options["next_text"] = next_text

        options["voice_settings"] = self._elevenlabs_voice_settings()

        response = (
            self.client
            .text_to_speech
            .convert_with_timestamps(**options)
        )

        encoded_audio = getattr(
            response,
            "audio_base_64",
            None
        )

        if not encoded_audio:
            encoded_audio = getattr(
                response,
                "audio_base64",
                None
            )

        if not encoded_audio:
            raise RuntimeError(
                "ElevenLabs returned no audio data."
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(base64.b64decode(encoded_audio))

        alignment = (
            getattr(response, "normalized_alignment", None)
            or getattr(response, "alignment", None)
        )
        alignment_end = None
        if alignment is not None:
            end_times = getattr(
                alignment,
                "character_end_times_seconds",
                None
            )
            if end_times:
                alignment_end = float(end_times[-1])

        # Character alignment is useful for text timing, but the saved media
        # duration is the authoritative duration used by the scheduler.
        duration = get_media_duration(str(output))

        print(
            f"Generated {language} narration: "
            f"{duration:.2f} seconds"
            + (
                f" (alignment ends at {alignment_end:.2f}s)"
                if alignment_end is not None
                else ""
            )
        )

        return duration
