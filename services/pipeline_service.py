from services.summary_builder import SummaryBuilder

from llm.generator import CommentaryGenerator
from llm.video_prompt import VideoCommentaryPrompt

from utils.file_utils import save_text
from utils.video_metadata import get_video_duration

from tts.service import TTSService, build_tts_text

from video.merger import VideoMerger

from app.config import (
    ENABLE_TTS,
    ENABLE_VIDEO_MERGE,
)


class PipelineService:

    def __init__(
        self,
        api_key,
        model,
    ):

        self.summary_builder = SummaryBuilder()

        self.generator = CommentaryGenerator(
            api_key=api_key,
            model=model,
        )

        self.tts = TTSService()
        self.merger = VideoMerger()

    def run(
        self,
        video_path,
        max_frames=None,
    ):

        duration = get_video_duration(video_path)

        prompt = VideoCommentaryPrompt().build(
            duration
        )

        commentary = self.generator.generate_from_video(
            video_path,
            prompt,
            duration_seconds=duration,
        )

        events = commentary["events"]

        english = " ".join(
            event["english_tts"].strip()
            for event in events
        )

        arabic = " ".join(
            event["arabic_tts"].strip()
            for event in events
        )

        english_tts = "\n".join(
            build_tts_text(event, "english")
            for event in events
        )

        arabic_tts = "\n".join(
            build_tts_text(event, "arabic")
            for event in events
        )

        save_text(
            english,
            "media/output/english_commentary.txt",
        )

        save_text(
            arabic,
            "media/output/arabic_commentary.txt",
        )

        save_text(
            english_tts,
            "media/output/english_tts_script.txt",
        )

        save_text(
            arabic_tts,
            "media/output/arabic_tts_script.txt",
        )

        audio_segments = None

        if ENABLE_TTS:
            audio_segments = self.tts.generate(events)

            segments_by_language = {
                language: {
                    int(segment["event_id"]): segment
                    for segment in audio_segments.get(
                        language,
                        [],
                    )
                }
                for language in ("english", "arabic")
            }

            for index, event in enumerate(events):
                event_id = int(
                    event.get("event_id", index)
                )
                event["timing_diagnostics"] = {}

                for language in ("english", "arabic"):
                    segment = segments_by_language[
                        language
                    ].get(event_id)

                    if segment is not None:
                        event["timing_diagnostics"][
                            language
                        ] = segment["diagnostics"]

                primary = (
                    event["timing_diagnostics"].get(
                        "english"
                    )
                    or event[
                        "timing_diagnostics"
                    ].get("arabic")
                )

                if primary is not None:
                    event.update(primary)

        summary = self.summary_builder.build_events(
            events
        )

        save_text(
            summary,
            "media/output/observations.txt",
        )

        if ENABLE_VIDEO_MERGE:

            if audio_segments is None:
                raise RuntimeError(
                    "Video merging requires ENABLE_TTS = True."
                )

            self.merger.merge(
                video_path,
                audio_segments["english"],
                "media/output/english_commentary.mp4",
            )

            self.merger.merge(
                video_path,
                audio_segments["arabic"],
                "media/output/arabic_commentary.mp4",
            )

        return {
            "summary": summary,
            "english": english,
            "arabic": arabic,
            "english_video":
                "media/output/english_commentary.mp4",
            "arabic_video":
                "media/output/arabic_commentary.mp4",
        }
