import math
from pathlib import Path

from tts.generator import TTSGenerator


DELIVERY_TAGS = {
    "focused": "[focused]",
    "building": "[building excitement]",
    "excited": "[excited]",
    "shouting": "[shouts]",
}


BETWEEN_LINES_SECONDS = 0.12


def build_tts_text(event, language):

    line = event[f"{language}_tts"].strip()

    delivery = event.get(
        "delivery",
        "focused",
    ).lower()

    tag = DELIVERY_TAGS.get(
        delivery,
        "[focused]",
    )

    return f"{tag} {line}"


def calculate_scheduled_starts(events, video_duration=None):

    starts = []

    for event in events:

        try:
            cue = float(event["cue_seconds"])
            event_end = float(event["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Event {len(starts)} has an invalid cue or end timestamp."
            ) from exc

        if not math.isfinite(cue) or cue < 0:
            raise ValueError(
                f"Event {len(starts)} has invalid cue_seconds={cue!r}; "
                "cues must be finite and non-negative."
            )
        if not math.isfinite(event_end) or event_end <= cue:
            raise ValueError(
                f"Event {len(starts)} has invalid end_seconds={event_end!r}; "
                "event end must be finite and after its cue."
            )
        if video_duration is not None and cue > video_duration:
            raise ValueError(
                f"Event {len(starts)} cue {cue:.3f}s exceeds video duration "
                f"{video_duration:.3f}s."
            )
        if starts and cue <= starts[-1]:
            raise ValueError(
                f"Event {len(starts)} cue {cue:.3f}s is not after the "
                f"previous cue {starts[-1]:.3f}s."
            )
        starts.append(cue)

    return starts


class TTSService:

    def generate(self, events, video_duration=None):

        generator = TTSGenerator()

        segment_directory = Path(
            "media/output/commentary_segments"
        )

        segment_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if video_duration is not None:
            try:
                video_duration = float(video_duration)
            except (TypeError, ValueError) as exc:
                raise ValueError("video_duration must be a finite positive number.") from exc
            if not math.isfinite(video_duration) or video_duration <= 0:
                raise ValueError("video_duration must be a finite positive number.")

        scheduled_starts = calculate_scheduled_starts(events, video_duration)

        generated = {
            "english": [],
            "arabic": [],
        }

        for language in ("english", "arabic"):

            plain_lines = [
                event[f"{language}_tts"].strip()
                for event in events
            ]

            for index, event in enumerate(events):

                speech_text = build_tts_text(
                    event,
                    language,
                )

                previous_text = (
                    plain_lines[index - 1]
                    if index > 0
                    else None
                )

                next_text = (
                    plain_lines[index + 1]
                    if index + 1 < len(events)
                    else None
                )

                output_path = (
                    segment_directory
                    / f"{language}_{index:03d}.mp3"
                )

                duration = generator.generate(
                    speech_text,
                    str(output_path),
                    language,
                    previous_text=previous_text,
                    next_text=next_text,
                )

                start = scheduled_starts[index]

                if index + 1 < len(events):
                    end = (
                        scheduled_starts[index + 1]
                        - BETWEEN_LINES_SECONDS
                    )
                else:
                    end = float(
                        event["end_seconds"]
                    )

                if video_duration is not None:
                    end = min(end, video_duration)

                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(
                        f"{language} event {index} returned invalid audio "
                        f"duration {duration!r}; duration must be finite and positive."
                    )
                available = end - start
                if not math.isfinite(end) or available <= 0:
                    raise ValueError(
                        f"{language} event {index} has no valid narration "
                        f"window: {start:.3f}s to {end!r}s."
                    )
                if duration > available:
                    raise ValueError(
                        f"{language} event {index} audio cannot fit its "
                        f"narration window: path={output_path}, "
                        f"window={start:.3f}s-{end:.3f}s, "
                        f"duration={duration:.3f}s, "
                        f"available={available:.3f}s."
                    )

                generated[language].append({
                    "path": str(output_path),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration": duration,
                })

                print(
                    f"{language} segment {index}: "
                    f"cue={event['cue_seconds']:.2f}s, "
                    f"play={start:.2f}s, "
                    f"window_end={end:.2f}s"
                )

        return generated
