from pathlib import Path
import os

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


def calculate_scheduled_starts(events, provider=None):
    provider = (provider or os.getenv("TTS_PROVIDER", "elevenlabs")).lower()
    starts = []
    previous = 0.0
    for event in events:
        peak = float(event.get("action_peak_seconds", event.get("cue_seconds", 0.0)))
        lead = {"live": 0.10, "reaction": 0.02, "build_up": 0.35, "result": 0.0}.get(str(event.get("mode", "live")).lower(), 0.10)
        start = max(0.0, peak - lead, previous)
        starts.append(start)
        previous = start
    return starts


def _diagnostic(event, start, duration, window_end, provider):
    peak = float(event.get("action_peak_seconds", event.get("cue_seconds", 0.0)))
    return {"action_peak_seconds": peak, "scheduled_start_seconds": round(start, 3), "audio_duration_seconds": round(float(duration), 3), "window_end_seconds": round(window_end, 3), "provider": provider, "semantic_drift_seconds": round(start - peak, 3), "transport_drift_seconds": 0.0}



class TTSService:

    def generate(self, events):

        generator = TTSGenerator()

        segment_directory = Path(
            "media/output/commentary_segments"
        )

        segment_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        scheduled_starts = (
            calculate_scheduled_starts(events)
        )

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
                peak = float(event.get("action_peak_seconds", event.get("cue_seconds", start)))

                if index + 1 < len(events):
                    end = (
                        scheduled_starts[index + 1]
                        - BETWEEN_LINES_SECONDS
                    )
                else:
                    end = float(
                        event["end_seconds"]
                    )

                # Never give the merger an invalid window.
                end = max(start + 0.25, min(end, float(event.get("end_seconds", end))))
                if index + 1 < len(events):
                    end = min(end, scheduled_starts[index + 1])
                window_end = max(start + 0.25, end)

                generated[language].append({
                    "path": str(output_path),
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration": duration,
                    "action_peak_seconds": peak,
                    "diagnostics": _diagnostic(event, start, duration, window_end, generator.provider),
                })

                print(
                    f"{language} segment {index}: "
                    f"cue={event['cue_seconds']:.2f}s, "
                    f"play={start:.2f}s, "
                    f"window_end={end:.2f}s"
                )

        return generated
