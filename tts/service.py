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
        provider = generator.provider
        directory = Path("media/output/commentary_segments")
        directory.mkdir(parents=True, exist_ok=True)
        generated = {"english": [], "arabic": []}
        for language in ("english", "arabic"):
            lines = [event[f"{language}_tts"].strip() for event in events]
            previous_end = 0.0
            for index, event in enumerate(events):
                limit = 14 if provider == "elevenlabs" and language == "english" else 12 if provider == "elevenlabs" else 10 if language == "english" else 9
                if len(lines[index].split()) > limit:
                    raise ValueError(f"{provider} {language} line {index} exceeds {limit} words")
                peak = float(event.get("action_peak_seconds", event.get("cue_seconds", 0.0)))
                start = max(0.0, float(event.get("commentary_start_seconds", peak)), previous_end)
                next_peak = float(events[index + 1].get("action_peak_seconds", events[index + 1].get("cue_seconds", start + 1.0))) if index + 1 < len(events) else float(event.get("end_seconds", start + 1.0))
                window_end = max(start + 0.01, min(float(event.get("end_seconds", next_peak)), next_peak))
                output = directory / f"{language}_{index:03d}.mp3"
                duration = generator.generate(build_tts_text(event, language), str(output), language, previous_text=lines[index - 1] if index else None, next_text=lines[index + 1] if index + 1 < len(events) else None)
                if start + duration >= window_end:
                    print(f"Omitting {language} segment {index}: audio crosses next event")
                    continue
                diagnostics = {"action_peak_seconds": peak, "scheduled_start_seconds": start, "audio_duration_seconds": duration, "window_end_seconds": window_end, "provider": provider, "semantic_drift_seconds": start - peak, "transport_drift_seconds": None}
                generated[language].append({"path": str(output), "start_seconds": start, "end_seconds": start + duration, "duration": duration, "action_peak_seconds": peak, "diagnostics": diagnostics, "event_id": int(event.get("event_id", index))})
                previous_end = start + duration
        return generated
