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
MAX_TEMPO = 1.35

WORD_LIMITS = {
    "edge": {"english": 10, "arabic": 9},
    "elevenlabs": {"english": 14, "arabic": 12},
}

PROVIDER_LEADS = {
    "edge": {
        "live": 0.14,
        "reaction": 0.0,
        "build_up": 0.40,
        "result": 0.0,
    },
    "elevenlabs": {
        "live": 0.10,
        "reaction": 0.0,
        "build_up": 0.35,
        "result": 0.0,
    },
}


def build_tts_text(event, language):
    line = event[f"{language}_tts"].strip()
    delivery = event.get("delivery", "focused").lower()
    tag = DELIVERY_TAGS.get(delivery, "[focused]")
    return f"{tag} {line}"


def _safe_float(value, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)

    if number != number:
        return float(fallback)

    return number


def calculate_scheduled_starts(events, provider=None):
    provider = (
        provider
        or os.getenv("TTS_PROVIDER", "elevenlabs")
    ).strip().lower()

    leads = PROVIDER_LEADS.get(
        provider,
        PROVIDER_LEADS["elevenlabs"],
    )

    starts = []

    for event in events:
        start_bound = max(
            0.0,
            _safe_float(event.get("start_seconds"), 0.0),
        )
        peak = max(
            start_bound,
            _safe_float(
                event.get(
                    "action_peak_seconds",
                    event.get("cue_seconds", start_bound),
                ),
                start_bound,
            ),
        )
        mode = str(event.get("mode", "live")).strip().lower()
        lead = leads.get(mode, leads["live"])
        peak_start = max(start_bound, peak - lead)

        proposed = _safe_float(
            event.get("commentary_start_seconds"),
            peak_start,
        )

        # Ignore stale model proposals far from the verified action peak.
        if abs(proposed - peak) > 0.75:
            proposed = peak_start

        if mode in {"reaction", "result"}:
            scheduled = max(peak, proposed)
        else:
            scheduled = min(peak_start, proposed)
            scheduled = max(start_bound, scheduled)

        starts.append(round(max(0.0, scheduled), 3))

    return starts


def _diagnostic(
    event,
    start,
    duration,
    window_end,
    provider,
):
    peak = _safe_float(
        event.get(
            "action_peak_seconds",
            event.get("cue_seconds", 0.0),
        ),
        0.0,
    )

    return {
        "action_peak_seconds": round(peak, 3),
        "scheduled_start_seconds": round(start, 3),
        "audio_duration_seconds": round(float(duration), 3),
        "window_end_seconds": round(window_end, 3),
        "provider": provider,
        "semantic_drift_seconds": round(start - peak, 3),
        # Measuring transport drift requires inspecting rendered output.
        "transport_drift_seconds": None,
    }


class TTSService:

    def generate(self, events):
        generator = TTSGenerator()
        provider = generator.provider

        directory = Path(
            "media/output/commentary_segments"
        )
        directory.mkdir(parents=True, exist_ok=True)

        generated = {
            "english": [],
            "arabic": [],
        }

        desired_starts = calculate_scheduled_starts(
            events,
            provider=provider,
        )

        limits = WORD_LIMITS.get(
            provider,
            WORD_LIMITS["elevenlabs"],
        )

        for language in ("english", "arabic"):
            lines = [
                event[f"{language}_tts"].strip()
                for event in events
            ]
            previous_end = 0.0

            for index, event in enumerate(events):
                word_limit = limits[language]

                if len(lines[index].split()) > word_limit:
                    raise ValueError(
                        f"{provider} {language} line {index} "
                        f"exceeds {word_limit} words"
                    )

                peak = _safe_float(
                    event.get(
                        "action_peak_seconds",
                        event.get("cue_seconds", 0.0),
                    ),
                    0.0,
                )
                start = max(
                    desired_starts[index],
                    previous_end + (
                        BETWEEN_LINES_SECONDS
                        if previous_end > 0
                        else 0.0
                    ),
                )

                if index + 1 < len(events):
                    next_peak = _safe_float(
                        events[index + 1].get(
                            "action_peak_seconds",
                            events[index + 1].get(
                                "cue_seconds",
                                start,
                            ),
                        ),
                        start,
                    )
                    next_boundary = (
                        next_peak - BETWEEN_LINES_SECONDS
                    )
                else:
                    next_boundary = _safe_float(
                        event.get("end_seconds"),
                        start,
                    )

                event_end = _safe_float(
                    event.get("end_seconds"),
                    next_boundary,
                )
                window_end = min(event_end, next_boundary)
                available = max(0.0, window_end - start)

                output_path = (
                    directory
                    / f"{language}_{index:03d}.mp3"
                )

                duration = generator.generate(
                    build_tts_text(event, language),
                    str(output_path),
                    language,
                    previous_text=(
                        lines[index - 1]
                        if index
                        else None
                    ),
                    next_text=(
                        lines[index + 1]
                        if index + 1 < len(events)
                        else None
                    ),
                )

                required_tempo = (
                    duration / available
                    if available > 0
                    else float("inf")
                )

                if (
                    window_end <= start
                    or required_tempo > MAX_TEMPO
                ):
                    print(
                        f"Omitting {language} segment {index}: "
                        "audio cannot fit before the next action."
                    )
                    continue

                diagnostics = _diagnostic(
                    event,
                    start,
                    duration,
                    window_end,
                    provider,
                )

                generated[language].append({
                    "path": str(output_path),
                    "start_seconds": start,
                    "end_seconds": window_end,
                    "duration": duration,
                    "action_peak_seconds": peak,
                    "diagnostics": diagnostics,
                    "event_id": int(
                        event.get("event_id", index)
                    ),
                })

                previous_end = window_end

        return generated
