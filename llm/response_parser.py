import json
import re


VALID_ACTION_TYPES = {
    "carry",
    "pass_release",
    "reception",
    "back_pass",
    "long_ball",
    "challenge",
    "clearance",
    "cross",
    "shot",
    "save",
    "restart",
    "movement",
    "unclear",
}

VALID_BALL_STATES = {
    "controlled",
    "loose",
    "in_flight",
    "not_visible",
}

VALID_CONFIDENCE_LEVELS = {
    "low",
    "medium",
    "high",
}

VALID_POSSESSION_TEAMS = {
    "white",
    "green",
    "unclear",
}

VALID_INTENSITIES = {
    "low",
    "medium",
    "high",
}

VALID_DELIVERIES = {
    "focused",
    "building",
    "excited",
    "shouting",
}

ENGLISH_COLOR_PATTERNS = {
    "white": re.compile(
        r"\bwhite(?:s|\s+(?:team|side|players?))?\b",
        flags=re.IGNORECASE,
    ),
    "green": re.compile(
        r"\bgreen(?:s|\s+(?:team|side|players?))?\b",
        flags=re.IGNORECASE,
    ),
}

ARABIC_COLOR_PATTERNS = {
    "white": re.compile(
        r"(?:الأبيض|الابيض|البيضاء|أبيض|ابيض|بيضاء|البيض)"
    ),
    "green": re.compile(
        r"(?:الأخضر|الاخضر|الخضراء|أخضر|اخضر|خضراء|الخضر)"
    ),
}

FALLBACK_LINES = {
    "carry": (
        "Advancing with the ball.",
        "تقدم واضح بالكرة.",
    ),
    "pass_release": (
        "The ball is played on.",
        "تمريرة جديدة الآن.",
    ),
    "reception": (
        "A controlled first touch.",
        "استلام جيد للكرة.",
    ),
    "back_pass": (
        "Back it goes.",
        "الكرة إلى الخلف.",
    ),
    "long_ball": (
        "The ball is sent long.",
        "كرة طويلة إلى الأمام.",
    ),
    "challenge": (
        "A firm challenge here.",
        "صراع قوي على الكرة.",
    ),
    "clearance": (
        "Cleared away from danger.",
        "إبعاد للكرة عن الخطر.",
    ),
    "cross": (
        "The ball comes across.",
        "كرة عرضية إلى الداخل.",
    ),
    "shot": (
        "The shot comes in!",
        "تسديدة قوية الآن!",
    ),
    "save": (
        "The keeper makes the save!",
        "الحارس يتصدى للكرة!",
    ),
    "restart": (
        "Play restarts here.",
        "استئناف اللعب الآن.",
    ),
    "movement": (
        "Players move into space.",
        "تحرك جديد في الملعب.",
    ),
    "unclear": (
        "Play continues here.",
        "اللعب متواصل هنا.",
    ),
}


def _load_json(text):

    if not text or not str(text).strip():
        raise ValueError(
            "Gemini returned an empty JSON response."
        )

    cleaned = str(text).strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini did not return valid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    events = result.get("events")

    if not isinstance(events, list) or not events:
        raise ValueError(
            "Gemini response must contain events."
        )

    return events


def _normalise_choice(value, allowed, default):

    choice = str(value or default).strip().lower()

    if choice not in allowed:
        return default

    return choice


def _event_times(
    event,
    index,
    duration_seconds,
):

    try:
        start = float(event["start_seconds"])
        end = float(event["end_seconds"])
        cue = float(event["cue_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Event {index} contains invalid timestamps."
        ) from exc

    if start < 0:
        raise ValueError(
            f"Event {index} has a negative start time."
        )

    if end <= start:
        raise ValueError(
            f"Event {index} must end after it starts."
        )

    if duration_seconds is not None:
        duration = float(duration_seconds)

        if duration <= 0:
            raise ValueError(
                "Video duration must be greater than zero."
            )

        if start >= duration:
            return None

        end = min(end, duration)

    cue = max(start, min(end, cue))

    return start, end, cue


def parse_analysis_response(
    text,
    duration_seconds=None,
):

    raw_events = _load_json(text)
    parsed = []

    required = {
        "start_seconds",
        "end_seconds",
        "cue_seconds",
        "action_type",
        "description",
        "ball_state",
        "action_confidence",
    }

    for index, event in enumerate(raw_events):

        if not isinstance(event, dict):
            raise ValueError(
                f"Event {index} must be an object."
            )

        missing = sorted(required.difference(event))

        if missing:
            raise ValueError(
                f"Event {index} is missing: "
                + ", ".join(missing)
            )

        times = _event_times(
            event,
            index,
            duration_seconds,
        )

        if times is None:
            continue

        start, end, cue = times
        description = str(event["description"]).strip()

        if not description:
            raise ValueError(
                f"Event {index} has an empty description."
            )

        parsed.append({
            "event_id": index,
            "start_seconds": start,
            "end_seconds": end,
            "cue_seconds": cue,
            "action_type": _normalise_choice(
                event["action_type"],
                VALID_ACTION_TYPES,
                "unclear",
            ),
            "description": description,
            "ball_state": _normalise_choice(
                event["ball_state"],
                VALID_BALL_STATES,
                "not_visible",
            ),
            "action_confidence": _normalise_choice(
                event["action_confidence"],
                VALID_CONFIDENCE_LEVELS,
                "low",
            ),
        })

    if not parsed:
        raise ValueError(
            "Gemini returned no events inside the video timeline."
        )

    parsed.sort(key=lambda item: item["cue_seconds"])

    for event_id, event in enumerate(parsed):
        event["event_id"] = event_id

    return {"events": parsed}


def parse_verification_response(
    text,
    draft_events,
    duration_seconds=None,
):

    raw_events = _load_json(text)

    raw_by_id = {}

    for raw in raw_events:
        if isinstance(raw, dict):
            try:
                raw_by_id[int(raw["event_id"])] = raw
            except (KeyError, TypeError, ValueError):
                continue

    verified = []

    for draft in draft_events:

        event_id = int(draft["event_id"])
        raw = raw_by_id.get(event_id)

        if raw is None:
            event = dict(draft)
            event.update({
                "possession_team": "unclear",
                "possession_confidence": "low",
                "evidence_notes": (
                    "The visual verifier omitted this event."
                ),
            })
            verified.append(event)
            continue

        merged = dict(draft)
        merged.update(raw)
        merged["event_id"] = event_id

        times = _event_times(
            merged,
            event_id,
            duration_seconds,
        )

        if times is None:
            continue

        start, end, cue = times

        action_type = _normalise_choice(
            merged.get("action_type"),
            VALID_ACTION_TYPES,
            draft["action_type"],
        )

        ball_state = _normalise_choice(
            merged.get("ball_state"),
            VALID_BALL_STATES,
            draft["ball_state"],
        )

        possession_team = _normalise_choice(
            merged.get("possession_team"),
            VALID_POSSESSION_TEAMS,
            "unclear",
        )

        possession_confidence = _normalise_choice(
            merged.get("possession_confidence"),
            VALID_CONFIDENCE_LEVELS,
            "low",
        )

        description = str(
            merged.get("description", draft["description"])
        ).strip()

        if not description:
            description = draft["description"]

        # Team control is not a valid claim while the ball is loose,
        # travelling, hidden, or explicitly handled by a goalkeeper.
        if (
            ball_state != "controlled"
            or re.search(
                r"\b(?:goalkeeper|keeper)\b",
                description,
                flags=re.IGNORECASE,
            )
        ):
            possession_team = "unclear"
            possession_confidence = "low"

        verified.append({
            "event_id": event_id,
            "start_seconds": start,
            "end_seconds": end,
            "cue_seconds": cue,
            "action_type": action_type,
            "description": description,
            "ball_state": ball_state,
            "action_confidence": _normalise_choice(
                merged.get("action_confidence"),
                VALID_CONFIDENCE_LEVELS,
                draft["action_confidence"],
            ),
            "possession_team": possession_team,
            "possession_confidence": (
                possession_confidence
            ),
            "evidence_notes": str(
                merged.get(
                    "evidence_notes",
                    "No visual evidence note was returned.",
                )
            ).strip(),
        })

    if not verified:
        raise ValueError(
            "The verifier returned no usable events."
        )

    verified.sort(key=lambda item: item["cue_seconds"])

    return {"events": verified}


def _mentioned_colors(text, patterns):

    return {
        color
        for color, pattern in patterns.items()
        if pattern.search(text)
    }


def _fallback_for(action_type):

    return FALLBACK_LINES.get(
        action_type,
        FALLBACK_LINES["unclear"],
    )


def parse_commentary_response(
    text,
    verified_events,
    duration_seconds=None,
):

    raw_events = _load_json(text)

    raw_by_id = {}

    for raw in raw_events:
        if isinstance(raw, dict):
            try:
                raw_by_id[int(raw["event_id"])] = raw
            except (KeyError, TypeError, ValueError):
                continue

    final_events = []

    required_text_fields = (
        "english_tts",
        "arabic_tts",
        "english_neutral_tts",
        "arabic_neutral_tts",
    )

    for event in verified_events:

        event_id = int(event["event_id"])
        raw = raw_by_id.get(event_id)

        if raw is None:
            raise ValueError(
                f"Commentary writer omitted event {event_id}."
            )

        missing = [
            field
            for field in required_text_fields
            if field not in raw
        ]

        if missing:
            raise ValueError(
                f"Commentary event {event_id} is missing: "
                + ", ".join(missing)
            )

        english = str(raw["english_tts"]).strip()
        arabic = str(raw["arabic_tts"]).strip()
        neutral_english = str(
            raw["english_neutral_tts"]
        ).strip()
        neutral_arabic = str(
            raw["arabic_neutral_tts"]
        ).strip()

        if not all((
            english,
            arabic,
            neutral_english,
            neutral_arabic,
        )):
            raise ValueError(
                f"Commentary event {event_id} has an empty line."
            )

        team = _normalise_choice(
            event.get("possession_team"),
            VALID_POSSESSION_TEAMS,
            "unclear",
        )

        mention_team = bool(
            event.get("mention_team", False)
        )

        english_claims = _mentioned_colors(
            english,
            ENGLISH_COLOR_PATTERNS,
        )
        arabic_claims = _mentioned_colors(
            arabic,
            ARABIC_COLOR_PATTERNS,
        )

        main_is_consistent = (
            team != "unclear"
            and english_claims.issubset({team})
            and arabic_claims.issubset({team})
        )

        if not mention_team or not main_is_consistent:
            english = neutral_english
            arabic = neutral_arabic

        selected_english_claims = _mentioned_colors(
            english,
            ENGLISH_COLOR_PATTERNS,
        )
        selected_arabic_claims = _mentioned_colors(
            arabic,
            ARABIC_COLOR_PATTERNS,
        )

        selected_claims_are_allowed = (
            mention_team
            and team != "unclear"
            and selected_english_claims.issubset({team})
            and selected_arabic_claims.issubset({team})
        )

        if (
            selected_english_claims
            or selected_arabic_claims
        ) and not selected_claims_are_allowed:
            english, arabic = _fallback_for(
                event["action_type"]
            )

        final = dict(event)
        final.update({
            "english_tts": english,
            "arabic_tts": arabic,
            "english_neutral_tts": neutral_english,
            "arabic_neutral_tts": neutral_arabic,
            "intensity": _normalise_choice(
                raw.get("intensity"),
                VALID_INTENSITIES,
                "low",
            ),
            "delivery": _normalise_choice(
                raw.get("delivery"),
                VALID_DELIVERIES,
                "focused",
            ),
        })

        final_events.append(final)

    final_events.sort(
        key=lambda item: item["cue_seconds"]
    )

    return {"events": final_events}
