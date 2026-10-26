import json


class VideoCommentaryPrompt:

    ACTION_TYPES = (
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
    )

    def build(self, duration_seconds: float) -> str:

        return self.build_analysis(duration_seconds)

    def build_analysis(
        self,
        duration_seconds: float,
    ) -> str:

        duration = max(1, round(duration_seconds))

        event_count = max(
            6,
            min(14, round(duration / 3.5)),
        )

        action_types = "|".join(self.ACTION_TYPES)

        return f"""
Analyze this {duration}-second football video and extract approximately
{event_count} synchronized atomic actions.

This is the first, neutral observation pass. Do not write commentary yet and do
not identify either team by shirt colour. A separate verifier will decide team
possession from timestamped evidence images.

ACTION GROUNDING

- Watch the complete video before selecting events.
- Each event must contain exactly one action that is visibly happening now.
- Track the ball through the moments immediately before and after each cue.
- Distinguish carefully between these different facts:
  a ball being kicked, a ball travelling, a receiving touch, controlled
  possession, a loose-ball challenge, and a completed pass.
- A kick is pass_release until a receiver visibly controls it.
- Do not call a long forward ball a shot unless it is visibly directed at goal.
- Do not call movement an attack unless the ball is controlled and advancing.
- If the ball cannot be followed, use action_type "movement" or "unclear" and
  describe only visible player movement.
- Never infer an action from what happens several seconds later.

TIMING

- start_seconds is the beginning of the atomic action.
- end_seconds is when that same action finishes.
- cue_seconds is the earliest point when the neutral description is visually
  true in full.
- Treat cue_seconds as the narration start: never place it before the described
  action is visibly underway. If the exact onset is uncertain, choose a later
  timestamp within the verified action rather than anticipating it.
- cue_seconds must be between start_seconds and end_seconds.
- Normally leave at least 2.5 seconds between cues.
- Use timestamps in 0.5-second increments.
- Do not put a cue in the final 1.5 seconds.

DESCRIPTION

- description must be objective, present tense, and contain no team colour,
  team name, player name, shirt number, score, competition, or invented role.
- Describe direction or location only when visually clear.
- Do not use vague analysis such as "building patiently", "looking for space",
  "controlling the tempo", or "trying to create an opportunity".
- Good neutral descriptions state the observable action: a player releases a
  backward pass, the goalkeeper controls the ball, the ball travels long, a
  player carries down the right touchline, or two players challenge for a loose
  ball.

Return only valid JSON:

{{
  "events": [
    {{
      "event_id": 0,
      "start_seconds": 0.0,
      "end_seconds": 2.5,
      "cue_seconds": 0.5,
      "action_type": "{action_types}",
      "description": "One neutral objective visible action",
      "ball_state": "controlled|loose|in_flight|not_visible",
      "action_confidence": "low|medium|high"
    }}
  ]
}}
""".strip()

    def build_verification(
        self,
        draft_events,
    ) -> str:

        draft_json = json.dumps(
            {"events": draft_events},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        action_types = "|".join(self.ACTION_TYPES)

        return f"""
Act as a strict football video fact-checker. The attached full video and the
timestamped four-panel evidence sheets are the visual source of truth.

The draft below is untrusted. Correct it whenever the pictures disagree. Do not
preserve a draft claim merely because it sounds plausible.

DRAFT EVENTS
{draft_json}

For every event, inspect the four panels in chronological order and also check
the matching moment in the full video.

VERIFICATION METHOD

1. Locate the ball in more than one panel.
2. Decide whether it is controlled, loose, or travelling.
3. If controlled, identify the same controlling outfield player across nearby
   panels and inspect that player's shirt—not the closest player's shirt in one
   frozen image.
4. White means a predominantly white outfield shirt. Green means the bright
   green outfield shirt. Black officials and differently dressed goalkeepers
   are neither white nor green outfield possession evidence.
5. Never infer team identity from attack direction, pitch side, player count,
   earlier possession, or the draft.
6. Correct the action description as well as the possession. A pass release,
   travelling ball, reception, carry, challenge, cross, and shot are different
   actions.
7. Re-check absolute timing against the timestamp labels on the evidence
   panels. cue_seconds is when narration starts: it must not precede the
   described action becoming visibly identifiable. When the onset is uncertain,
   choose a later point inside the verified action; do not shift events by an
   estimated offset or by stretching/compressing the timeline.

POSSESSION RULES

- possession_team is "white", "green", or "unclear".
- Use a colour only when an outfield player visibly controls the ball across at
  least two nearby panels. Otherwise use "unclear".
- A goalkeeper action, loose ball, aerial ball, tackle, blocked view, or single
  ambiguous touch must not be assigned to an outfield shirt colour.
- possession_confidence is high only when both the ball and controller's shirt
  remain clear. When uncertain, lower confidence instead of guessing.

OUTPUT RULES

- Return every draft event exactly once with the same event_id.
- You may correct start_seconds, end_seconds, cue_seconds, action_type,
  description, and ball_state.
- description must remain neutral: no white, green, team identity, name, number,
  score, or invented position.
- evidence_notes must explain the concrete visible evidence and any uncertainty.
- No commentary sentences yet.

Return only valid JSON:

{{
  "events": [
    {{
      "event_id": 0,
      "start_seconds": 0.0,
      "end_seconds": 2.5,
      "cue_seconds": 0.5,
      "action_type": "{action_types}",
      "description": "Corrected neutral objective action",
      "ball_state": "controlled|loose|in_flight|not_visible",
      "action_confidence": "low|medium|high",
      "possession_team": "white|green|unclear",
      "possession_confidence": "low|medium|high",
      "evidence_notes": "Specific visual evidence for this decision"
    }}
  ]
}}
""".strip()

    def build_commentary(
        self,
        verified_events,
    ) -> str:

        public_events = []

        for event in verified_events:
            public_events.append({
                "event_id": event["event_id"],
                "cue_seconds": event["cue_seconds"],
                "action_type": event["action_type"],
                "description": event["description"],
                "ball_state": event["ball_state"],
                "action_confidence": event[
                    "action_confidence"
                ],
                "possession_team": event[
                    "possession_team"
                ],
                "possession_confidence": event[
                    "possession_confidence"
                ],
                "mention_team": bool(
                    event.get("mention_team", False)
                ),
            })

        event_json = json.dumps(
            {"events": public_events},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return f"""
Write natural live English and Modern Standard Arabic football commentary from
the verified event facts below. This is a writing-only pass. Do not reinterpret
the video, change an action, change possession, or add a future result.

VERIFIED EVENTS
{event_json}

FACTUAL RULES

- Each line must describe only that event's verified description and action.
- Do not add names, numbers, club names, positions, score, fouls, shots, saves,
  goals, or danger unless the verified description explicitly contains it.
- A pass release must not claim the pass was completed.
- A travelling or loose ball must not be described as controlled possession.
- Low-confidence events require conservative literal wording.

TEAM-NAMING RULES

- If mention_team is false, neither main nor neutral lines may contain white,
  green, a team identity, or an equivalent Arabic colour expression.
- If mention_team is true, the main lines may name exactly the verified
  possession_team once. Do not begin other lines with a shirt colour.
- The neutral alternatives must never name a team or colour.
- Do not turn every event into "White..." or "Green...". Most lines should lead
  with the action, ball, location, pressure, or goalkeeper.

STYLE

- Sound like immediate television commentary, not a tactical report.
- Use varied, direct football language specific to the action.
- Avoid robotic filler such as "continues possession", "looks to build",
  "assesses the options", "controls the tempo", and "moves the ball well".
- Do not reuse the same sentence opening in consecutive events.
- English: 3 to 8 spoken words, one short sentence.
- Arabic: 3 to 7 spoken words, one natural short sentence.
- English and Arabic must express the same fact and intensity.
- Focused delivery for routine control or backward play; building for verified
  forward movement; excited only for genuine visible danger; shouting only for
  an explicit verified shot, major save, or goal.

EXAMPLES OF TONE ONLY

- back_pass: "Back it goes." / "الكرة إلى الخلف."
- long_ball: "The keeper sends it long." / "الحارس يرسلها بعيدة."
- carry: "Driving down the right!" / "انطلاقة عبر الجهة اليمنى!"
- challenge: "A strong midfield challenge." / "صراع قوي في الوسط."
- pass_release: "A sharp ball inside." / "تمريرة سريعة إلى الداخل."

Return only valid JSON:

{{
  "events": [
    {{
      "event_id": 0,
      "english_tts": "Short live English line",
      "arabic_tts": "جملة عربية قصيرة مباشرة",
      "english_neutral_tts": "Same action without team identity",
      "arabic_neutral_tts": "نفس الحدث دون هوية الفريق",
      "intensity": "low|medium|high",
      "delivery": "focused|building|excited|shouting"
    }}
  ]
}}
""".strip()
