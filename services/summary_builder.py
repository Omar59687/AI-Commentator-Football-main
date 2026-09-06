class SummaryBuilder:

    def build_events(self, events):

        lines = [
            "VIDEO-GROUNDED EVENT TIMELINE",
            "",
        ]

        for event in events:

            event_id = int(
                event.get("event_id", 0)
            )

            start = float(
                event.get("start_seconds", 0)
            )

            end = float(
                event.get("end_seconds", start)
            )

            cue = float(event.get("cue_seconds", start))
            action_peak = float(event.get("action_peak_seconds", cue))
            scheduled_start = float(event.get("scheduled_start_seconds", event.get("commentary_start_seconds", cue)))
            mode = str(event.get("mode", "live"))
            provider = str(event.get("provider", "unavailable"))
            duration = event.get("audio_duration_seconds")
            window_end = event.get("window_end_seconds")
            semantic_drift = event.get("semantic_drift_seconds")
            transport_drift = event.get("transport_drift_seconds")

            description = str(
                event.get(
                    "description",
                    "Visible play continues.",
                )
            )

            intensity = str(
                event.get("intensity", "low")
            ).lower()

            possession = str(
                event.get("possession_team", "unclear")
            ).lower()

            possession_confidence = str(
                event.get(
                    "possession_confidence",
                    "low",
                )
            ).lower()

            verification_source = str(
                event.get(
                    "verification_source",
                    "unavailable",
                )
            )

            action_type = str(
                event.get("action_type", "unclear")
            ).lower()

            ball_state = str(
                event.get("ball_state", "not_visible")
            ).lower()

            mention_team = bool(
                event.get("mention_team", False)
            )

            gemini_team = str(
                event.get(
                    "gemini_possession_team",
                    "unclear",
                )
            ).lower()

            local_team = str(
                event.get(
                    "local_possession_team",
                    "unclear",
                )
            ).lower()

            evidence_notes = str(
                event.get(
                    "evidence_notes",
                    "No Gemini evidence note.",
                )
            )

            local_evidence = str(
                event.get(
                    "local_evidence",
                    "No local evidence note.",
                )
            )

            lines.append(
                f"Event {event_id:02d} | "
                f"{start:05.1f}s-{end:05.1f}s "
                f"(commentary cue {cue:05.1f}s, action peak {action_peak:05.1f}s, "
                f"scheduled start {scheduled_start:05.1f}s, mode={mode}, provider={provider}, "
                f"audio={duration}, window={window_end}, semantic drift={semantic_drift}, transport drift={transport_drift}) "
                f"[{intensity}] "
                f"[action={action_type}, ball={ball_state}] "
                f"[possession={possession}, "
                f"confidence={possession_confidence}, "
                f"source={verification_source}, "
                f"spoken={mention_team}]: "
                f"{description}"
            )

            lines.append(
                "  Verification: "
                f"Gemini={gemini_team}; "
                f"local={local_team}; "
                f"Gemini evidence={evidence_notes}; "
                f"local evidence={local_evidence}"
            )

        return "\n".join(lines)

    def build(self, observations):

        lines = []

        lines.append("MATCH OBSERVATIONS\n")

        for obs in observations:

            line = (
                f"Frame {obs['frame']}: "
                f"{obs['players']} players detected, "
                f"{obs['referees']} referees detected, "
            )

            if obs["ball_detected"]:
                line += "ball visible"
            else:
                line += "ball not visible"

            if obs["closest_player"] is not None:
                line += (
                    f", closest player to ball: "
                    f"{obs['closest_player']}"
                )

            lines.append(line)

        return "\n".join(lines)
