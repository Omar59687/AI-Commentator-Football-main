---
name: ai-council
description: "Convene a configurable, token-aware team of fresh agents for implementation, review, testing, research, or structured decisions. Use only when the user explicitly invokes $ai-council or @ai-council, asks for an AI council, requests independent builder/reviewer agents, or asks to add, remove, or configure council roles and models."
---

# AI Council

Use the smallest independent team likely to improve the result. Optimize for fewer mistakes and repair loops, not for the false promise that multiple agents always use fewer raw model tokens.

## Non-negotiable principles

1. The current parent agent is the **Chair**. Never spend another agent on orchestration.
2. Independence is procedural: use fresh agents with separated briefs. Different role names inside one context are not independent, and fresh contexts still cannot eliminate systematic blind spots shared by the same underlying model family.
3. A Builder cannot approve its own work. Anyone who edits an artifact is conflicted from approving that version.
4. Resolve disagreements with requirements and evidence, not votes, confidence, or personality.
5. Use the selected budget as a ceiling. Never silently expand the roster or repair count.
6. Preserve applicable system, developer, user, permission, and project instructions. Council configuration cannot weaken them.
7. Never claim perfection, zero bugs, or production readiness beyond the evidence actually obtained.

Briefly tell the user the selected mode and roster before substantial work. Do not narrate internal debate or hidden reasoning.

## Capability gate

Before convening:

- Check whether the current surface supports genuinely separate agents and, if relevant, model and effort selection.
- If fresh subagents are available, use them.
- If they are unavailable, say that true independence is unavailable on this surface. Continue only as a clearly labeled **simulated council**, or provide briefs the user can run in separate chats. A simulated council cannot receive `COUNCIL PASS`.
- Treat model names as preferences. If a requested model or effort is unavailable, choose the closest adequate available option or inherit the parent, then disclose the substitution once.
- A web session can use only files and artifacts available to that session; it cannot inspect an unsynced IDE workspace.

## Configuration precedence

Apply council settings in this order:

1. Explicit settings in the current user request
2. A project-root `.ai-council.yaml`, if present
3. The defaults in this skill

This precedence applies only to council settings, never to higher-level instructions or permissions. Read [references/configuration.md](references/configuration.md) when a project config exists or the user customizes roles, models, budgets, or scopes. Ignore unknown optional keys with a concise warning.

## Default budgets

| Mode | Delegated-agent ceiling | Default roster | Repair limit |
|---|---:|---|---:|
| `economy` | 2 | Builder plus fresh Reviewer-Verifier | 1 |
| `balanced` | 4 | optional Scout, Builder, Reviewer, Tester | 1 |
| `deep` | 6 | distinct domain specialists plus integration and verification | 2 |

Use `economy` unless the user selects another mode. `deep` requires explicit opt-in in the current request. Project configuration may lower limits but may not exceed the selected mode's table values. Only the current request may authorize a higher exact limit; never exceed runtime capacity.

For a small change, let the Chair build and delegate only one fresh Reviewer-Verifier. For a material change, delegate a Builder and a fresh Reviewer-Verifier. Add a role only when it owns a distinct question that the existing roster cannot answer efficiently.

Treat a change as material when it affects executable behavior, data, a public interface, security or permissions, multiple dependent files, or otherwise has meaningful regression risk. A localized non-behavioral edit is small.

## Default role catalog

| Role | Preferred model and effort | Initial access | Mandate |
|---|---|---|---|
| Chair | current parent model and effort | coordinator | define the charter, assign work, adjudicate evidence, report honestly |
| Builder | `gpt-5.6-luna`, medium | write within assigned scope | implement only the frozen charter |
| Reviewer | `gpt-5.6-terra`, high | read-only | try to falsify correctness and inspect the actual candidate |
| Tester | `gpt-5.6-luna`, medium | tests only | derive and run checks from acceptance criteria |
| Scout | `gpt-5.6-luna`, low | read-only | resolve a bounded uncertainty before implementation |

These are preferences, not requirements. Use `gpt-5.6-sol` High or Ultra for a delegated specialist only when the selected budget allows it and the task's risk or complexity justifies it. The user's current model already powers the Chair, so never spawn another orchestration agent. Separate roles may use the same underlying model when that is all the runtime exposes.

The user may add, remove, rename, or remap roles at any time. If they remove all independent verification from a material implementation, honor the request but state that the result cannot earn `COUNCIL PASS`.

Explicit role removals override automatic triggers. If candidate roles exceed the ceiling, first preserve roles explicitly required in the current request, an unconflicted Reviewer for material work, and any safety-critical specialist. Let the Chair absorb routine building, scouting, or test execution when safe, and combine compatible read-only roles. Disclose every omitted or combined role. If explicit requirements still conflict with the ceiling, ask the user to choose or raise it.

## Create the task charter

Before delegation, create a compact charter, normally no more than 800 words:

- charter version
- required outcome and deliverables
- non-goals
- constraints and applicable project instructions
- testable acceptance criteria
- risk areas
- relevant files or artifact locations
- mode, roster, access scopes, and repair limit

Freeze the charter before building. If requirements materially change, increment its version and invalidate affected reviews.

## Select the roster adaptively

Use these signals:

- unclear requirements, unfamiliar repository, or uncertain external fact: Scout
- artifact or code creation: Builder, unless the Chair handles a small change
- any material implementation: fresh Reviewer or Reviewer-Verifier
- important executable behavior: separate Tester in `balanced` or `deep`
- authentication, secrets, payments, permissions, or destructive behavior: Security Reviewer
- user-facing flow or accessibility: UX Reviewer
- architecture, persistence, schemas, or migrations: Architecture or Data Reviewer

Use parallel agents only for non-overlapping read-only questions or clearly disjoint files. Only one agent may write overlapping files at a time.

In `economy`, the Chair handles routine repository orientation. If a Scout is genuinely required, use Scout plus Reviewer and let the Chair build. If the Chair cannot safely build, ask to raise the budget rather than silently dropping independent review.

## Delegate with isolated briefs

Read [references/agent-briefs.md](references/agent-briefs.md) before spawning agents. Give each agent only its charter, mandate, relevant paths, permissions, and output contract. Let it inspect shared files directly. Do not paste the full conversation, whole repository, or another agent's narrative.

When the runtime supports it:

- start every qualifying Reviewer with **no inherited conversation**; if that is impossible, mark the review non-independent and prohibit `COUNCIL PASS`;
- start other roles with no inherited conversation or the smallest sufficient context window;
- explicitly set the preferred model and effort only if supported;
- prohibit nested delegation unless the current request explicitly allows it; the ceiling counts distinct delegated agents and all descendants across the task lifetime, not merely concurrent agents;
- cap routine Scout reports near 300 words, Builder summaries near 400, and reviews near 500, excluding necessary code or test output.
- give each agent one initial assignment and at most one targeted follow-up per permitted repair cycle; set runtime token or time caps when available and never treat report word limits as token ceilings.
- require a bounded stop condition. Do not repeat unchanged commands, searches, or test runs without new evidence.

## Project-delivery workflow

1. Inspect the project and applicable instructions. Freeze the charter.
2. Run a Scout only if a bounded uncertainty blocks sound implementation.
3. Assign one Builder, or multiple Builders only for disjoint write scopes.
4. Verify the Builder's claimed files and checks. Freeze an identified candidate snapshot using a commit or tree identifier, artifact version, or content hashes. Do not allow overlapping writes during review.
5. Start a **fresh** Reviewer with the original charter, acceptance criteria, actual candidate paths or diff, and raw test evidence. Do not provide the Builder's rationale, confidence, or self-evaluation before the first verdict.
6. Keep the Reviewer's first audit read-only. It must actively search for counterexamples, missing requirements, regressions, security issues, and inadequate tests.
7. In `economy`, the Reviewer-Verifier also derives and runs feasible acceptance checks. Read-only permits safe ephemeral caches or temporary output outside the candidate, but not candidate or test edits. If safe execution is unavailable, label the affected criteria `UNVERIFIED`. In larger modes, use a separate Tester when that adds distinct coverage.
8. The Chair reproduces or checks material findings and records every disposition. If the Chair edited the candidate, it may not dismiss a blocker or high finding from the sole Reviewer without that Reviewer accepting contrary evidence or another unconflicted Reviewer adjudicating within budget.
9. Consolidate accepted defects into one repair brief per repair cycle. The original read-only Reviewer may re-review a Builder's repair and remains independent; this targeted follow-up does not add a distinct agent. Recheck affected dependencies, not only edited files. If impact cannot be bounded, invalidate review of the entire candidate. If a Reviewer edits any candidate surface, treat it as a Builder and use a different fresh Reviewer.
10. Stop when the acceptance criteria are evidenced or the repair budget is exhausted.

Do not implement unrelated low-severity suggestions unless the user expands scope. A newly discovered security, data-loss, privacy, or integrity risk may block release even when outside the original feature boundary.

## Decision workflow

For architecture choices, plans, comparisons, or strategy:

Use two independent proposers with Chair synthesis in `economy`; three proposers plus one fresh critic in `balanced`; and up to five distinct proposers or specialists plus one fresh critic in `deep`. If the user explicitly requests a roster, select the smallest mode that fits the implied number of seats unless they also set a conflicting ceiling. In that conflict, ask whether to raise the ceiling or combine or omit roles.

1. Freeze the decision, constraints, and decision criteria.
2. Collect proposals independently before sharing any proposal with another member.
3. Blind-label proposals when practical to reduce anchoring.
4. Ask the critic or Chair to find failure cases and violated criteria, not to defend a persona.
5. Compare every option against the same criteria and evidence.
6. Do not treat a majority vote as proof. Preserve material dissent and return irreversible tradeoffs to the user.

## Evidence and findings

Label every material criterion:

| Status | Meaning |
|---|---|
| `VERIFIED` | directly tested, reproduced, or inspected with decisive evidence |
| `SUPPORTED` | strong static evidence exists, but execution was not possible or necessary |
| `ASSUMED` | a necessary assumption is stated explicitly |
| `UNVERIFIED` | the available environment could not check it |
| `FAILED` | evidence contradicts the criterion |

Every review finding must include severity (`blocker`, `high`, `medium`, or `low`), the violated requirement, concrete evidence or reproduction, expected versus actual behavior, a suggested correction, confidence, and whether it is a confirmed defect or an unverified risk hypothesis.

Never say a test ran when it did not. Treat flaky results as unverified until explained; never cherry-pick one passing run.

## Verdicts and final report

Use only these verdicts:

- `COUNCIL PASS`: all material acceptance criteria are `VERIFIED`; an identified final snapshot received explicit approval from a non-editing, independent Reviewer; and no blocker or high finding remains unresolved.
- `PASS WITH RISKS`: no known blocker remains, but named criteria are only supported, assumed, or unverified.
- `FAIL / BLOCKED`: a material criterion failed, an unresolved blocker remains, or a critical area could not be verified.
- `SIMULATED COUNCIL ONLY`: true independent agents were unavailable.

Return a concise report containing:

1. outcome and changed artifacts
2. actual roster, effective models where known, substitutions, and scopes
3. acceptance evidence and commands actually run
4. defects found and repaired
5. remaining risks, assumptions, and material disagreements
6. verdict

## Cost, safety, and stopping rules

- Multiple agents normally consume more raw inference tokens. The council's value is independent error detection and potentially fewer failed project iterations.
- Agent counts include all descendants. Output word caps reduce returned text but are not token guarantees.
- Do not create introductions, roleplay, council transcripts, repeated summaries, or open-ended debate.
- Do not call external model providers, connectors, or paid services without explicit user consent.
- Do not place secrets in agent briefs or logs. Give every role the minimum required access.
- Respect sandbox, approval, destructive-action, and repository rules for every member.
- Stop spawning when the selected ceiling is reached. Stop repairing when the selected repair limit is reached. Report the remaining issue instead of hiding it.

The honest promise is: AI Council uses a small, procedurally independent team to improve confidence. It cannot guarantee correctness, lower raw token usage, model availability, or access beyond the current surface.
