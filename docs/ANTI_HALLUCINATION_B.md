# ANTI_HALLUCINATION.md
# Behaviour Rules for the Coding Agent

This repository is being built under strict file ownership and interface contracts.

`WORKPLAN.md` and `CONTRACTS.md` are the source of truth.

Your job is NOT to redesign the project.
Your job is ONLY to complete the task requested in the current prompt.

---

# 1. Never change architecture

Do NOT:

- redesign the architecture
- refactor unrelated files
- rename folders
- rename modules
- move files
- introduce new abstractions
- create helper frameworks
- replace existing implementations
- reorganise the repository

If the requested task can be completed inside existing files,
DO NOT suggest architectural improvements.

---

# 2. Never edit files outside the requested scope

Only edit files explicitly requested.

If additional files appear necessary:

STOP.

Explain why they are needed.

Wait for approval.

Never make those edits automatically.

---

# 3. Respect WORKPLAN.md

Follow file ownership exactly.

Never edit files owned by the other track.

Never touch:

- backend/schemas.py
- backend/tools/base.py
- docs/CONTRACTS.md

unless explicitly instructed.

---

# 4. Respect CONTRACTS.md

Treat every interface as frozen.

Never:

- rename ToolResult fields
- rename ToolContext fields
- change tool names
- modify endpoint schemas
- modify Pydantic models
- change artifact keys
- invent new contracts

Use exactly what CONTRACTS.md specifies.

---

# 5. Never implement features not requested

If asked to implement

feature_engineer.py

DO NOT also implement

- rules.py
- risk.py
- Streamlit UI
- tests
- documentation
- planner changes

Complete ONLY the requested task.

---

# 6. No speculative coding

Never write code for assumptions.

If information is missing:

Ask.

Do not guess.

Do not fabricate behaviour.

---

# 7. No silent dependency additions

Never install or introduce libraries unless explicitly requested.

Never modify:

requirements.txt

requirements-data.txt

package.json

pyproject.toml

unless instructed.

---

# 8. Never change public interfaces

Never rename

functions

classes

methods

arguments

return types

unless explicitly instructed.

Internal implementation may change.

Interfaces must remain stable.

---

# 9. Preserve existing code

If editing an existing file:

Preserve

comments

logging

docstrings

public APIs

unless explicitly asked to modify them.

---

# 10. Do not optimise unless asked

Avoid:

performance improvements

clean-up

formatting

lint fixes

type hint rewrites

variable renaming

code style changes

unless they are required to complete the requested task.

---

# 11. No placeholder implementations

Do not write

TODO

pass

fake implementations

dummy returns

mock values

unless explicitly requested.

Return production-ready code.

---

# 12. Minimise changes

The preferred solution is the one with the smallest correct diff.

Avoid touching unrelated lines.

---

# 13. Explain before changing behaviour

If the requested change would alter behaviour outside the requested scope:

STOP.

Explain:

- what changes
- why
- which files are affected

Wait for confirmation.

---

# 14. Preserve backwards compatibility

Existing code that currently works should continue working.

Never break existing callers.

---

# 15. Never duplicate existing logic

Search existing files before writing new code.

Reuse existing utilities where possible.

Do not create duplicate implementations.

---

# 16. Output format

For every task provide:

1. Files modified
2. Summary of changes
3. Why those files only
4. Any assumptions made

Nothing else.

---

# 17. If uncertain

If confidence is below 95%:

Ask a question.

Do not invent an implementation.

---

# 18. Primary objective

Correctness > completeness.

Completeness > optimisation.

Optimisation > elegance.

Architecture changes are NEVER the objective unless explicitly requested.

---

# 19. Success criterion

The task is successful if:

- only requested files changed
- CONTRACTS.md remains valid
- WORKPLAN.md ownership remains respected
- no unrelated code changed
- project behaviour outside the requested scope is unchanged