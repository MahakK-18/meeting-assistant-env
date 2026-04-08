# 🗒️ Meeting Assistant RL Environment

**Team:** The Logic Loop  
**Hackathon:** Meta PyTorch OpenEnv Hackathon × SST 2026

---

## What is this?

An **OpenEnv RL environment** where an agent learns to extract structured information from raw meeting notes — specifically a concise **summary** and a list of **action items**.

This is useful for training LLMs to become better meeting assistants: structured, precise, and actionable in their outputs.

---

## Environment Design

### Observation
The agent receives raw, unstructured meeting notes at each step:
```
"Client wants UI redesign by Friday. Mahak will handle backend updates.
 Team needs to fix login bug. Follow-up meeting next week."
```

### Action
The agent responds with:
- `summary` — a concise 1–2 sentence summary of the meeting
- `action_items` — a list of concrete tasks identified from the notes

### Reward (0.0 – 1.0)
| Component | Max | How it's scored |
|---|---|---|
| Summary quality | 0.5 | Keyword overlap with reference summary |
| Action item recall | 0.5 | How many reference actions the agent covered |

---

## Quick Start

```bash
pip install openenv-core transformers torch
```

```python
from meeting_assistant_env.server.meeting_environment import MeetingEnvironment
from meeting_assistant_env.models import MeetingAction

env = MeetingEnvironment(max_turns=3)
obs = env.reset()

print(obs.meeting_notes)
# → "Client wants UI redesign by Friday. ..."

result = env.step(MeetingAction(
    summary="UI redesign due Friday; backend and bug fixes assigned.",
    action_items=[
        "Complete UI redesign by Friday",
        "Mahak to handle backend updates",
        "Fix login bug",
        "Schedule follow-up meeting"
    ]
))

print(result.reward)   # → 0.85
print(result.info)     # → {"summary_score": 0.4, "action_item_score": 0.45, ...}
```

---

## Project Structure

```
meeting_assistant_env/
├── __init__.py                        # Public exports
├── models.py                          # MeetingAction, MeetingObservation
├── client.py                          # MeetingAssistantEnv (EnvClient)
├── openenv.yaml                       # Environment manifest
├── pyproject.toml                     # Dependencies
├── demo_notebook.ipynb                # Full demo: naive agent vs transformer agent
└── server/
    ├── meeting_environment.py         # Core RL logic, reward function
    ├── app.py                         # FastAPI entrypoint
    ├── requirements.txt
    └── Dockerfile
```

---

## Two Agents Compared (in demo notebook)

| Agent | Approach | Typical Reward |
|---|---|---|
| Naive (baseline) | Keyword matching | ~0.4–0.6 |
| Transformer | DistilBART + keyword extraction | ~0.6–0.85 |

The transformer agent builds directly on the original hackathon project — the same `sshleifer/distilbart-cnn-12-6` model, now wrapped inside a proper RL loop.

---

## Deploy to Hugging Face Spaces

```bash
pip install openenv-core
openenv push --repo-id your-username/meeting-assistant-env
```

---

## Why this problem?

Every team deals with meeting notes. Most action items get lost. An RL environment that trains agents to reliably extract structured outputs from messy human text has real-world value — and it's a natural fit for LLM post-training with GRPO/TRL.
