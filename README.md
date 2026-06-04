# Tabletop LLM Agent

A minimal interactive tabletop robot-arm agent for the Humanoid intern challenge.

## What it does
- 2D tabletop environment with a robot arm, an apple, a bowl, and a cup
- Structured observations as JSON
- Structured high-level actions: `MOVE_TO`, `GRASP`, `RELEASE`
- Planning loop driven by an LLM-compatible planner
- Web UI with a chat-like task box, scene view, logs, and step/auto-run controls

## Default demo task
- `Put the apple in the bowl.`

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.server:app --reload
```
Then open http://127.0.0.1:8000

## Optional LLM wiring
By default, the project uses a deterministic planner so it works out of the box.

To use an OpenAI-compatible API, set:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

The planner will call `/chat/completions` on the configured endpoint and expect JSON output.

## Notes
- This is intentionally a small, readable harness first.
- The world backend can be replaced later with a physics simulator or Isaac Sim.
