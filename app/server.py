from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .planner import Planner
from .world import TabletopWorld

app = FastAPI(title="Tabletop LLM Agent")

world = TabletopWorld()
planner = Planner()

world.reset()

plan_queue: List[dict] = []
conversation: List[str] = []

last_reasoning: str = ""
last_subgoal: str = ""
last_action: Optional[dict] = None
last_result: Optional[dict] = None


class TaskRequest(BaseModel):
    task: str


class StepRequest(BaseModel):
    auto: bool = False
    max_steps: int = 12

class AddObjectRequest(BaseModel):
    kind: Optional[str] = None


def _reset_runtime_state() -> None:
    global plan_queue, conversation, last_reasoning, last_subgoal, last_action, last_result
    plan_queue = []
    conversation = []
    last_reasoning = ""
    last_subgoal = ""
    last_action = None
    last_result = None


def _extract_actions_from_plan(plan: Any) -> tuple[List[dict], str, str]:
    """
    Supports:
    - {"reasoning": ..., "subgoal": ..., "action": {...}}
    - {"reasoning": ..., "subgoal": ..., "plan": [{...}, {...}]}
    """
    reasoning = ""
    subgoal = ""
    actions: List[dict] = []

    if not isinstance(plan, dict):
        return actions, reasoning, subgoal

    reasoning = str(plan.get("reasoning", "") or "")
    subgoal = str(plan.get("subgoal", "") or "")

    if isinstance(plan.get("plan"), list):
        for item in plan["plan"]:
            if isinstance(item, dict):
                actions.append(item)

    elif isinstance(plan.get("action"), dict):
        actions.append(plan["action"])

    return actions, reasoning, subgoal


def _plan_next_actions() -> bool:
    """
    Fill plan_queue from planner if it's empty.
    Returns True if we got at least one action.
    """
    global plan_queue, last_reasoning, last_subgoal

    plan = planner.plan(world.task, world.observe(), conversation[-8:])
    actions, reasoning, subgoal = _extract_actions_from_plan(plan)

    if reasoning:
        world.log.append(f"Planner: {reasoning}")
        last_reasoning = reasoning

    if subgoal:
        world.log.append(f"Subgoal: {subgoal}")
        last_subgoal = subgoal

    if actions:
        plan_queue = actions
        return True

    world.log.append(f"Planner returned invalid action: {plan}")
    return False


def _execute_one_action() -> dict:
    global last_action, last_result

    action = plan_queue.pop(0)
    last_action = action
    last_result = world.execute(action)
    conversation.append(f"Planner action: {json.dumps(action)}")
    return last_result


def _payload() -> dict:
    return {
        "ok": True,
        "done": world.done,
        "observation": world.observe(),
        "render": world.render_events(),
        "plan_queue": plan_queue,
        "last_reasoning": last_reasoning,
        "last_subgoal": last_subgoal,
        "last_action": last_action,
        "last_result": last_result,
        "log": world.log,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


@app.get("/api/state")
def state():
    return _payload()


@app.post("/api/reset")
def reset():
    _reset_runtime_state()
    obs = world.reset()
    return {
        "ok": True,
        "observation": obs,
        "render": world.render_events(),
        "plan_queue": plan_queue,
        "last_reasoning": last_reasoning,
        "last_subgoal": last_subgoal,
        "last_action": last_action,
        "last_result": last_result,
        "log": world.log,
    }


@app.post("/api/task")
def set_task(req: TaskRequest):
    global plan_queue, last_reasoning, last_subgoal, last_action, last_result

    world.set_task(req.task)
    conversation.append(f"User: {req.task}")

    plan_queue = []
    last_reasoning = ""
    last_subgoal = ""
    last_action = None
    last_result = None

    return _payload()


@app.post("/api/step")
def step(req: StepRequest):
    if world.done:
        return _payload()

    if not plan_queue:
        if not _plan_next_actions():
            return {
                "ok": False,
                "error": "Planner failed to provide a valid action.",
                **_payload(),
            }

    _execute_one_action()

    steps = 1
    while req.auto and not world.done and steps < req.max_steps:
        if not plan_queue and not _plan_next_actions():
            break
        _execute_one_action()
        steps += 1

    return _payload()

@app.post("/api/add_object")
def add_object(req: AddObjectRequest):
    if req.kind:
        # TODO: support specifying object type and properties in the request
        # For now, just call the random version
        added = world.add_random_object()
    else:
        added = world.add_random_object()

    return {
        "ok": True,
        "added": added,
        "observation": world.observe(),
        "render": world.render_events(),
        "plan_queue": plan_queue,
        "last_reasoning": last_reasoning,
        "last_subgoal": last_subgoal,
        "last_action": last_action,
        "last_result": last_result,
        "log": world.log,
    }


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tabletop LLM Agent</title>
  <style>
    :root {
      --bg:#0f172a;
      --panel:#111827;
      --panel2:#0b1220;
      --text:#e5e7eb;
      --muted:#94a3b8;
      --line:#1f2937;
      --accent:#22c55e;
    }
    * { box-sizing: border-box; }
    body {
      margin:0;
      font-family: Inter, Arial, sans-serif;
      background: linear-gradient(180deg, #0f172a, #020617);
      color: var(--text);
    }
    .wrap {
      display:grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap:16px;
      padding:16px;
      height:100vh;
    }
    .card {
      background: rgba(15,23,42,0.85);
      border:1px solid var(--line);
      border-radius:18px;
      padding:14px;
      box-shadow: 0 8px 30px rgba(0,0,0,.25);
      min-width: 0;
    }
    h1,h2 { margin:0 0 10px 0; }
    h1 { font-size: 22px; }
    h2 { font-size: 16px; color: #dbeafe; }
    .scene {
      width:100%;
      aspect-ratio: 16/10;
      background: linear-gradient(180deg, #1f2937, #111827);
      border-radius: 16px;
      border:1px solid #334155;
      position:relative;
      overflow:hidden;
    }
    .legend {
      display:flex;
      gap:12px;
      flex-wrap:wrap;
      color: var(--muted);
      font-size:12px;
      margin-top:8px;
    }
    .pill {
      padding:4px 8px;
      border-radius:999px;
      background:#111827;
      border:1px solid #334155;
    }
    .row {
      display:flex;
      gap:10px;
      align-items:center;
      margin-top:10px;
      flex-wrap:wrap;
    }
    input[type=text] {
      flex:1;
      min-width: 260px;
      background:#0b1220;
      color:var(--text);
      border:1px solid #334155;
      border-radius:12px;
      padding:12px 14px;
      outline:none;
    }
    button {
      border:0;
      border-radius:12px;
      padding:11px 14px;
      cursor:pointer;
      font-weight:600;
    }
    .primary { background: #22c55e; color:#052e16; }
    .secondary { background:#1e293b; color:var(--text); border:1px solid #334155; }
    .log, .obs, .meta {
      background: var(--panel2);
      border:1px solid #334155;
      border-radius:14px;
      padding:12px;
      overflow:auto;
      white-space:pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size:12px;
      line-height:1.45;
    }
    .log, .obs { height: 220px; }
    .meta { height: 112px; }
    .split {
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap:12px;
      margin-top: 12px;
    }
    .muted { color: var(--muted); font-size:12px; }
    .status { margin-top:8px; color:#bbf7d0; }
    .label {
      display:inline-block;
      margin-bottom:6px;
      font-size:12px;
      color:#cbd5e1;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    svg text {
      font-size: 13px;
      font-weight: 700;
      paint-order: stroke;
      stroke: rgba(15,23,42,0.7);
      stroke-width: 3px;
      stroke-linejoin: round;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Tabletop LLM Agent</h1>
      <div class="muted">Type a task, let the planner choose actions, and watch the world update.</div>

      <div id="scene" class="scene"></div>

      <div class="legend">
        <span class="pill">LLM planner</span>
        <span class="pill">Structured actions</span>
        <span class="pill">Observe → Act → Replan</span>
      </div>

      <div class="row">
        <input id="task" type="text" value="Put the apple in the bowl." />
        <button class="primary" onclick="setTask()">Set task</button>
        <button class="secondary" onclick="step(false)">Step</button>
        <button class="secondary" onclick="step(true)">Auto run</button>
        <button class="secondary" onclick="resetWorld()">Reset</button>
        <button class="secondary" onclick="addRandomObject()">Add Object</button>
      </div>

      <div id="status" class="status"></div>
    </div>

    <div class="card">
      <div class="split">
        <div>
          <div class="label">Observation</div>
          <div id="obs" class="obs"></div>
        </div>
        <div>
          <div class="label">Agent state</div>
          <div id="meta" class="meta"></div>
        </div>
      </div>
      <div style="height:12px"></div>
      <div class="label">Log</div>
      <div id="log" class="log"></div>
    </div>
  </div>

<script>
let state = null;

function esc(s) {
  return String(s)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;');
}

function n(v, fallback = 0) {
  const x = Number(v);
  return Number.isFinite(x) ? x : fallback;
}

function isContainer(name, obj) {
  const lower = String(name || "").toLowerCase();
  const category = String((obj && obj.category) || "").toLowerCase();
  return category === "container" || ["bowl", "plate", "cup", "box", "tray"].includes(lower);
}

function drawObject(name, obj) {
  if (!obj || obj.visible === false) return "";

  const x = n(obj.x, 0);
  const y = n(obj.y, 0);
  const size = Math.max(8, n(obj.size, 18));
  const color = obj.color || "#9ca3af";
  const lower = String(name || "").toLowerCase();
  const held = !!obj.held;
  const heldStroke = held ? 'stroke-dasharray="5 4" stroke-width="3"' : 'stroke-width="2"';

  const labelX = x - Math.max(18, size);
  const labelY = y - Math.max(18, size) - 6;

  if (lower === "bowl" || lower.includes("bowl")) {
    return `
      <ellipse cx="${x}" cy="${y + size * 0.55}" rx="${size * 1.05}" ry="${size * 0.5}" fill="#000" opacity="0.18"/>
      <path d="M ${x - size} ${y} Q ${x} ${y + size} ${x + size} ${y} Z"
            fill="${color}" opacity="0.95" stroke="#bfdbfe" ${heldStroke}/>
      <text x="${labelX}" y="${labelY}" fill="#bfdbfe">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "plate" || lower.includes("plate")) {
    return `
      <ellipse cx="${x}" cy="${y + size * 0.25}" rx="${size * 1.2}" ry="${size * 0.52}" fill="#000" opacity="0.16"/>
      <ellipse cx="${x}" cy="${y}" rx="${size * 1.15}" ry="${size * 0.5}"
               fill="${color}" opacity="0.95" stroke="#e2e8f0" ${heldStroke}/>
      <ellipse cx="${x}" cy="${y}" rx="${size * 0.55}" ry="${size * 0.20}"
               fill="none" stroke="#f8fafc" stroke-opacity="0.7" stroke-width="2"/>
      <text x="${labelX}" y="${labelY}" fill="#e2e8f0">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "cup" || lower.includes("cup")) {
    return `
      <ellipse cx="${x}" cy="${y + size * 0.45}" rx="${size * 0.85}" ry="${size * 0.5}" fill="#000" opacity="0.18"/>
      <rect x="${x - size}" y="${y - size}" width="${size * 2}" height="${size * 2}" rx="${size * 0.45}"
            fill="${color}" opacity="0.95" stroke="#fef3c7" ${heldStroke}/>
      <text x="${labelX}" y="${labelY}" fill="#fef3c7">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  return `
    <ellipse cx="${x}" cy="${y + size * 0.55}" rx="${size * 0.85}" ry="${size * 0.35}" fill="#000" opacity="0.18"/>
    <circle cx="${x}" cy="${y}" r="${size}" fill="${color}" stroke="#fecaca" ${heldStroke}/>
    <text x="${labelX}" y="${labelY}" fill="#fecaca">${esc(name)}${held ? " (held)" : ""}</text>
  `;
}

function renderScene(r) {
  const w = r.width || 800;
  const h = r.height || 520;
  const robot = r.robot || {};
  const objs = r.objects || {};
  const bowlArea = r.bowl_area || null;

  const robotX = n(robot.x, n(robot.pos && robot.pos[0], w * 0.22));
  const robotY = n(robot.y, n(robot.pos && robot.pos[1], h * 0.50));

  const objectLayers = Object.entries(objs)
    .map(([name, obj]) => {
      const px = n(obj.x, n((obj.pos && obj.pos[0]) != null ? obj.pos[0] * w : 0, 0));
      const py = n(obj.y, n((obj.pos && obj.pos[1]) != null ? obj.pos[1] * h : 0, 0));
      const enriched = {
        ...obj,
        x: px,
        y: py,
      };
      return drawObject(name, enriched);
    })
    .join("\n");

  const bowlZone = bowlArea
    ? `<rect x="${n(bowlArea.x,0)}" y="${n(bowlArea.y,0)}" width="${n(bowlArea.w,0)}" height="${n(bowlArea.h,0)}" rx="18" fill="#2563eb" opacity="0.20" stroke="#60a5fa" stroke-width="2"/>
       <text x="${n(bowlArea.x,0)+12}" y="${n(bowlArea.y,0)+22}" fill="#bfdbfe">goal zone</text>`
    : "";

  const svg = `
  <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="tableGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#334155"/>
        <stop offset="100%" stop-color="#1f2937"/>
      </linearGradient>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000" flood-opacity="0.28"/>
      </filter>
    </defs>

    <rect x="0" y="0" width="${w}" height="${h}" fill="url(#tableGrad)"/>
    <rect x="20" y="20" width="${w-40}" height="${h-40}" rx="24" fill="#0f172a" opacity="0.28" stroke="#475569"/>

    ${bowlZone}

    <g filter="url(#shadow)">
      <circle cx="${robotX}" cy="${robotY}" r="18" fill="#22c55e" stroke="#bbf7d0" stroke-width="3"/>
      <rect x="${robotX - 8}" y="${robotY - 48}" width="16" height="34" rx="8" fill="#86efac"/>
      <line x1="${robotX}" y1="${robotY - 10}" x2="${robotX + 34}" y2="${robotY - 22}" stroke="#d1fae5" stroke-width="5" stroke-linecap="round"/>
      <line x1="${robotX + 34}" y1="${robotY - 22}" x2="${robotX + 54}" y2="${robotY - 14}" stroke="#d1fae5" stroke-width="5" stroke-linecap="round"/>
      <text x="${robotX - 12}" y="${robotY + 40}" fill="#dcfce7">robot</text>
    </g>

    ${objectLayers}
  </svg>`;

  document.getElementById("scene").innerHTML = svg;
}

function render(obs, renderData, log, meta) {
  state = { obs, renderData, log, meta };
  document.getElementById("obs").textContent = JSON.stringify(obs, null, 2);
  document.getElementById("log").textContent = (log || []).join("\n");
  document.getElementById("meta").textContent = JSON.stringify(meta || {}, null, 2);
  renderScene(renderData || {});

  const status = document.getElementById("status");
  if (obs && obs.done) {
    status.textContent = obs.success ? "Task complete." : "Task finished.";
  } else if (obs) {
    const holding = obs.robot && obs.robot.holding ? obs.robot.holding : "nothing";
    status.textContent = "Step " + obs.step + " | holding: " + holding;
  } else {
    status.textContent = "";
  }
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  return await res.json();
}

async function refreshState() {
  const res = await fetch("/api/state");
  const data = await res.json();
  render(data.observation, data.render, data.log, {
    last_reasoning: data.last_reasoning || "",
    last_subgoal: data.last_subgoal || "",
    last_action: data.last_action || null,
    last_result: data.last_result || null,
    plan_queue: data.plan_queue || []
  });
}

async function resetWorld() {
  const data = await post("/api/reset", {});
  render(data.observation, data.render, data.log, {
    last_reasoning: data.last_reasoning || "",
    last_subgoal: data.last_subgoal || "",
    last_action: data.last_action || null,
    last_result: data.last_result || null,
    plan_queue: data.plan_queue || []
  });
}

async function setTask() {
  const task = document.getElementById("task").value;
  const data = await post("/api/task", { task });
  render(data.observation, data.render, data.log, {
    last_reasoning: data.last_reasoning || "",
    last_subgoal: data.last_subgoal || "",
    last_action: data.last_action || null,
    last_result: data.last_result || null,
    plan_queue: data.plan_queue || []
  });
}

async function addRandomObject() {
  const data = await post('/api/add_object', {});
  render(
    data.observation,
    data.render,
    data.log,
    {
      last_reasoning: data.last_reasoning || "",
      last_subgoal: data.last_subgoal || "",
      last_action: data.last_action || null,
      last_result: data.last_result || null,
      plan_queue: data.plan_queue || []
    }
  );
}

async function step(auto) {
  const data = await post("/api/step", { auto: auto, max_steps: 12 });
  render(data.observation, data.render, data.log, {
    last_reasoning: data.last_reasoning || "",
    last_subgoal: data.last_subgoal || "",
    last_action: data.last_action || null,
    last_result: data.last_result || null,
    plan_queue: data.plan_queue || []
  });
}

refreshState();
</script>
</body>
</html>
"""