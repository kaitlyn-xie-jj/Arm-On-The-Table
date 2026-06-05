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

class MoveObjectRequest(BaseModel):
    name: str
    x: float
    y: float


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

# @app.post("/api/add_object")
# def add_object(req: AddObjectRequest):
#     if req.kind:
#         # TODO: support specifying object type and properties in the request
#         # For now, just call the random version
#         added = world.add_random_object()
#     else:
#         added = world.add_random_object()

#     return {
#         "ok": True,
#         "added": added,
#         "observation": world.observe(),
#         "render": world.render_events(),
#         "plan_queue": plan_queue,
#         "last_reasoning": last_reasoning,
#         "last_subgoal": last_subgoal,
#         "last_action": last_action,
#         "last_result": last_result,
#         "log": world.log,
#     }

@app.post("/api/add_object")
def add_object(req: AddObjectRequest):
    global plan_queue, last_reasoning, last_subgoal, last_action, last_result

    added = world.add_random_object()

    plan_queue = []
    last_reasoning = ""
    last_subgoal = ""
    last_action = None
    last_result = None

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

@app.post("/api/move_object")
def move_object(req: MoveObjectRequest):
    global plan_queue, last_reasoning, last_subgoal, last_action, last_result

    ok, message = world.move_object(req.name, req.x, req.y)

    # World changes from manual moves should invalidate the current plan and prompt replanning
    plan_queue = []
    last_reasoning = ""
    last_subgoal = ""
    last_action = None
    last_result = None

    return {
        "ok": ok,
        "message": message,
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
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-8px); }
    }
    @keyframes pulse-glow {
      0%, 100% { filter: drop-shadow(0 0 8px rgba(34, 197, 94, 0.3)); }
      50% { filter: drop-shadow(0 0 16px rgba(34, 197, 94, 0.5)); }
    }
    @keyframes grid-fade {
      0% { opacity: 0.1; }
      100% { opacity: 0.15; }
    }
    :root {
      --bg:#0f172a;
      --bg-dark:#020617;
      --panel:#111827;
      --panel2:#0b1220;
      --text:#e5e7eb;
      --muted:#94a3b8;
      --line:#1f2937;
      --accent:#22c55e;
      --accent-light:#86efac;
    }
    * { 
      box-sizing: border-box; 
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #020617 100%);
      color: var(--text);
      overflow: hidden;
    }
    .wrap {
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 20px;
      padding: 24px;
      height: 100vh;
    }
    .card {
      background: rgba(15, 23, 42, 0.4);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(51, 65, 85, 0.4);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.3),
        inset 1px 1px 0 rgba(255, 255, 255, 0.1);
      min-width: 0;
      display: flex;
      flex-direction: column;
      transition: all 0.3s ease;
    }
    .card:hover {
      border-color: rgba(51, 65, 85, 0.6);
      box-shadow: 
        0 25px 70px rgba(0, 0, 0, 0.4),
        inset 1px 1px 0 rgba(255, 255, 255, 0.15);
    } 
      margin: 0 0 16px 0; 
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    h1 { 
      font-size: 28px;
      background: linear-gradient(135deg, #e5e7eb, #bfdbfe);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    h2 { 
      font-size: 16px; 
      color: #bfdbfe;
    }
    .scene {
      width: 100%;
      aspect-ratio: 16/10;
      background: 
        linear-gradient(90deg, rgba(15, 23, 42, 0) 1px, transparent 1px),
        linear-gradient(rgba(15, 23, 42, 0) 1px, transparent 1px),
        linear-gradient(135deg, #1a2342 0%, #0f172a 50%, #0d1520 100%);
      background-size: 40px 40px, 40px 40px, 100% 100%;
      border-radius: 24px;
      border: 2px solid rgba(51, 65, 85, 0.5);
      position: relative;
      overflow: hidden;
      box-shadow: 
        inset 0 2px 16px rgba(0, 0, 0, 0.5),
        0 20px 60px rgba(0, 0, 0, 0.3);
      margin: 0 0 16px 0;
    }
    .scene::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 30% 40%, rgba(34, 197, 94, 0.05), transparent 50%);
      pointer-events: none;
      z-index: 1;
    }
    .legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 16px 0;
    }
    .pill {
      padding: 6px 12px;
      border-radius: 20px;
      background: rgba(17, 24, 39, 0.5);
      border: 1px solid rgba(51, 65, 85, 0.5);
      backdrop-filter: blur(8px);
      font-weight: 500;
      transition: all 0.2s ease;
    }
    .pill:hover {
      background: rgba(34, 197, 94, 0.1);
      border-color: rgba(34, 197, 94, 0.3);
    }
    .row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin: 16px 0;
      flex-wrap: wrap;
    }
    input[type=text] {
      flex: 1;
      min-width: 260px;
      background: rgba(11, 18, 32, 0.6);
      backdrop-filter: blur(8px);
      color: var(--text);
      border: 1px solid rgba(51, 65, 85, 0.5);
      border-radius: 12px;
      padding: 12px 16px;
      outline: none;
      font-size: 14px;
      transition: all 0.2s ease;
    }
    input[type=text]:focus {
      border-color: rgba(34, 197, 94, 0.5);
      background: rgba(11, 18, 32, 0.8);
      box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.1);
    }
    button {
      border: 0;
      border-radius: 12px;
      padding: 12px 20px;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    button:active {
      transform: translateY(0);
    }
    .primary { 
      background: linear-gradient(135deg, #22c55e, #16a34a);
      color: #052e16;
      box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
    }
    .primary:hover {
      box-shadow: 0 8px 25px rgba(34, 197, 94, 0.4);
    }
    .secondary { 
      background: rgba(30, 41, 59, 0.6);
      color: var(--text);
      border: 1px solid rgba(51, 65, 85, 0.5);
      backdrop-filter: blur(8px);
    }
    .secondary:hover {
      background: rgba(30, 41, 59, 0.9);
      border-color: rgba(51, 65, 85, 0.8);
    }
    .log, .obs, .meta {
      background: rgba(11, 18, 32, 0.5);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(51, 65, 85, 0.4);
      border-radius: 16px;
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
      font-size: 12px;
      line-height: 1.6;
      color: #d1d5db;
      transition: all 0.2s ease;
    }
    .log:hover, .obs:hover, .meta:hover {
      background: rgba(11, 18, 32, 0.7);
      border-color: rgba(51, 65, 85, 0.6);
    }
    .log, .obs { 
      height: 220px; 
    }
    .meta { 
      height: 120px; 
    }
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 0 0 16px 0;
      flex: 1;
    }
    .split > div {
      display: flex;
      flex-direction: column;
      gap: 8px;
      min-height: 0;
    }
    .muted { 
      color: var(--muted); 
      font-size: 13px;
      line-height: 1.5;
    }
    .status { 
      margin-top: 12px; 
      color: #86efac;
      font-weight: 500;
      font-size: 13px;
    }
    .label {
      display: inline-block;
      margin-bottom: 8px;
      font-size: 11px;
      color: #cbd5e1;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-weight: 600;
    }
    svg text {
      font-size: 13px;
      font-weight: 700;
      paint-order: stroke;
      stroke: rgba(15, 23, 42, 0.8);
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
      <defs>
        <radialGradient id="bowlGrad${x}${y}" cx="40%" cy="40%">
          <stop offset="0%" stop-color="#fff9e6"/>
          <stop offset="50%" stop-color="${color}"/>
          <stop offset="100%" stop-color="#a89968"/>
        </radialGradient>
        <linearGradient id="bowlShine${x}${y}" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#fff" stop-opacity="0.4"/>
          <stop offset="50%" stop-color="#fff" stop-opacity="0.1"/>
          <stop offset="100%" stop-color="#000" stop-opacity="0.1"/>
        </linearGradient>
      </defs>
      
      <!-- Bowl shadow on ground -->
      <ellipse cx="${x}" cy="${y + size * 0.65}" rx="${size * 1.1}" ry="${size * 0.3}" fill="#000" opacity="0.25"/>
      
      <!-- Bowl body (curved container) -->
      <path d="M ${x - size * 1.1} ${y - size * 0.1} Q ${x - size * 1.15} ${y + size * 0.3}, ${x - size * 0.8} ${y + size * 0.6}
               L ${x + size * 0.8} ${y + size * 0.6} Q ${x + size * 1.15} ${y + size * 0.3}, ${x + size * 1.1} ${y - size * 0.1} Z"
            fill="url(#bowlGrad${x}${y})" stroke="#8b7355" stroke-width="1.5" ${heldStroke}/>
      
      <!-- Bowl rim/edge (top opening) -->
      <ellipse cx="${x}" cy="${y - size * 0.1}" rx="${size * 1.1}" ry="${size * 0.25}"
               fill="none" stroke="#a89968" stroke-width="2.5" opacity="0.7"/>
      
      <!-- Bowl interior base (inner bottom) -->
      <ellipse cx="${x}" cy="${y + size * 0.5}" rx="${size * 0.7}" ry="${size * 0.25}"
               fill="#e8dcc8" opacity="0.6" stroke="none"/>
      
      <!-- Bowl interior highlight (inner shine) -->
      <ellipse cx="${x - size * 0.5}" cy="${y + size * 0.2}" rx="${size * 0.6}" ry="${size * 0.35}"
               fill="url(#bowlShine${x}${y})" stroke="none"/>
      
      <!-- Bowl side highlight (outer shine) -->
      <ellipse cx="${x - size * 0.6}" cy="${y + size * 0.1}" rx="${size * 0.3}" ry="${size * 0.4}"
               fill="#fff" opacity="0.3" stroke="none"/>
      
      <!-- Bowl texture line (horizontal band) -->
      <ellipse cx="${x}" cy="${y + size * 0.15}" rx="${size * 0.95}" ry="${size * 0.15}"
               fill="none" stroke="#8b7355" stroke-width="0.5" opacity="0.4"/>
      
      <!-- Optional: content inside bowl (semi-transparent) -->
      <ellipse cx="${x}" cy="${y + size * 0.35}" rx="${size * 0.65}" ry="${size * 0.22}"
               fill="#f5deb3" opacity="0.3" stroke="none"/>
      
      <text x="${labelX}" y="${labelY}" fill="#8b7355" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "plate" || lower.includes("plate")) {
    return `
      <ellipse cx="${x}" cy="${y + size * 0.25}" rx="${size * 1.2}" ry="${size * 0.52}" fill="#000" opacity="0.2"/>
      <ellipse cx="${x}" cy="${y}" rx="${size * 1.15}" ry="${size * 0.5}"
               fill="${color}" opacity="0.95" stroke="#e2e8f0" ${heldStroke}/>
      <ellipse cx="${x}" cy="${y}" rx="${size * 0.55}" ry="${size * 0.20}"
               fill="none" stroke="#f8fafc" stroke-opacity="0.7" stroke-width="2"/>
      <ellipse cx="${x - size * 0.5}" cy="${y - size * 0.15}" rx="${size * 0.35}" ry="${size * 0.15}"
               fill="#fff" opacity="0.2" stroke="none"/>
      <text x="${labelX}" y="${labelY}" fill="#e2e8f0" font-weight="500">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "cup" || lower.includes("cup")) {
    return `
      <defs>
        <linearGradient id="cupGrad${x}${y}" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#e0e7ff"/>
          <stop offset="50%" stop-color="${color}"/>
          <stop offset="100%" stop-color="#a5b4fc"/>
        </linearGradient>
      </defs>
      
      <!-- Cup shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.65}" rx="${size * 0.8}" ry="${size * 0.25}" fill="#000" opacity="0.25"/>
      
      <!-- Cup body (cylinder) -->
      <path d="M ${x - size * 0.75} ${y - size * 0.4} L ${x - size * 0.7} ${y + size * 0.5} Q ${x} ${y + size * 0.65}, ${x + size * 0.7} ${y + size * 0.5} L ${x + size * 0.75} ${y - size * 0.4} Z"
            fill="url(#cupGrad${x}${y})" stroke="#6366f1" ${heldStroke}/>
      
      <!-- Cup inner shadow (depth) -->
      <ellipse cx="${x - size * 0.4}" cy="${y - size * 0.2}" rx="${size * 0.25}" ry="${size * 0.15}" 
               fill="#000" opacity="0.1" stroke="none"/>
      
      <!-- Cup rim (top open edge) -->
      <ellipse cx="${x}" cy="${y - size * 0.4}" rx="${size * 0.75}" ry="${size * 0.2}"
               fill="none" stroke="#818cf8" stroke-width="2" opacity="0.8"/>
      
      <!-- Cup liquid (if not empty, semi-transparent) -->
      <ellipse cx="${x}" cy="${y + size * 0.2}" rx="${size * 0.65}" ry="${size * 0.25}"
               fill="#fbbf24" opacity="0.4" stroke="none"/>
      
      <!-- Cup shine/highlight -->
      <ellipse cx="${x - size * 0.5}" cy="${y - size * 0.1}" rx="${size * 0.25}" ry="${size * 0.35}"
               fill="#fff" opacity="0.25" stroke="none"/>
      
      <!-- Cup handle -->
      <path d="M ${x + size * 0.75} ${y - size * 0.1} Q ${x + size * 1.3} ${y + size * 0.2}, ${x + size * 0.75} ${y + size * 0.4}"
            fill="none" stroke="#6366f1" stroke-width="${size * 0.15}" stroke-linecap="round" opacity="0.9"/>
      
      <!-- Handle highlight -->
      <path d="M ${x + size * 0.78} ${y - size * 0.08} Q ${x + size * 1.25} ${y + size * 0.22}, ${x + size * 0.78} ${y + size * 0.38}"
            fill="none" stroke="#a5b4fc" stroke-width="${size * 0.07}" stroke-linecap="round" opacity="0.5"/>
      
      <text x="${labelX}" y="${labelY}" fill="#6366f1" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "apple" || lower.includes("apple")) {
    return `
      <defs>
        <radialGradient id="appleGrad${x}${y}">
          <stop offset="0%" stop-color="#f87171"/>
          <stop offset="70%" stop-color="#dc2626"/>
          <stop offset="100%" stop-color="#991b1b"/>
        </radialGradient>
      </defs>
      <!-- Apple shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.55}" rx="${size * 0.85}" ry="${size * 0.35}" fill="#000" opacity="0.25"/>
      
      <!-- Main apple body - bumpy sphere -->
      <circle cx="${x}" cy="${y}" r="${size}" fill="url(#appleGrad${x}${y})" stroke="#7f1d1d" ${heldStroke}/>
      
      <!-- Apple top indent -->
      <ellipse cx="${x}" cy="${y - size * 0.8}" rx="${size * 0.4}" ry="${size * 0.25}" fill="#000" opacity="0.2"/>
      
      <!-- Apple stem -->
      <rect x="${x - size * 0.08}" y="${y - size * 1.1}" width="${size * 0.16}" height="${size * 0.35}" rx="${size * 0.08}" 
            fill="#78350f" stroke="#451a03" stroke-width="1"/>
      <ellipse cx="${x}" cy="${y - size * 1.1}" rx="${size * 0.1}" ry="${size * 0.08}" fill="#92400e"/>
      
      <!-- Apple leaf -->
      <ellipse cx="${x + size * 0.35}" cy="${y - size * 0.95}" rx="${size * 0.25}" ry="${size * 0.15}" 
               fill="#16a34a" stroke="#15803d" stroke-width="1" transform="rotate(-30 ${x + size * 0.35} ${y - size * 0.95})"/>
      <path d="M ${x + size * 0.15} ${y - size * 0.95} Q ${x + size * 0.4} ${y - size * 0.85}, ${x + size * 0.55} ${y - size * 0.92}" 
            stroke="#15803d" stroke-width="1" fill="none" opacity="0.6"/>
      
      <!-- Apple shine/highlight -->
      <ellipse cx="${x - size * 0.3}" cy="${y - size * 0.3}" rx="${size * 0.35}" ry="${size * 0.45}" 
               fill="#fff" opacity="0.15" stroke="none"/>
      
      <!-- Apple cheek blush (optional highlight for rosy look) -->
      <circle cx="${x + size * 0.4}" cy="${y + size * 0.1}" r="${size * 0.2}" fill="#fca5a5" opacity="0.3"/>
      
      <text x="${labelX}" y="${labelY}" fill="#fecaca" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "banana" || lower.includes("banana")) {
    return `
      <!-- Banana shadow -->
      <ellipse cx="${x + size * 0.1}" cy="${y + size * 0.4}" rx="${size * 0.6}" ry="${size * 0.15}" fill="#000" opacity="0.2"/>
      
      <!-- Banana body with curved shape -->
      <ellipse cx="${x}" cy="${y - size * 0.15}" rx="${size * 0.35}" ry="${size * 0.8}" 
               fill="#fcd34d" stroke="#d97706" stroke-width="1.5" ${heldStroke}
               transform="rotate(-20 ${x} ${y})"/>
      
      <!-- Banana highlight overlay -->
      <ellipse cx="${x - size * 0.15}" cy="${y - size * 0.25}" rx="${size * 0.25}" ry="${size * 0.65}" 
               fill="#fef9e7" opacity="0.4" stroke="none"
               transform="rotate(-20 ${x} ${y})"/>
      
      <!-- Banana tip curve -->
      <path d="M ${x + size * 0.25} ${y - size * 0.6} Q ${x + size * 0.4} ${y - size * 0.5}, ${x + size * 0.35} ${y - size * 0.3}"
            stroke="#f59e0b" stroke-width="${size * 0.04}" fill="none" stroke-linecap="round" opacity="0.6"/>
      
      <!-- Banana stem crown -->
      <rect x="${x - size * 0.1}" y="${y - size * 1.0}" width="${size * 0.2}" height="${size * 0.4}" 
            rx="${size * 0.08}" fill="#92400e"/>
      <circle cx="${x}" cy="${y - size * 1.0}" r="${size * 0.12}" fill="#b45309"/>
      
      <text x="${labelX}" y="${labelY}" fill="#d97706" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "book" || lower.includes("book")) {
    return `
      <!-- Book shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.55}" rx="${size * 0.9}" ry="${size * 0.25}" fill="#000" opacity="0.22"/>
      
      <!-- Book spine side edge -->
      <rect x="${x + size * 0.65}" y="${y - size * 0.6}" width="${size * 0.15}" height="${size * 1.2}" rx="2"
            fill="#8b6f47" stroke="#6b5744" stroke-width="1"/>
      
      <!-- Main book cover -->
      <rect x="${x - size * 0.65}" y="${y - size * 0.6}" width="${size * 1.3}" height="${size * 1.2}" rx="6"
            fill="${color}" stroke="#6b5744" stroke-width="1.5" ${heldStroke}/>
      
      <!-- Book cover pattern decoration -->
      <line x1="${x - size * 0.5}" y1="${y - size * 0.3}" x2="${x + size * 0.5}" y2="${y - size * 0.3}"
            stroke="#e5c5a0" stroke-width="1" opacity="0.5"/>
      <line x1="${x - size * 0.5}" y1="${y}" x2="${x + size * 0.5}" y2="${y}"
            stroke="#e5c5a0" stroke-width="1" opacity="0.5"/>
      <line x1="${x - size * 0.5}" y1="${y + size * 0.3}" x2="${x + size * 0.5}" y2="${y + size * 0.3}"
            stroke="#e5c5a0" stroke-width="1" opacity="0.5"/>
      
      <!-- Book title area -->
      <rect x="${x - size * 0.55}" y="${y - size * 0.2}" width="${size * 1.1}" height="${size * 0.5}" rx="3"
            fill="#000" opacity="0.15" stroke="none"/>
      
      <!-- Book shine highlight -->
      <rect x="${x - size * 0.63}" y="${y - size * 0.58}" width="${size * 0.3}" height="${size * 1.16}" rx="4"
            fill="#fff" opacity="0.12" stroke="none"/>
      
      <!-- Pages visible at edge -->
      <line x1="${x + size * 0.64}" y1="${y - size * 0.55}" x2="${x + size * 0.64}" y2="${y + size * 0.55}"
            stroke="#f5e6d3" stroke-width="2" opacity="0.6"/>
      
      <text x="${labelX}" y="${labelY}" fill="#a0826d" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "ball" || lower.includes("ball")) {
    return `
      <!-- Ball shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.65}" rx="${size * 1.0}" ry="${size * 0.3}" fill="#000" opacity="0.25"/>
      
      <!-- Main ball body -->
      <circle cx="${x}" cy="${y}" r="${size}" fill="${color}" stroke="#1f2937" stroke-width="1.5" ${heldStroke}/>
      
      <!-- Ball glossy highlight -->
      <circle cx="${x - size * 0.35}" cy="${y - size * 0.35}" r="${size * 0.35}" 
              fill="#fff" opacity="0.25" stroke="none"/>
      
      <!-- Secondary shine for depth -->
      <ellipse cx="${x + size * 0.25}" cy="${y - size * 0.2}" rx="${size * 0.2}" ry="${size * 0.25}"
               fill="#fff" opacity="0.1" stroke="none"/>
      
      <!-- Ball seam pattern -->
      <path d="M ${x - size * 0.1} ${y - size * 0.8} Q ${x} ${y - size * 0.5}, ${x + size * 0.1} ${y - size * 0.8}"
            stroke="#1f2937" stroke-width="1" fill="none" opacity="0.4"/>
      
      <text x="${labelX}" y="${labelY}" fill="#fff" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "orange" || lower.includes("orange")) {
    return `
      <!-- Orange shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.6}" rx="${size * 1.0}" ry="${size * 0.32}" fill="#000" opacity="0.25"/>
      
      <!-- Main orange body -->
      <circle cx="${x}" cy="${y}" r="${size}" fill="#f97316" stroke="#b45309" stroke-width="1.5" ${heldStroke}/>
      
      <!-- Orange peel texture with radial lines -->
      <g opacity="0.3">
        <line x1="${x}" y1="${y - size}" x2="${x}" y2="${y + size}" stroke="#ea580c" stroke-width="0.8"/>
        <line x1="${x - size * 0.866}" y1="${y - size * 0.5}" x2="${x + size * 0.866}" y2="${y + size * 0.5}" stroke="#ea580c" stroke-width="0.8"/>
        <line x1="${x - size * 0.866}" y1="${y + size * 0.5}" x2="${x + size * 0.866}" y2="${y - size * 0.5}" stroke="#ea580c" stroke-width="0.8"/>
      </g>
      
      <!-- Orange shine -->
      <circle cx="${x - size * 0.3}" cy="${y - size * 0.3}" r="${size * 0.3}" 
              fill="#fff" opacity="0.22" stroke="none"/>
      
      <!-- Orange leaf detail -->
      <ellipse cx="${x - size * 0.3}" cy="${y - size * 1.1}" rx="${size * 0.15}" ry="${size * 0.25}"
               fill="#16a34a" stroke="#15803d" stroke-width="1" opacity="0.8"/>
      <path d="M ${x - size * 0.25} ${y - size * 1.0} Q ${x - size * 0.35} ${y - size * 1.15}, ${x - size * 0.3} ${y - size * 1.2}"
            stroke="#15803d" stroke-width="0.8" fill="none" opacity="0.6"/>
      
      <!-- Stem connecting leaf -->
      <line x1="${x - size * 0.3}" y1="${y - size * 1.05}" x2="${x - size * 0.3}" y2="${y - size * 1.25}"
            stroke="#92400e" stroke-width="1"/>
      
      <text x="${labelX}" y="${labelY}" fill="#ea580c" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "toy" || lower.includes("toy")) {
    return `
      <!-- Toy shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.6}" rx="${size * 0.95}" ry="${size * 0.3}" fill="#000" opacity="0.25"/>
      
      <!-- Toy body cube-like shape -->
      <g>
        <!-- Front face -->
        <rect x="${x - size * 0.7}" y="${y - size * 0.5}" width="${size * 1.4}" height="${size * 1.0}" rx="4"
              fill="${color}" stroke="#374151" stroke-width="1.5" ${heldStroke}/>
        
        <!-- Right face for 3D effect -->
        <polygon points="${x + size * 0.7},${y - size * 0.5} ${x + size * 0.85},${y - size * 0.35} ${x + size * 0.85},${y + size * 0.65} ${x + size * 0.7},${y + size * 0.5}"
                 fill="#000" opacity="0.15" stroke="none"/>
        
        <!-- Top face for 3D effect -->
        <polygon points="${x - size * 0.7},${y - size * 0.5} ${x + size * 0.7},${y - size * 0.5} ${x + size * 0.85},${y - size * 0.35} ${x - size * 0.85},${y - size * 0.35}"
                 fill="#fff" opacity="0.15" stroke="none"/>
      </g>
      
      <!-- Toy colorful button pattern -->
      <circle cx="${x - size * 0.3}" cy="${y - size * 0.15}" r="${size * 0.1}" fill="#ef4444" opacity="0.8"/>
      <circle cx="${x}" cy="${y - size * 0.15}" r="${size * 0.1}" fill="#f59e0b" opacity="0.8"/>
      <circle cx="${x + size * 0.3}" cy="${y - size * 0.15}" r="${size * 0.1}" fill="#3b82f6" opacity="0.8"/>
      
      <circle cx="${x - size * 0.3}" cy="${y + size * 0.15}" r="${size * 0.1}" fill="#10b981" opacity="0.8"/>
      <circle cx="${x}" cy="${y + size * 0.15}" r="${size * 0.1}" fill="#8b5cf6" opacity="0.8"/>
      <circle cx="${x + size * 0.3}" cy="${y + size * 0.15}" r="${size * 0.1}" fill="#ec4899" opacity="0.8"/>
      
      <!-- Toy shine -->
      <rect x="${x - size * 0.65}" y="${y - size * 0.45}" width="${size * 0.4}" height="${size * 0.3}" rx="3"
            fill="#fff" opacity="0.15" stroke="none"/>
      
      <text x="${labelX}" y="${labelY}" fill="#fff" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  if (lower === "pear" || lower.includes("pear")) {
    return `
      <!-- Pear shadow -->
      <ellipse cx="${x}" cy="${y + size * 0.6}" rx="${size * 0.9}" ry="${size * 0.3}" fill="#000" opacity="0.25"/>
      
      <!-- Pear body with narrower top and wider bottom -->
      <path d="M ${x} ${y - size * 0.8} C ${x - size * 0.4} ${y - size * 0.6}, ${x - size * 0.6} ${y}, ${x - size * 0.5} ${y + size * 0.5} C ${x - size * 0.4} ${y + size * 0.7}, ${x + size * 0.4} ${y + size * 0.7}, ${x + size * 0.5} ${y + size * 0.5} C ${x + size * 0.6} ${y}, ${x + size * 0.4} ${y - size * 0.6}, ${x} ${y - size * 0.8}"
            fill="#c7d636" stroke="#a3a832" stroke-width="1.5" ${heldStroke}/>
      
      <!-- Pear gradient shading -->
      <ellipse cx="${x - size * 0.2}" cy="${y - size * 0.3}" rx="${size * 0.3}" ry="${size * 0.5}"
               fill="#dcff6e" opacity="0.3" stroke="none"/>
      
      <!-- Pear top indent -->
      <circle cx="${x}" cy="${y - size * 0.78}" r="${size * 0.08}" fill="#a3a832" opacity="0.6"/>
      
      <!-- Pear stem -->
      <rect x="${x - size * 0.06}" y="${y - size * 1.0}" width="${size * 0.12}" height="${size * 0.22}" rx="2"
            fill="#8b7355" stroke="#6b5a45" stroke-width="0.8"/>
      
      <!-- Pear leaf -->
      <ellipse cx="${x + size * 0.2}" cy="${y - size * 0.8}" rx="${size * 0.18}" ry="${size * 0.12}"
               fill="#16a34a" stroke="#15803d" stroke-width="1" opacity="0.85"
               transform="rotate(-25 ${x + size * 0.2} ${y - size * 0.8})"/>
      
      <!-- Leaf vein detail -->
      <line x1="${x + size * 0.08}" y1="${y - size * 0.85}" x2="${x + size * 0.32}" y2="${y - size * 0.75}"
            stroke="#15803d" stroke-width="0.6" opacity="0.6"/>
      
      <!-- Pear rosy cheek blush -->
      <ellipse cx="${x + size * 0.35}" cy="${y + size * 0.15}" rx="${size * 0.15}" ry="${size * 0.12}"
               fill="#fb7185" opacity="0.25" stroke="none"/>
      
      <!-- Pear shine highlight -->
      <ellipse cx="${x - size * 0.25}" cy="${y - size * 0.2}" rx="${size * 0.18}" ry="${size * 0.35}"
               fill="#fff" opacity="0.18" stroke="none"/>
      
      <text x="${labelX}" y="${labelY}" fill="#a3a832" font-weight="600">${esc(name)}${held ? " (held)" : ""}</text>
    `;
  }

  // Default rendering for other objects - improved version
  return `
    <!-- Object shadow -->
    <ellipse cx="${x}" cy="${y + size * 0.55}" rx="${size * 0.85}" ry="${size * 0.35}" fill="#000" opacity="0.25"/>
    
    <!-- Main object -->
    <circle cx="${x}" cy="${y}" r="${size}" fill="${color}" stroke="#fecaca" ${heldStroke}/>
    
    <!-- Object highlight -->
    <circle cx="${x - size * 0.3}" cy="${y - size * 0.3}" r="${size * 0.25}" fill="#fff" opacity="0.2" stroke="none"/>
    
    <!-- Additional shine for better 3D effect -->
    <ellipse cx="${x - size * 0.2}" cy="${y - size * 0.4}" rx="${size * 0.2}" ry="${size * 0.15}" 
             fill="#fff" opacity="0.15" stroke="none"/>
    
    <text x="${labelX}" y="${labelY}" fill="#fecaca" font-weight="500">${esc(name)}${held ? " (held)" : ""}</text>
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

      return `
      <g
        class="obj-item draggable-object"
        data-name="${name}"
        style="cursor: grab;"
      >
        ${drawObject(name, enriched)}
      </g>
      `;
        
    })
    .join("\n");

  const bowlZone = "";

  const svg = `
  <svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 20px 40px rgba(0,0,0,0.2))">
    <defs>
      <linearGradient id="tableGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#334155"/>
        <stop offset="100%" stop-color="#1f2937"/>
      </linearGradient>
      <filter id="shadow" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000" flood-opacity="0.35"/>
      </filter>
      <filter id="glow">
        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <style>
        @keyframes pulse-object { 0%, 100% { filter: drop-shadow(0 6px 12px rgba(0,0,0,0.35)); } 50% { filter: drop-shadow(0 8px 16px rgba(0,0,0,0.45)); } }
        .obj-item { animation: pulse-object 3s ease-in-out infinite; }
      </style>
    </defs>

    <rect x="0" y="0" width="${w}" height="${h}" fill="url(#tableGrad)"/>
    <rect x="20" y="20" width="${w-40}" height="${h-40}" rx="24" fill="#0f172a" opacity="0.28" stroke="#475569" stroke-width="2"/>

    ${bowlZone}

    <g filter="url(#shadow)">
      <!-- Robot body/torso with gradient -->
      <defs>
        <linearGradient id="robotGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:#34d399;stop-opacity:1" />
          <stop offset="100%" style="stop-color:#10b981;stop-opacity:1" />
        </linearGradient>
      </defs>
      
      <!-- Robot body/torso -->
      <rect x="${robotX - 14}" y="${robotY - 8}" width="28" height="36" rx="6" fill="url(#robotGradient)" stroke="#16a34a" stroke-width="2"/>
      
      <!-- Body shine effect -->
      <rect x="${robotX - 12}" y="${robotY - 6}" width="10" height="28" rx="4" fill="#6ee7b7" opacity="0.3"/>
      
      <!-- Robot head -->
      <rect x="${robotX - 18}" y="${robotY - 50}" width="36" height="38" rx="8" fill="#10b981" stroke="#059669" stroke-width="2"/>
      
      <!-- Head shine -->
      <ellipse cx="${robotX - 8}" cy="${robotY - 42}" rx="10" ry="8" fill="#34d399" opacity="0.4"/>
      
      <!-- Head panel highlight -->
      <rect x="${robotX - 16}" y="${robotY - 48}" width="32" height="8" rx="4" fill="#34d399" opacity="0.6"/>
      
      <!-- Left eye -->
      <circle cx="${robotX - 7}" cy="${robotY - 35}" r="4" fill="#fbbf24"/>
      <circle cx="${robotX - 7}" cy="${robotY - 35}" r="2.5" fill="#1f2937"/>
      <circle cx="${robotX - 5.5}" cy="${robotY - 36}" r="1.5" fill="#fef3c7" opacity="0.8"/>
      
      <!-- Right eye -->
      <circle cx="${robotX + 7}" cy="${robotY - 35}" r="4" fill="#fbbf24"/>
      <circle cx="${robotX + 7}" cy="${robotY - 35}" r="2.5" fill="#1f2937"/>
      <circle cx="${robotX + 8.5}" cy="${robotY - 36}" r="1.5" fill="#fef3c7" opacity="0.8"/>
      
      <!-- Antenna left with glow -->
      <line x1="${robotX - 12}" y1="${robotY - 50}" x2="${robotX - 20}" y2="${robotY - 68}" stroke="#34d399" stroke-width="3" stroke-linecap="round" opacity="0.7"/>
      <line x1="${robotX - 12}" y1="${robotY - 50}" x2="${robotX - 20}" y2="${robotY - 68}" stroke="#6ee7b7" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="${robotX - 20}" cy="${robotY - 68}" r="3" fill="#34d399"/>
      <circle cx="${robotX - 20}" cy="${robotY - 68}" r="1.5" fill="#6ee7b7" opacity="0.6"/>
      
      <!-- Antenna right with glow -->
      <line x1="${robotX + 12}" y1="${robotY - 50}" x2="${robotX + 20}" y2="${robotY - 68}" stroke="#34d399" stroke-width="3" stroke-linecap="round" opacity="0.7"/>
      <line x1="${robotX + 12}" y1="${robotY - 50}" x2="${robotX + 20}" y2="${robotY - 68}" stroke="#6ee7b7" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="${robotX + 20}" cy="${robotY - 68}" r="3" fill="#34d399"/>
      <circle cx="${robotX + 20}" cy="${robotY - 68}" r="1.5" fill="#6ee7b7" opacity="0.6"/>
      
      <!-- Left arm -->
      <g>
        <line x1="${robotX - 14}" y1="${robotY + 2}" x2="${robotX - 38}" y2="${robotY + 8}" stroke="#16a34a" stroke-width="6" stroke-linecap="round"/>
        <line x1="${robotX - 14}" y1="${robotY + 2}" x2="${robotX - 38}" y2="${robotY + 8}" stroke="#86efac" stroke-width="3" stroke-linecap="round" opacity="0.6"/>
        <circle cx="${robotX - 38}" cy="${robotY + 8}" r="5" fill="#10b981" stroke="#34d399" stroke-width="2"/>
        <circle cx="${robotX - 37}" cy="${robotY + 7}" r="2" fill="#6ee7b7" opacity="0.5"/>
      </g>
      
      <!-- Right arm -->
      <g>
        <line x1="${robotX + 14}" y1="${robotY + 2}" x2="${robotX + 38}" y2="${robotY + 8}" stroke="#16a34a" stroke-width="6" stroke-linecap="round"/>
        <line x1="${robotX + 14}" y1="${robotY + 2}" x2="${robotX + 38}" y2="${robotY + 8}" stroke="#86efac" stroke-width="3" stroke-linecap="round" opacity="0.6"/>
        <circle cx="${robotX + 38}" cy="${robotY + 8}" r="5" fill="#10b981" stroke="#34d399" stroke-width="2"/>
        <circle cx="${robotX + 37}" cy="${robotY + 7}" r="2" fill="#6ee7b7" opacity="0.5"/>
      </g>
      
      <!-- Left leg -->
      <g>
        <line x1="${robotX - 7}" y1="${robotY + 28}" x2="${robotX - 10}" y2="${robotY + 50}" stroke="#16a34a" stroke-width="5" stroke-linecap="round"/>
        <line x1="${robotX - 7}" y1="${robotY + 28}" x2="${robotX - 10}" y2="${robotY + 50}" stroke="#86efac" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
        <rect x="${robotX - 14}" y="${robotY + 50}" width="8" height="12" rx="3" fill="#10b981" stroke="#34d399" stroke-width="1.5"/>
        <rect x="${robotX - 13}" y="${robotY + 51}" width="6" height="4" rx="2" fill="#6ee7b7" opacity="0.4"/>
      </g>
      
      <!-- Right leg -->
      <g>
        <line x1="${robotX + 7}" y1="${robotY + 28}" x2="${robotX + 10}" y2="${robotY + 50}" stroke="#16a34a" stroke-width="5" stroke-linecap="round"/>
        <line x1="${robotX + 7}" y1="${robotY + 28}" x2="${robotX + 10}" y2="${robotY + 50}" stroke="#86efac" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>
        <rect x="${robotX + 6}" y="${robotY + 50}" width="8" height="12" rx="3" fill="#10b981" stroke="#34d399" stroke-width="1.5"/>
        <rect x="${robotX + 7}" y="${robotY + 51}" width="6" height="4" rx="2" fill="#6ee7b7" opacity="0.4"/>
      </g>
      
      <!-- Chest panel -->
      <rect x="${robotX - 10}" y="${robotY + 4}" width="20" height="16" rx="4" fill="#34d399" opacity="0.4" stroke="#10b981" stroke-width="1.5"/>
      <circle cx="${robotX - 6}" cy="${robotY + 9}" r="2" fill="#6ee7b7" opacity="0.7"/>
      <circle cx="${robotX + 2}" cy="${robotY + 9}" r="2" fill="#6ee7b7" opacity="0.7"/>
      <circle cx="${robotX + 6}" cy="${robotY + 9}" r="2" fill="#6ee7b7" opacity="0.7"/>
      
      <!-- Label -->
      <text x="${robotX - 12}" y="${robotY + 72}" fill="#dcfce7" font-size="12" font-weight="600">robot</text>
    </g>

    ${objectLayers}
  </svg>`;

  document.getElementById("scene").innerHTML = svg;
  attachDragHandlers();
}

function attachDragHandlers() {

  const svg = document.querySelector(
    "#scene svg"
  );

  if (!svg) return;

  let dragging = null;

  function getSVGPoint(evt) {

    const pt = svg.createSVGPoint();

    pt.x = evt.clientX;

    pt.y = evt.clientY;

    return pt.matrixTransform(
      svg.getScreenCTM().inverse()
    );
  }

  svg.querySelectorAll(
    ".draggable-object"
  ).forEach(el => {

    el.addEventListener(
      "pointerdown",
      evt => {

        dragging = {
          el,
          name: el.dataset.name
        };

        el.style.cursor =
          "grabbing";
      }
    );
  });

  svg.addEventListener(
    "pointermove",
    evt => {

      if (!dragging) return;

      const p = getSVGPoint(evt);

      dragging.el.setAttribute(
        "transform",
        `translate(${p.x-50},${p.y-50})`
      );
    }
  );

  svg.addEventListener(
    "pointerup",
    async evt => {

      if (!dragging) return;

      const p = getSVGPoint(evt);

      const wx =
        p.x / (
          state.renderData.width
        );

      const wy =
        p.y / (
          state.renderData.height
        );

      const data = await post(
        "/api/move_object",
        {
          name: dragging.name,
          x: wx,
          y: wy
        }
      );

      render(
        data.observation,
        data.render,
        data.log,
        {
          last_reasoning:
            data.last_reasoning || "",

          last_subgoal:
            data.last_subgoal || "",

          last_action:
            data.last_action || null,

          last_result:
            data.last_result || null,

          plan_queue:
            data.plan_queue || []
        }
      );

      dragging.el.style.cursor =
        "grab";

      dragging = null;
    }
  );
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