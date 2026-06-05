from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from google import genai


SYSTEM_PROMPT = """
You are a tabletop robot planner.

Your job is to decide the NEXT action only.

You will receive:
- task
- task_grounding
- affordances
- progress
- robot
- objects

Available actions:
1. MOVE_TO(target)
2. GRASP(target)
3. RELEASE()
4. OPEN_GRIPPER()
5. CLOSE_GRIPPER()

Rules:
- Return JSON ONLY.
- Choose exactly ONE action.
- If task_grounding.mode == "dynamic", re-evaluate the target on every step.
- For dynamic tasks, prefer the currently newest matching object using spawn_step.
- Prefer task_grounding.goal_object and task_grounding.goal_container when present.
- If a task refers to a color/category and no object matches it, do not guess a different category.
- Never substitute a container for a fruit or other mismatched object.
- Use affordances to avoid impossible actions.
- Use progress to continue from the current phase.
- Do not explain outside JSON.

Example:
{
  "reasoning": "The robot is far from the apple.",
  "subgoal": "reach_apple",
  "action": {
    "type": "MOVE_TO",
    "target": "apple"
  }
}
"""


def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None

    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class Planner:
    def __init__(self):
        self.use_api = bool(os.getenv("GEMINI_API_KEY"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if self.use_api:
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

    def plan(self, task: str, observation: Dict, history: List[str]) -> Dict:
        if self.use_api:
            try:
                return self._plan_with_api(task, observation, history)
            except Exception as e:
                print(f"[Planner] Gemini failed: {e}")
                print("[Planner] Falling back to heuristic planner.")

        return self._heuristic_plan(task, observation)

    def _heuristic_plan(self, task: str, observation: Dict) -> Dict:
        task_l = task.lower()

        robot = observation.get("robot", {})
        objs = observation.get("objects", {})
        grounding = observation.get("task_grounding", {})
        affordances = observation.get("affordances", {})
        progress = observation.get("progress", {})

        grounding = observation.get("task_grounding", {})
        affordances = observation.get("affordances", {})
        progress = observation.get("progress", {})

        mode = grounding.get("mode", "static")

        def first_existing(names):
            for name in names or []:
                if name in objs:
                    return name
            return None

        def dist_to(name: str) -> float:
            item = objs.get(name)
            if not item:
                return 999.0
            rpos = robot.get("pos", [0, 0])
            opos = item.get("pos", [0, 0])
            return abs(rpos[0] - opos[0]) + abs(rpos[1] - opos[1])

        # 1) Prefer grounded targets from observation
        target_object = grounding.get("goal_object")
        target_container = grounding.get("goal_container")

        if mode == "dynamic":
            # For dynamic tasks, trust the latest observation grounding every step.
            if target_object is None:
                newest_candidates = [
                    name for name, obj in objs.items()
                    if obj.get("category") != "container"
                ]
                if newest_candidates:
                    target_object = max(
                        newest_candidates,
                        key=lambda n: objs[n].get("spawn_step", -1)
                    )

        if not target_object:
            target_object = first_existing(grounding.get("goal_object_candidates", []))

        if not target_container:
            target_container = first_existing(grounding.get("goal_container_candidates", []))

        # 2) Lightweight semantic fallback for object
        if not target_object:
            if "red fruit" in task_l:
                for name, obj in objs.items():
                    if obj.get("category") == "fruit" and obj.get("color") == "red":
                        target_object = name
                        break
            elif "yellow fruit" in task_l:
                for name, obj in objs.items():
                    if obj.get("category") == "fruit" and obj.get("color") == "yellow":
                        target_object = name
                        break
            elif "fruit" in task_l:
                for name, obj in objs.items():
                    if obj.get("category") == "fruit":
                        target_object = name
                        break

        # 3) Lightweight semantic fallback for container
        if not target_container:
            for name, obj in objs.items():
                if obj.get("category") == "container" and name.lower() in task_l:
                    target_container = name
                    break

        if not target_container:
            if "bowl" in task_l:
                target_container = "bowl" if "bowl" in objs else None
            elif "plate" in task_l:
                target_container = "plate" if "plate" in objs else None
            elif "cup" in task_l:
                target_container = "cup" if "cup" in objs else None

        # If still nothing, do not guess
        if not target_object:
            return {
                "reasoning": "No matching object found in observation.",
                "subgoal": "clarify",
                "action": {
                    "type": "MOVE_TO",
                    "target": "home",
                },
            }

        holding = robot.get("holding")
        phase = progress.get("phase", "")

        # 4) Pick phase
        if holding != target_object:
            obj = objs.get(target_object)
            if not obj:
                return {
                    "reasoning": f"Target object '{target_object}' is missing from observation.",
                    "subgoal": "clarify",
                    "action": {
                        "type": "MOVE_TO",
                        "target": "home",
                    },
                }

            obj_aff = affordances.get(target_object, {})
            if not obj_aff.get("graspable", True):
                return {
                    "reasoning": f"{target_object} is not graspable.",
                    "subgoal": "clarify_object",
                    "action": {
                        "type": "MOVE_TO",
                        "target": "home",
                    },
                }

            if phase in {"seek_object", "idle", "carrying"} and dist_to(target_object) > 0.06:
                return {
                    "reasoning": f"Move to {target_object}.",
                    "subgoal": f"reach_{target_object}",
                    "action": {
                        "type": "MOVE_TO",
                        "target": target_object,
                    },
                }

            return {
                "reasoning": f"Pick {target_object}.",
                "subgoal": f"pick_{target_object}",
                "action": {
                    "type": "GRASP",
                    "target": target_object,
                },
            }

        # 5) Place phase
        if not target_container:
            return {
                "reasoning": f"Holding {target_object}, but no target container was identified.",
                "subgoal": "clarify_container",
                "action": {
                    "type": "MOVE_TO",
                    "target": "home",
                },
            }

        container = objs.get(target_container)
        if not container:
            return {
                "reasoning": f"Target container '{target_container}' is missing from observation.",
                "subgoal": "clarify_container",
                "action": {
                    "type": "MOVE_TO",
                    "target": "home",
                },
            }

        container_aff = affordances.get(target_container, {})
        if container_aff and not container_aff.get("placeable", True):
            return {
                "reasoning": f"{target_container} is not a valid place target.",
                "subgoal": "clarify_container",
                "action": {
                    "type": "MOVE_TO",
                    "target": "home",
                },
            }

        if phase in {"seek_container", "carrying"} and dist_to(target_container) > 0.06:
            return {
                "reasoning": f"Move to {target_container}.",
                "subgoal": f"reach_{target_container}",
                "action": {
                    "type": "MOVE_TO",
                    "target": target_container,
                },
            }

        return {
            "reasoning": "Release object.",
            "subgoal": "release",
            "action": {
                "type": "RELEASE",
            },
        }

    def _plan_with_api(self, task: str, observation: Dict, history: List[str]) -> Dict:
        prompt = f"""
{SYSTEM_PROMPT}

TASK:
{task}

OBSERVATION:
{json.dumps(observation, indent=2)}

ACTION HISTORY:
{json.dumps(history, indent=2)}

Return JSON ONLY.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        content = (response.text or "").strip()
        print("\n[GEMINI RAW RESPONSE]\n", content, "\n")

        parsed = _extract_json(content)
        if not parsed:
            raise ValueError(f"Model response was not JSON:\n{content}")

        return parsed