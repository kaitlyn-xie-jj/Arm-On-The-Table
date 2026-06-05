from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from google import genai


SYSTEM_PROMPT = """
You are a tabletop robot planner.

Your job is to decide the NEXT action only.

Available actions:
1. MOVE_TO(target)
2. GRASP(target)
3. RELEASE()
4. OPEN_GRIPPER()
5. CLOSE_GRIPPER()

Rules:
- Return JSON ONLY.
- Choose exactly ONE action.
- Use the current observation.
- Prefer the shortest valid next step.
- Do not explain outside JSON.

You will also receive:
- task_grounding
- affordances
- progress

Rules:
- Prefer task_grounding.goal_object and task_grounding.goal_container when present.
- If a task refers to a color/category and no object matches it, do not guess a different category.
- Never substitute a container for a fruit or other mismatched object.
- Use affordances to avoid impossible actions.

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

        target_object = None
        target_container = None

        for name, obj in objs.items():
            category = obj.get("category", "")

            if category != "container" and name.lower() in task_l:
                target_object = name

            if category == "container" and name.lower() in task_l:
                target_container = name

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

        if not target_object:
            return {
                "reasoning": "No target object found.",
                "subgoal": "idle",
                "action": {
                    "type": "MOVE_TO",
                    "target": "home",
                },
            }

        holding = robot.get("holding")

        # Pick phase
        if holding != target_object:
            obj = objs[target_object]
            rpos = robot.get("pos", [0, 0])
            opos = obj.get("pos", [0, 0])

            d = abs(rpos[0] - opos[0]) + abs(rpos[1] - opos[1])

            if d > 0.06:
                return {
                    "reasoning": f"Move to {target_object}",
                    "subgoal": f"reach_{target_object}",
                    "action": {
                        "type": "MOVE_TO",
                        "target": target_object,
                    },
                }

            return {
                "reasoning": f"Pick {target_object}",
                "subgoal": f"pick_{target_object}",
                "action": {
                    "type": "GRASP",
                    "target": target_object,
                },
            }

        # Place phase
        if target_container:
            container = objs[target_container]
            rpos = robot.get("pos", [0, 0])
            cpos = container.get("pos", [0, 0])

            d = abs(rpos[0] - cpos[0]) + abs(rpos[1] - cpos[1])

            if d > 0.06:
                return {
                    "reasoning": f"Move to {target_container}",
                    "subgoal": f"reach_{target_container}",
                    "action": {
                        "type": "MOVE_TO",
                        "target": target_container,
                    },
                }

            return {
                "reasoning": "Release object",
                "subgoal": "release",
                "action": {
                    "type": "RELEASE",
                },
            }

        return {
            "reasoning": "Return home",
            "subgoal": "home",
            "action": {
                "type": "MOVE_TO",
                "target": "home",
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