from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import math
import random

Point = Tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


@dataclass
class ObjectState:

    name: str

    pos: Point

    color: str

    category: str = "object"

    movable: bool = True

    size: int = 18

    visible: bool = True

    held: bool = False

    container: Optional[str] = None

    spawn_step: int = 0

    def to_dict(self):

        d = asdict(self)

        d["pos"] = [
            round(self.pos[0], 3),
            round(self.pos[1], 3),
        ]

        return d


@dataclass
class RobotState:
    pos: Point = (0.22, 0.50)
    gripper_open: bool = True
    holding: Optional[str] = None
    home: Point = (0.12, 0.50)

    def to_dict(self):
        return {
            "pos": [round(self.pos[0], 3), round(self.pos[1], 3)],
            "gripper_open": self.gripper_open,
            "holding": self.holding,
            "home": [round(self.home[0], 3), round(self.home[1], 3)],
        }


@dataclass
class TabletopWorld:
    width: int = 800
    height: int = 520
    step_count: int = 0
    task: str = ""
    done: bool = False
    success: bool = False
    log: List[str] = field(default_factory=list)
    robot: RobotState = field(default_factory=RobotState)
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    bowl_area: Tuple[float, float, float, float] = (0.73, 0.58, 0.18, 0.16)
    goal_object: Optional[str] = None
    goal_container: Optional[str] = None
    goal_object_candidates: List[str] = field(default_factory=list)
    goal_container_candidates: List[str] = field(default_factory=list)
    last_action: Optional[dict] = None
    last_result: Optional[str] = None
    task_mode: str = "static"
    goal_object: Optional[str] = None
    goal_container: Optional[str] = None

    def reset(self):

        self.step_count = 0

        self.task = ""

        self.done = False

        self.success = False

        self.log = []

        self.robot = RobotState()

        self.objects = {

            "apple": ObjectState(
                name="apple",
                pos=(0.34, 0.32),
                color="red",
                category="fruit"
            ),

            "banana": ObjectState(
                name="banana",
                pos=(0.48, 0.28),
                color="yellow",
                category="fruit"
            ),

            "book": ObjectState(
                name="book",
                pos=(0.62, 0.42),
                color="green",
                category="book"
            ),

            "cup": ObjectState(
                name="cup",
                pos=(0.56, 0.22),
                color="orange",
                category="container"
            ),

            "bowl": ObjectState(
                name="bowl",
                pos=(0.78, 0.72),
                color="blue",
                category="container",
                movable=False,
                size=28
            ),

            "plate": ObjectState(
                name="plate",
                pos=(0.78, 0.28),
                color="white",
                category="container",
                movable=False,
                size=28
            ),
        }

        self.goal_object = None
        self.goal_container = None

        self.goal_object_candidates = []
        self.goal_container_candidates = []
        self.last_action = None
        self.last_result = None

        return self.observe()

    def observe(self):
        goal_object, goal_object_candidates = self._resolve_goal_object()
        goal_container, goal_container_candidates = self._resolve_goal_container()

        self.goal_object = goal_object
        self.goal_container = goal_container
        self.goal_object_candidates = goal_object_candidates
        self.goal_container_candidates = goal_container_candidates

        task_grounding = {
            "mode": self._task_mode(),
            "goal_object": goal_object,
            "goal_container": goal_container,
            "goal_object_candidates": goal_object_candidates,
            "goal_container_candidates": goal_container_candidates,
        }

        if goal_object and goal_object in self.objects:
            obj = self.objects[goal_object]
            task_grounding["goal_attributes"] = {
                "category": obj.category,
                "color": obj.color,
            }
        else:
            task_grounding["goal_attributes"] = {}

        affordances = {}
        robot_pos = self.robot.pos
        for name, obj in self.objects.items():
            affordances[name] = {
                "graspable": obj.movable and obj.category != "container",
                "placeable": obj.category == "container",
                "reachable": dist(robot_pos, obj.pos) <= 0.08,
                "held": obj.held,
                "spawn_step": obj.spawn_step,
            }

        progress = {
            "phase": (
                "done" if self.done
                else "seek_object" if self.robot.holding is None
                else "seek_container" if self.goal_container
                else "carrying"
            ),
            "last_action": self.last_action,
            "last_result": self.last_result,
        }

        return {
            "task": self.task,
            "step": self.step_count,
            "done": self.done,
            "success": self.success,
            "task_grounding": task_grounding,
            "progress": progress,
            "robot": self.robot.to_dict(),
            "affordances": affordances,
            "objects": {name: obj.to_dict() for name, obj in self.objects.items()},
            "bowl_area": [round(v, 3) for v in self.bowl_area],
        }
    
    
    def _build_task_grounding(self):
        goal_attributes = {}

        if self.goal_object and self.goal_object in self.objects:
            obj = self.objects[self.goal_object]
            goal_attributes = {
                "category": obj.category,
                "color": obj.color,
            }

        return {
            "goal_object": self.goal_object,
            "goal_container": self.goal_container,
            "goal_object_candidates": self.goal_object_candidates,
            "goal_container_candidates": self.goal_container_candidates,
            "goal_attributes": goal_attributes,
        }


    def _build_affordances(self):
        affordances = {}
        robot_pos = self.robot.pos

        for name, obj in self.objects.items():
            affordances[name] = {
                "graspable": obj.movable and obj.category != "container",
                "placeable": obj.category == "container",
                "reachable": dist(robot_pos, obj.pos) <= 0.08,
            }

        return affordances

    def _build_task_mode(self) -> str:
        task_l = self.task.lower()
        dynamic_keywords = (
            "newest",
            "latest",
            "most recent",
            "newly added",
            "current",
        )
        return "dynamic" if any(k in task_l for k in dynamic_keywords) else "static"

    def _build_progress(self):
        if self.done:
            phase = "done"
        elif self.robot.holding is None:
            phase = "seek_object" if self.goal_object else "idle"
        elif self.goal_container:
            phase = "seek_container"
        else:
            phase = "carrying"

        return {
            "phase": phase,
            "last_action": self.last_action,
            "last_result": self.last_result,
        }

    def world_to_px(self, p: Point) -> Tuple[int, int]:
        return int(p[0] * self.width), int(p[1] * self.height)

    def px_to_world(self, x: int, y: int) -> Point:
        return x / self.width, y / self.height

    def is_in_bowl(self, obj_name: str) -> bool:
        if obj_name not in self.objects:
            return False
        ox, oy = self.objects[obj_name].pos
        bx, by, bw, bh = self.bowl_area
        return bx <= ox <= bx + bw and by <= oy <= by + bh
    
    def _task_mode(self) -> str:
        task_l = self.task.lower()
        dynamic_keywords = (
            "newest",
            "latest",
            "most recent",
            "newly added",
            "current",
        )
        return "dynamic" if any(k in task_l for k in dynamic_keywords) else "static"

    def _resolve_goal_object(self):
        task_l = self.task.lower()
        mode = self._task_mode()

        candidates: List[str] = []

        for name, obj in self.objects.items():
            if obj.category == "container":
                continue

            matched = False

            if name.lower() in task_l:
                matched = True

            if obj.color.lower() in task_l:
                matched = True

            if "fruit" in task_l and obj.category == "fruit":
                matched = True

            if "item" in task_l:
                matched = True

            if matched:
                candidates.append(name)

        # Dynamic tasks like "newest item" should prefer the highest spawn_step.
        if mode == "dynamic" and any(
            k in task_l for k in ("newest", "latest", "most recent", "newly added", "current")
        ):
            if not candidates:
                candidates = [
                    name for name, obj in self.objects.items()
                    if obj.category != "container"
                ]

            candidates = sorted(
                candidates,
                key=lambda n: (self.objects[n].spawn_step, n),
                reverse=True,
            )
        else:
            candidates = sorted(
                candidates,
                key=lambda n: (self.objects[n].spawn_step, n),
                reverse=True,
            )

        return (candidates[0] if candidates else None), candidates
    
    def _resolve_goal_container(self):
        task_l = self.task.lower()
        candidates: List[str] = []

        for name, obj in self.objects.items():
            if obj.category != "container":
                continue

            matched = False

            if name.lower() in task_l:
                matched = True

            if "bowl" in task_l and name == "bowl":
                matched = True

            if "plate" in task_l and name == "plate":
                matched = True

            if "cup" in task_l and name == "cup":
                matched = True

            if matched:
                candidates.append(name)

        candidates = sorted(
            candidates,
            key=lambda n: (self.objects[n].spawn_step, n),
            reverse=True,
        )

        return (candidates[0] if candidates else None), candidates

    def set_task(self, task: str):
        self.task = task.strip()
        self.done = False
        self.success = False
        self.goal_object = None
        self.goal_container = None
        self.goal_object_candidates = []
        self.goal_container_candidates = []

        task_l = self.task.lower()

        object_scores = []
        container_scores = []

        for name, obj in self.objects.items():
            score = 0

            if name.lower() in task_l:
                score += 10

            if obj.color.lower() in task_l:
                score += 4

            if obj.category.lower() in task_l and obj.category != "container":
                score += 3

            if obj.category == "fruit" and "fruit" in task_l:
                score += 3

            if obj.category == "container" and any(x in task_l for x in ["bowl", "plate", "cup", "box", "tray"]):
                if name.lower() in task_l:
                    score += 6

            if score > 0:
                if obj.category == "container":
                    container_scores.append((score, name))
                else:
                    object_scores.append((score, name))

        object_scores.sort(reverse=True)
        container_scores.sort(reverse=True)

        self.goal_object_candidates = [name for _, name in object_scores]
        self.goal_container_candidates = [name for _, name in container_scores]

        if self.goal_object_candidates:
            self.goal_object = self.goal_object_candidates[0]
        if self.goal_container_candidates:
            self.goal_container = self.goal_container_candidates[0]

        self.log.append(f"Task set: {self.task}")

    def _unique_object_name(self, base_name: str) -> str:
        if base_name not in self.objects:
            return base_name

        idx = 2
        while f"{base_name}_{idx}" in self.objects:
            idx += 1
        return f"{base_name}_{idx}"

    def add_random_object(self) -> str:
        catalog = [
            {"name": "pear", "color": "green", "category": "fruit", "movable": True},
            {"name": "orange", "color": "orange", "category": "fruit", "movable": True},
            {"name": "toy", "color": "purple", "category": "object", "movable": True},
            {"name": "ball", "color": "blue", "category": "object", "movable": True},
        ]

        template = random.choice(catalog)
        name = self._unique_object_name(template["name"])

        # randomly pick up a place that's not crowded
        for _ in range(100):
            x = random.uniform(0.25, 0.75)
            y = random.uniform(0.18, 0.78)

            too_close = False
            for obj in self.objects.values():
                if dist((x, y), obj.pos) < 0.10:
                    too_close = True
                    break

            if not too_close:
                self.objects[name] = ObjectState(
                    name=name,
                    pos=(x, y),
                    color=template["color"],
                    category=template["category"],
                    movable=template["movable"],
                    spawn_step=self.step_count,
                )
                self.log.append(f"Added object: {name}")
                return name

        # fallback
        self.objects[name] = ObjectState(
            name=name,
            pos=(0.50, 0.50),
            color=template["color"],
            category=template["category"],
            movable=template["movable"],
        )
        self.log.append(f"Added object: {name}")
        return name

    def _move_towards(self, target: Point):
        self.robot.pos = target
        if self.robot.holding and self.robot.holding in self.objects:
            self.objects[self.robot.holding].pos = target

    def execute(self, action: Dict):
        if self.done:
            return {"ok": False, "message": "Task already complete.", "observation": self.observe()}

        kind = action.get("type")
        target = action.get("target")
        note = ""
        ok = True

        if kind == "MOVE_TO":

            if target == "home":

                self._move_towards(
                    self.robot.home
                )

                note = "Returned home."

            elif target in self.objects:

                self._move_towards(
                    self.objects[target].pos
                )

                note = f"Moved to {target}."

            else:

                ok = False

                note = (
                    f"Unknown MOVE_TO target: "
                    f"{target}"
                )

        elif kind == "GRASP":
            if not self.robot.gripper_open:
                ok = False
                note = "Gripper is already closed."
            elif target not in self.objects:
                ok = False
                note = f"Unknown grasp target: {target}"
            else:
                obj = self.objects[target]

                if not obj.movable:

                    ok = False

                    note = (
                        f"{target} "
                        f"cannot be grasped."
                    )

                elif dist(
                    self.robot.pos,
                    obj.pos
                ) > 0.08:

                    ok = False

                    note = (
                        f"{target} "
                        f"is too far to grasp."
                    )

                else:

                    self.robot.gripper_open = False

                    self.robot.holding = target

                    obj.held = True

                    obj.pos = self.robot.pos

                    note = (
                        f"Grasped {target}."
                    )

        elif kind == "RELEASE":

            held = self.robot.holding

            if held is None:

                ok = False

                note = "Nothing to release."

            else:

                obj = self.objects[held]

                self.robot.gripper_open = True

                self.robot.holding = None

                obj.held = False

                obj.pos = self.robot.pos

                nearest_container = None

                nearest_dist = 999

                for name, candidate in self.objects.items():

                    if (
                        candidate.category
                        != "container"
                    ):
                        continue

                    d = dist(
                        candidate.pos,
                        obj.pos
                    )

                    if d < nearest_dist:

                        nearest_dist = d

                        nearest_container = name

                if (
                    nearest_container
                    and nearest_dist < 0.10
                ):

                    obj.container = (
                        nearest_container
                    )

                    note = (
                        f"Released {held} into "
                        f"{nearest_container}."
                    )

                else:

                    note = (
                        f"Released {held}."
                    )

                #
                # success check
                #

                if (
                    self.goal_object
                    and self.goal_container
                    and held
                    == self.goal_object
                    and obj.container
                    == self.goal_container
                ):

                    self.success = True

                    self.done = True

                    note += (
                        " Task complete!"
                    )

        elif kind == "OPEN_GRIPPER":
            self.robot.gripper_open = True
            note = "Gripper opened."

        elif kind == "CLOSE_GRIPPER":
            self.robot.gripper_open = False
            note = "Gripper closed."

        else:
            ok = False
            note = f"Unknown action type: {kind}"

        self.step_count += 1

        self.last_action = action
        self.last_result = note

        self.log.append(f"{self.step_count:02d}. {action} -> {note}")
        return {"ok": ok, "message": note, "observation": self.observe()}

    def move_object(self, name: str, x: float, y: float):
        if name not in self.objects:
            return False, f"Unknown object: {name}"

        obj = self.objects[name]

        # If you don't want bowl/plate to be moved, keep this restriction
        if not obj.movable:
            return False, f"{name} is not movable."

        # Constrain movement within tabletop bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))

        obj.pos = (x, y)

        # Sync robot position if robot is holding this object
        if self.robot.holding == name:
            self.robot.pos = (x, y)

        self.log.append(f"Moved object {name} to ({x:.3f}, {y:.3f})")
        return True, f"Moved {name}."


    def render_events(self):
        # For UI reuse, return positions in pixel coordinates.
        return {
            "width": self.width,
            "height": self.height,
            "robot": {"x": self.world_to_px(self.robot.pos)[0], "y": self.world_to_px(self.robot.pos)[1], "holding": self.robot.holding},
            "objects": {
                name: {**obj.to_dict(), "x": self.world_to_px(obj.pos)[0], "y": self.world_to_px(obj.pos)[1]}
                for name, obj in self.objects.items()
            },
            "bowl_area": {
                "x": int(self.bowl_area[0] * self.width),
                "y": int(self.bowl_area[1] * self.height),
                "w": int(self.bowl_area[2] * self.width),
                "h": int(self.bowl_area[3] * self.height),
            },
        }
