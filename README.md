# Arm-On-The-Table

An interactive tabletop robot agent that connects a Large Language Model (LLM) to a dynamic environment through a structured observation and action interface.

This project was built for the **Humanoid Summer Internship Challenge**.

---

# Overview

The goal of this project is not to build a realistic robot simulator, but to design a robust **agent harness** between an intelligent planner and an environment.

The system demonstrates:

* A dynamic tabletop world
* Structured observations
* Structured actions
* LLM-driven planning
* Environment feedback
* Continuous replanning

The agent observes the world, chooses an action, executes it, receives updated observations, and replans until the task is completed.

---

# System Architecture

```text
User Task
    ↓
Observation
    ↓
LLM Planner
    ↓
Structured Action
    ↓
Executor
    ↓
Environment Update
    ↓
New Observation
    ↓
Replan
```

The planner never directly manipulates the world.

Instead, it outputs high-level actions which are executed by the environment.

---

# Environment

The environment is a 2D tabletop world containing:

* Robot Arm
* Bowl
* Plate
* Cup
* Fruits
* Books
* Dynamically Added Objects

Objects can be added during runtime.

The world state changes continuously and the planner must react to those changes.

---

# Observation Design

The agent receives a structured observation at every planning step.

Example:

```json
{
  "task": "Put the red fruit in the bowl",

  "task_grounding": {
    "goal_object": "apple",
    "goal_container": "bowl"
  },

  "progress": {
    "phase": "seek_object"
  },

  "robot": {
    "holding": null
  },

  "affordances": {
    "apple": {
      "graspable": true,
      "reachable": true
    }
  },

  "objects": {
    ...
  }
}
```

The observation contains three explicit reasoning layers:

### Task Grounding

Maps natural language instructions to candidate objects and containers.

### Affordances

Describes what actions are possible for each object.

Examples:

* graspable
* placeable
* reachable

### Progress

Tracks the current execution state.

Examples:

* seek_object
* seek_container
* carrying
* done

---

# Action Space

The planner can only use structured actions.

```json
{
  "type": "MOVE_TO",
  "target": "apple"
}
```

```json
{
  "type": "GRASP",
  "target": "apple"
}
```

```json
{
  "type": "RELEASE"
}
```

Available actions:

* MOVE_TO
* GRASP
* RELEASE
* OPEN_GRIPPER
* CLOSE_GRIPPER

---

# Dynamic Replanning

The environment can change while the task is running.

New objects may appear at runtime.

For tasks that depend on temporal context such as:

```text
Put the newest item in the bowl.
```

the agent continuously re-evaluates the observation and updates its target.

This demonstrates a closed-loop planning architecture rather than a fixed script.

---

# Example Tasks

Basic manipulation:

```text
Put the apple in the bowl.
```

Semantic grounding:

```text
Put the red fruit in the bowl.
```

```text
Put the yellow fruit on the plate.
```

Dynamic world reasoning:

```text
Put the newest item in the bowl.
```

Relational reasoning:

```text
Move the book next to the cup.
```

---

# Running the Project

## Create Environment

```bash
conda create -n humanoid_agent python=3.11
conda activate humanoid_agent
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch

```bash
uvicorn app.server:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# Gemini Integration

Set your Gemini API key:

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Linux / macOS:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Optional:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

If the API is unavailable, the system falls back to a deterministic planner.

---

# Design Choices

This project prioritizes:

* Agent harness quality
* Observation design
* Replanning capability
* Simplicity
* Explainability

Instead of relying on raw images or physics-heavy simulation, the agent operates on structured world state representations that expose the information necessary for task completion.

This allows the focus to remain on the interface between the LLM and the environment.

---

# Future Extensions

* Multi-object planning
* Multi-step task decomposition
* Dynamic object insertion
* Scene graph observations
* Physics-based simulators
* Isaac Sim integration
* Real robot execution
