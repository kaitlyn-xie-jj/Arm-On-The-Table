# Arm-On-The-Table

Interactive Tabletop LLM Agent with Dynamic Grounding and Replanning


---

# Project Goal

The purpose of this project is not to build a realistic robot simulator.

Instead, the goal is to design a robust **agent harness** between a Large Language Model (LLM) and an interactive environment.

The system demonstrates how an LLM can:

* Observe a changing world
* Ground language instructions into environment entities
* Select structured actions
* Execute actions through an environment interface
* Replan when the environment changes

---

# Demo Overview

The environment is a tabletop manipulation world containing:

* Robot arm
* Bowl
* Plate
* Cup
* Fruits
* Books
* Dynamically added objects

Users can:

* Assign tasks in natural language
* Add new objects during execution
* Move existing objects during execution
* Observe the planner adapt to world changes

---

# Agent Harness

The system follows a closed-loop architecture.

```text
User Task
    ↓
Observation
    ↓
LLM Planner
    ↓
Structured Action
    ↓
Environment Executor
    ↓
World Update
    ↓
New Observation
    ↓
Replan
```

The LLM never directly modifies the world state.

Instead, it operates through a constrained action interface.

---

# Observation Representation

At every planning step the agent receives a structured observation.

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

The observation is intentionally structured rather than image-based.

The goal is to expose task-relevant information directly to the planner.

---

# Grounding Strategy

The project supports multiple forms of grounding.

## Object Grounding

Natural language references are mapped to environment objects.

Examples:

```text
red fruit
→ apple
```

```text
yellow fruit
→ banana
```

---

## Category Grounding

Instructions can refer to object categories rather than object names.

Examples:

```text
fruit
```

```text
container
```

---

## Temporal Grounding

The planner can reason about time-dependent object references.

Examples:

```text
Put the newest item in the bowl.
```

```text
Put the most recently added object in the bowl.
```

The grounding is recomputed from the current world state.

---

## Spatial Grounding

The planner supports spatial relationships between objects.

Examples:

```text
Move the apple near the bowl.
```

```text
Move the book left of the cup.
```

```text
Move the toy between the bowl and the plate.
```

Spatial relations are converted into executable environment actions.

---

# Affordance Representation

Each object exposes task-relevant affordances.

Examples:

```json
{
  "graspable": true,
  "placeable": false,
  "reachable": true
}
```

This allows the planner to reason about what actions are currently possible.

---

# Action Space

The planner operates through a structured action interface.

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

```json
{
  "type": "MOVE_RELATIVE",
  "target": "apple",
  "relation": "near",
  "reference": "bowl"
}
```

Supported actions:

* MOVE_TO
* GRASP
* RELEASE
* OPEN_GRIPPER
* CLOSE_GRIPPER
* MOVE_RELATIVE

---

# Dynamic Replanning

The environment can change during task execution.

Examples:

* New objects are inserted
* Existing objects are moved
* User interaction changes the world state

Whenever the world changes:

```text
Environment Change
        ↓
Observation Update
        ↓
Task Grounding Update
        ↓
Planner Replan
```

The agent therefore behaves as a closed-loop system rather than executing a fixed script.

---

# Example Tasks

## Basic Manipulation

```text
Put the apple in the bowl.
```

---

## Semantic Grounding

```text
Put the red fruit in the bowl.
```

```text
Put the yellow fruit on the plate.
```

---

## Dynamic World Reasoning

```text
Put the newest item in the bowl.
```

---

## Spatial Grounding

```text
Move the apple near the bowl.
```

```text
Move the book left of the cup.
```

---

# Example Replanning Scenario

Task:

```text
Put the newest item in the bowl.
```

Initial world:

```text
apple
banana
book
```

Planner selects:

```text
book
```

A new object is inserted:

```text
pear
```

Observation updates.

The planner recomputes grounding and switches its target to:

```text
pear
```

This demonstrates dynamic grounding and replanning.

---

# Design Decisions

## Why Structured Observations?

The challenge focuses on the interface between the agent and the environment.

Instead of solving perception, the project exposes:

* object attributes
* affordances
* task grounding
* execution progress

This allows the evaluation to focus on planning and interaction design.

---

## Why Structured Actions?

The planner cannot directly manipulate the world.

All interaction occurs through a constrained action space.

This improves:

* interpretability
* debuggability
* reproducibility

---

## Why Dynamic Replanning?

Real-world environments change.

A useful agent must adapt to:

* new objects
* moved objects
* changing task context

rather than following a fixed action script.

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

## Configure Gemini

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-2.5-flash
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

# Future Work

* Multi-object planning
* Richer spatial relations
* Scene graph observations
* Physics simulation
* Isaac Sim integration
* Real robot deployment
