## Overview

This project compares three artificial intelligence decision-making approaches in a survival simulation:

* **Behavior Tree (BT)**
* **Goal-Oriented Action Planning (GOAP)**
* **Hybrid BT + GOAP**

The simulation evaluates how each approach performs in environments with different levels of difficulty by measuring survival, planning efficiency, and decision-making performance.

This project was developed as part of a Bachelor's thesis.

---

## Features

* Three AI architectures:

  * Behavior Tree
  * GOAP planner
  * Hybrid BT + GOAP
* Three simulation scenarios:

  * Simple
  * Dynamic
  * Hostile
* Random world events (including storms)
* Resource collection
* Hunger, health, and energy management
* Enemy encounters
* Performance comparison between AI approaches
* Automatic generation of performance graphs

---

## Simulation

The agent must survive by:

* Finding food
* Collecting wood
* Building a fire
* Cooking food
* Escaping enemies
* Managing hunger and energy
* Reacting to environmental changes

The world changes every simulation step, forcing the AI to continuously make decisions.

---

## Performance Metrics

The experiment compares:

* Survival time
* Decision-making time
* Number of replanning operations
* Successful actions
* Average plan length

Each scenario is executed multiple times to obtain statistically meaningful results.

---

## Scenarios

### Simple

* Plenty of food
* Plenty of wood
* Few enemies

### Dynamic

* Moderate resources
* More frequent enemies
* Changing environment

### Hostile

* Limited resources
* Frequent enemies
* Highest difficulty

---

## Output

The program generates graphs such as:

* `survival_simple.png`

* `survival_dynamic.png`

* `survival_hostile.png`

* `decision_simple.png`

* `decision_dynamic.png`

* `decision_hostile.png`

* `replans_simple.png`

* `replans_dynamic.png`

* `replans_hostile.png`

These graphs compare the performance of the three AI approaches.

---

## Purpose

The objective of this project is to evaluate whether combining Behavior Trees with GOAP can improve decision quality and adaptability compared to using either approach independently.

---
