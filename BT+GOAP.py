# =========================================================
# IMPORTS
# =========================================================

import random
import time
import heapq
import statistics
from dataclasses import dataclass
from typing import Dict, Callable

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SIMULATION_STEPS = 1000
RUNS_PER_SCENARIO = 100


# =========================================================
# WORLD
# =========================================================

class World:

    def __init__(self, scenario="simple"):

        self.scenario = scenario

        self.food_available = True
        self.wood_available = True

        self.enemy_near = False

        self.storm = False

        self.step_count = 0

    def update(self):

        self.step_count += 1

        self.storm = False

        # storm event
        if 100 <= self.step_count <= 150:
            self.storm = True

        # SIMPLE
        if self.scenario == "simple":

            self.food_available = random.random() < 0.8
            self.wood_available = random.random() < 0.8

            self.enemy_near = random.random() < 0.1

        # DYNAMIC
        elif self.scenario == "dynamic":

            self.food_available = random.random() < 0.5
            self.wood_available = random.random() < 0.6

            self.enemy_near = random.random() < 0.3

        # HOSTILE
        elif self.scenario == "hostile":

            self.food_available = random.random() < 0.45
            self.wood_available = random.random() < 0.5

            self.enemy_near = random.random() < 0.5

        # storm modifiers
        if self.storm:

            self.food_available = random.random() < 0.2
            self.enemy_near = random.random() < 0.6


# =========================================================
# BLACKBOARD
# =========================================================

class Blackboard:

    def __init__(self):

        self.memory = {

            "current_plan": [],
            "replans": 0,
            "last_goal": None,
            "interrupts": 0
        }


# =========================================================
# AGENT
# =========================================================

class Agent:

    def __init__(self):

        self.health = 100

        self.hunger = 20

        self.energy = 100

        self.has_raw_food = False
        self.has_cooked_food = False

        self.has_wood = False
        self.has_fire = False

        self.alive = True

        # metrics
        self.successful_actions = 0
        self.failed_actions = 0

        self.replans = 0

        self.total_explored_nodes = 0

        self.total_plan_length = 0

    def get_state(self, world):

        return {

            "hungry": self.hunger > 60,

            "tired": self.energy < 30,

            "enemy_near": world.enemy_near,

            "has_raw_food": self.has_raw_food,

            "has_cooked_food": self.has_cooked_food,

            "has_wood": self.has_wood,

            "has_fire": self.has_fire
        }

    def tick(self, world):

        self.hunger += random.randint(2, 5)

        self.energy -= random.randint(1, 3)

        if world.enemy_near:
            self.health -= random.randint(1, 4)

        if self.hunger > 100:
            self.health -= 2

        if world.storm:
            self.energy -= 2

        self.hunger = min(self.hunger, 120)

        self.energy = max(self.energy, 0)

        if self.health <= 0 or self.energy <= 0:
            self.alive = False


# =========================================================
# ACTIONS
# =========================================================

@dataclass
class Action:

    name: str

    preconditions: Dict

    effects: Dict

    cost: int

    def applicable(self, state):

        return all(
            state.get(k) == v
            for k, v in self.preconditions.items()
        )


ACTIONS = [

    Action(
        "gather_wood",
        {"has_wood": False},
        {"has_wood": True},
        2
    ),

    Action(
        "find_food",
        {"has_raw_food": False},
        {"has_raw_food": True},
        3
    ),

    Action(
        "make_fire",
        {
            "has_wood": True,
            "has_fire": False
        },
        {
            "has_fire": True
        },
        2
    ),

    Action(
        "cook_food",
        {
            "has_raw_food": True,
            "has_fire": True
        },
        {
            "has_cooked_food": True,
            "has_raw_food": False
        },
        2
    ),

    Action(
        "eat",
        {
            "has_cooked_food": True
        },
        {
            "hungry": False,
            "has_cooked_food": False
        },
        1
    ),

    Action(
        "rest",
        {},
        {
            "tired": False
        },
        2
    ),

    Action(
        "hide",
        {
            "enemy_near": True
        },
        {
            "enemy_near": False
        },
        1
    )
]


# =========================================================
# ACTION EXECUTION
# =========================================================

def execute_action(action, agent, world):

    success = False

    # GATHER WOOD
    if action.name == "gather_wood":

        agent.energy -= 5

        if world.wood_available:

            agent.has_wood = True

            success = True

    # FIND FOOD
    elif action.name == "find_food":

        agent.energy -= 8

        if world.food_available:

            agent.has_raw_food = True

            success = True

    # MAKE FIRE
    elif action.name == "make_fire":

        if agent.has_wood:

            agent.has_fire = True

            success = True

    # COOK FOOD
    elif action.name == "cook_food":

        if (
            agent.has_raw_food
            and
            agent.has_fire
        ):

            agent.has_cooked_food = True

            agent.has_raw_food = False

            success = True

    # EAT
    elif action.name == "eat":

        if agent.has_cooked_food:

            agent.hunger = max(
                0,
                agent.hunger - 70
            )

            agent.has_cooked_food = False

            success = True

    # REST
    elif action.name == "rest":

        recovery = 20

        if world.storm:
            recovery = 10

        agent.energy = min(
            100,
            agent.energy + recovery
        )

        success = True

    # HIDE
    elif action.name == "hide":

        if world.enemy_near:

            world.enemy_near = False

            success = True

    # METRICS
    if success:
        agent.successful_actions += 1
    else:
        agent.failed_actions += 1

    return success


# =========================================================
# GOAP PLANNER
# =========================================================

class Node:

    def __init__(
        self,
        state,
        parent=None,
        action=None,
        g=0,
        h=0
    ):

        self.state = state

        self.parent = parent

        self.action = action

        self.g = g

        self.h = h

        self.f = g + h

    def __lt__(self, other):

        return self.f < other.f


def serialize_state(state):

    return tuple(sorted(state.items()))


def apply_action(state, action):

    new_state = state.copy()

    for key, value in action.effects.items():
        new_state[key] = value

    return new_state


def heuristic(state):

    score = 0

    if state["hungry"]:
        score += 10

    if state["tired"]:
        score += 6

    if state["enemy_near"]:
        score += 12

    return score


# =========================================================
# UTILITY AI
# =========================================================

def calculate_utility(agent, world):

    utilities = {

        "survive": 0,
        "eat": 0,
        "rest": 0,
        "hide": 0
    }

    # HIDE
    if world.enemy_near:

        utilities["hide"] += 60

        if agent.health < 35:
            utilities["hide"] += 30

    # EAT
    if agent.hunger > 60:

        utilities["eat"] += agent.hunger

    # REST
    if agent.energy < 40:

        utilities["rest"] += (
            100 - agent.energy
        )

    # SURVIVE
    if agent.health < 50:

        utilities["survive"] += (
            100 - agent.health
        )

    selected_goal = max(
        utilities,
        key=utilities.get
    )

    return selected_goal


# =========================================================
# GOALS
# =========================================================

def goal_function(state):

    return (

        not state["hungry"]
        and
        not state["tired"]
        and
        not state["enemy_near"]
    )


def goal_function_by_goal(goal, state):

    # EAT
    if goal == "eat":

        return not state["hungry"]

    # REST
    elif goal == "rest":

        return not state["tired"]

    # HIDE
    elif goal == "hide":

        return not state["enemy_near"]

    # SURVIVE
    elif goal == "survive":

        return (
            not state["hungry"]
            and
            not state["enemy_near"]
        )

    return False


def reconstruct_plan(node):

    actions = []

    while node.parent:

        actions.append(node.action)

        node = node.parent

    actions.reverse()

    return actions


# =========================================================
# GOAP WITH GOALS
# =========================================================

def goap_plan_with_goal(
    initial_state,
    actions,
    goal
):

    open_list = []

    closed = set()

    explored_nodes = 0

    root = Node(
        initial_state,
        g=0,
        h=heuristic(initial_state)
    )

    heapq.heappush(open_list, root)

    while open_list:

        current = heapq.heappop(open_list)

        explored_nodes += 1

        if goal_function_by_goal(
            goal,
            current.state
        ):

            return (
                reconstruct_plan(current),
                explored_nodes
            )

        serialized = serialize_state(
            current.state
        )

        if serialized in closed:
            continue

        closed.add(serialized)

        for action in actions:

            if action.applicable(
                current.state
            ):

                new_state = apply_action(
                    current.state,
                    action
                )

                child = Node(
                    new_state,
                    parent=current,
                    action=action,
                    g=current.g + action.cost,
                    h=heuristic(new_state)
                )

                heapq.heappush(
                    open_list,
                    child
                )

    return [], explored_nodes


# =========================================================
# PURE GOAP
# =========================================================

def goap_plan(initial_state, actions):

    return goap_plan_with_goal(
        initial_state,
        actions,
        "survive"
    )


# =========================================================
# REPLANNING
# =========================================================

def should_replan(
    agent,
    world,
    current_plan
):

    if not current_plan:
        return True

    if world.enemy_near:
        return True

    if agent.hunger > 90:
        return True

    if agent.energy < 15:
        return True

    next_action = current_plan[0]

    if (
        next_action.name == "find_food"
        and
        not world.food_available
    ):
        return True

    if (
        next_action.name == "gather_wood"
        and
        not world.wood_available
    ):
        return True

    return False


# =========================================================
# BT CORE
# =========================================================

class BTNode:

    def run(
        self,
        agent,
        world,
        blackboard
    ):
        pass


class Selector(BTNode):

    def __init__(self, children):

        self.children = children

    def run(
        self,
        agent,
        world,
        blackboard
    ):

        for child in self.children:

            if child.run(
                agent,
                world,
                blackboard
            ):
                return True

        return False


class Sequence(BTNode):

    def __init__(self, children):

        self.children = children

    def run(
        self,
        agent,
        world,
        blackboard
    ):

        for child in self.children:

            if not child.run(
                agent,
                world,
                blackboard
            ):
                return False

        return True


class Condition(BTNode):

    def __init__(self, func: Callable):

        self.func = func

    def run(
        self,
        agent,
        world,
        blackboard
    ):

        return self.func(
            agent,
            world,
            blackboard
        )


class ActionNode(BTNode):

    def __init__(self, func: Callable):

        self.func = func

    def run(
        self,
        agent,
        world,
        blackboard
    ):

        return self.func(
            agent,
            world,
            blackboard
        )


# =========================================================
# PURE BT
# =========================================================

def build_behavior_tree():

    return Selector([

        # DANGER
        Sequence([

            Condition(
                lambda a, w, b:
                    w.enemy_near
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "hide",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            )
        ]),

        # HUNGER
        Sequence([

            Condition(
                lambda a, w, b:
                    a.hunger > 60
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "find_food",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "gather_wood",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "make_fire",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "cook_food",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "eat",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            )
        ]),

        # REST
        Sequence([

            Condition(
                lambda a, w, b:
                    a.energy < 30
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "rest",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            )
        ])
    ])


# =========================================================
# PURE GOAP
# =========================================================

class GOAPBrain:

    def __init__(self):

        self.current_plan = []

    def step(
        self,
        agent,
        world
    ):

        state = agent.get_state(world)

        if should_replan(
            agent,
            world,
            self.current_plan
        ):

            self.current_plan, explored = goap_plan(
                state,
                ACTIONS
            )

            # planning cost
            agent.energy -= 1

            agent.replans += 1

            agent.total_explored_nodes += explored

            agent.total_plan_length += len(
                self.current_plan
            )

        if self.current_plan:

            action = self.current_plan.pop(0)

            execute_action(
                action,
                agent,
                world
            )


# =========================================================
# HYBRID GOAP NODE
# =========================================================

class GOAPNode(BTNode):

    def run(
        self,
        agent,
        world,
        blackboard
    ):

        state = agent.get_state(world)

        current_plan = blackboard.memory[
            "current_plan"
        ]

        # UTILITY GOAL SELECTION
        selected_goal = calculate_utility(
            agent,
            world
        )

        blackboard.memory[
            "last_goal"
        ] = selected_goal

        # REPLANNING
        if should_replan(
            agent,
            world,
            current_plan
        ):

            current_plan, explored = (
                goap_plan_with_goal(
                    state,
                    ACTIONS,
                    selected_goal
                )
            )

            blackboard.memory[
                "current_plan"
            ] = current_plan

            blackboard.memory[
                "replans"
            ] += 1

            # planning cost
            agent.energy -= 1

            agent.replans += 1

            agent.total_explored_nodes += (
                explored
            )

            agent.total_plan_length += len(
                current_plan
            )

        # EXECUTE ACTION
        if current_plan:

            action = current_plan.pop(0)

            return execute_action(
                action,
                agent,
                world
            )

        return False


# =========================================================
# HYBRID BT + GOAP
# =========================================================

def build_hybrid_tree():

    return Selector([

        # EMERGENCY REACTIVE LAYER
        Sequence([

            Condition(
                lambda a, w, b:
                    (
                        w.enemy_near
                        and
                        a.health < 35
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "hide",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            )
        ]),

        # RECOVERY
        Sequence([

            Condition(
                lambda a, w, b:
                    (
                        a.health < 50
                        and
                        not w.enemy_near
                    )
            ),

            ActionNode(
                lambda a, w, b:
                    execute_action(
                        Action(
                            "rest",
                            {},
                            {},
                            1
                        ),
                        a,
                        w
                    )
            )
        ]),

        # STRATEGIC GOAP
        Sequence([

            Condition(
                lambda a, w, b:
                    a.alive
            ),

            GOAPNode()
        ])
    ])


# =========================================================
# RESULTS
# =========================================================

class SimulationResult:

    def __init__(self):

        self.survival_steps = 0

        self.average_decision_time = 0

        self.success_rate = 0

        self.average_explored_nodes = 0

        self.average_plan_length = 0

        self.replans = 0

        self.average_health = 0


# =========================================================
# SIMULATION
# =========================================================

def run_simulation(
    mode,
    scenario
):

    world = World(scenario)

    blackboard = Blackboard()

    agent = Agent()

    bt = build_behavior_tree()

    hybrid = build_hybrid_tree()

    goap = GOAPBrain()

    total_decision_time = 0

    health_sum = 0

    for step in range(
        SIMULATION_STEPS
    ):

        if not agent.alive:
            break

        world.update()

        start = time.perf_counter()

        # BT
        if mode == "BT":

            bt.run(
                agent,
                world,
                blackboard
            )

        # GOAP
        elif mode == "GOAP":

            goap.step(
                agent,
                world
            )

        # HYBRID
        elif mode == "HYBRID":

            hybrid.run(
                agent,
                world,
                blackboard
            )

        end = time.perf_counter()

        total_decision_time += (
            end - start
        )

        agent.tick(world)

        health_sum += agent.health

    result = SimulationResult()

    result.survival_steps = step

    result.average_decision_time = (
        total_decision_time /
        max(1, step)
    )

    total_actions = (

        agent.successful_actions +
        agent.failed_actions
    )

    if total_actions > 0:

        result.success_rate = (
            agent.successful_actions /
            total_actions
        )

    if agent.replans > 0:

        result.average_explored_nodes = (

            agent.total_explored_nodes /
            agent.replans
        )

        result.average_plan_length = (

            agent.total_plan_length /
            agent.replans
        )

    result.replans = agent.replans

    result.average_health = (
        health_sum / max(1, step)
    )

    return result


# =========================================================
# EXPERIMENTS
# =========================================================

def perform_experiment():

    scenarios = [

        "simple",

        "dynamic",

        "hostile"
    ]

    modes = [

        "BT",

        "GOAP",

        "HYBRID"
    ]

    final_results = {}

    for scenario in scenarios:

        print("\n============================")

        print(
            f"SCENARIO: {scenario.upper()}"
        )

        print("============================")

        final_results[scenario] = {}

        for mode in modes:

            survival_scores = []

            decision_times = []

            success_rates = []

            explored_nodes = []

            replans = []

            health_scores = []

            for _ in range(
                RUNS_PER_SCENARIO
            ):

                result = run_simulation(
                    mode,
                    scenario
                )

                survival_scores.append(
                    result.survival_steps
                )

                decision_times.append(
                    result.average_decision_time
                )

                success_rates.append(
                    result.success_rate
                )

                explored_nodes.append(
                    result.average_explored_nodes
                )

                replans.append(
                    result.replans
                )

                health_scores.append(
                    result.average_health
                )

            final_results[scenario][mode] = {

                "survival": survival_scores,

                "decision": decision_times,

                "success": success_rates,

                "explored": explored_nodes,

                "replans": replans,

                "health": health_scores
            }

            print(f"\n{mode}")

            print(
                f"Average survival: "
                f"{statistics.mean(survival_scores):.2f}"
            )

            print(
                f"Average decision time: "
                f"{statistics.mean(decision_times):.6f}s"
            )

            print(
                f"Average success rate: "
                f"{statistics.mean(success_rates):.2f}"
            )

            print(
                f"Average explored nodes: "
                f"{statistics.mean(explored_nodes):.2f}"
            )

            print(
                f"Average replans: "
                f"{statistics.mean(replans):.2f}"
            )

            print(
                f"Average health: "
                f"{statistics.mean(health_scores):.2f}"
            )

    return final_results


# =========================================================
# VISUALIZATION
# =========================================================

def build_graphs(results):

    for scenario in results.keys():

        # SURVIVAL
        plt.figure(figsize=(8, 5))

        survival_data = [

            results[scenario]["BT"]["survival"],

            results[scenario]["GOAP"]["survival"],

            results[scenario]["HYBRID"]["survival"]
        ]

        plt.boxplot(
            survival_data,
            labels=[
                "BT",
                "GOAP",
                "HYBRID"
            ]
        )

        plt.title(
            f"Survival Comparison ({scenario})"
        )

        plt.ylabel("Survival Steps")

        plt.grid(True)

        # SAVE
        plt.savefig(
            f"survival_{scenario}.png"
        )

        # DECISION TIME
        plt.figure(figsize=(8, 5))

        decision_means = [

            statistics.mean(
                results[scenario]["BT"]["decision"]
            ),

            statistics.mean(
                results[scenario]["GOAP"]["decision"]
            ),

            statistics.mean(
                results[scenario]["HYBRID"]["decision"]
            )
        ]

        plt.bar(
            [
                "BT",
                "GOAP",
                "HYBRID"
            ],
            decision_means
        )

        plt.title(
            f"Decision Time ({scenario})"
        )

        plt.ylabel("Seconds")

        plt.savefig(
            f"decision_{scenario}.png"
        )

        # REPLANS
        plt.figure(figsize=(8, 5))

        replans_means = [

            statistics.mean(
                results[scenario]["BT"]["replans"]
            ),

            statistics.mean(
                results[scenario]["GOAP"]["replans"]
            ),

            statistics.mean(
                results[scenario]["HYBRID"]["replans"]
            )
        ]

        plt.bar(
            [
                "BT",
                "GOAP",
                "HYBRID"
            ],
            replans_means
        )

        plt.title(
            f"Replanning Frequency ({scenario})"
        )

        plt.ylabel("Replans")

        plt.savefig(
            f"replans_{scenario}.png"
        )

    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    results = perform_experiment()

    build_graphs(results)