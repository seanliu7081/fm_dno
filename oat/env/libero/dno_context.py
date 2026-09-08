"""Privileged simulator context for a deliberately limited pick-place pilot.

This module does not import LIBERO or create a renderer. It only reads a live
LiberoEnv's simulator state. The goals come from that environment's parsed BDDL;
positions and grasp predicates come from its current simulator, not from a
candidate action suffix or from a learned rollout. These are oracle evaluation
inputs and must not be reported as an image-only deployment method.

Supported LIBERO-10 tasks have only independent On/In placement predicates.
Articulation, stove switching and other predicates are explicitly unsupported.
Sequential objects are selected by current contact and unsatisfied predicates.
Position subgoals and contact-based phases are heuristic: grasp orientation,
collision avoidance and contact dynamics are still left to the frozen policy.
"""
from __future__ import annotations

from typing import MutableMapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oat.env.libero.env import LiberoEnv


SUPPORTED_TASKS = (
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
)
PHASE_NAMES = ("reach", "grasp", "lift", "transport", "release")


class UnsupportedTaskError(ValueError):
    """The current oracle geometry adapter cannot represent this task."""


def _position(domain, name):
    try:
        value = domain.object_states_dict[name].get_geom_state()["pos"]
    except KeyError as error:
        raise UnsupportedTaskError(f"BDDL entity {name!r} has no live geometry state") from error
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Non-finite or malformed position for {name!r}")
    return value.copy()


def _bottom_height(obj):
    # A conservative scalar approximation; this is not an oriented collision box.
    bottom = np.asarray(obj.bottom_offset, dtype=np.float64)
    if bottom.shape != (3,) or not np.all(np.isfinite(bottom)):
        raise UnsupportedTaskError("Source object has no usable bottom_offset")
    return max(0.0, float(-bottom[2]))


def _placement_position(domain, predicate, source):
    relation, _, target = predicate
    position = _position(domain, target)
    if relation.lower() == "in":
        if target not in domain.object_sites_dict:
            raise UnsupportedTaskError("In targets must be explicit BDDL containment sites")
        # LIBERO's SiteObjectState.check_contain tests the source body position
        # inside this site's bounds, so its live center is a meaningful target.
        return position
    if target in domain.object_sites_dict:
        site = domain.object_sites_dict[target]
        rotation = np.asarray(domain.sim.data.get_site_xmat(target)).reshape(3, 3)
        size = np.asarray(site.size, dtype=np.float64)
        position[2] += float((np.abs(rotation) @ np.abs(size))[2]) + _bottom_height(source)
    else:
        target_obj = domain.get_object(target)
        if target_obj is None:
            raise UnsupportedTaskError(f"No placement surface for {target!r}")
        position[2] += float(np.asarray(target_obj.top_offset)[2]) + _bottom_height(source)
    return position


def _controller_context(robot):
    controller = robot.controller
    if getattr(controller, "control_dim", None) != 6 or not getattr(controller, "use_delta", True):
        raise UnsupportedTaskError("Oracle DNO currently requires a 6D delta OSC_POSE controller")
    values = {}
    for name in ("input_min", "input_max", "output_min", "output_max"):
        value = np.asarray(getattr(controller, name), dtype=np.float64)
        value = np.broadcast_to(value, (6,))[:3]
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Controller {name} must be finite")
        values[name] = value
    lo, hi = values["input_min"], values["input_max"]
    out_lo, out_hi = values["output_min"], values["output_max"]
    if np.any(hi <= lo) or np.any(out_hi <= out_lo):
        raise ValueError("Controller scaling ranges must have positive width")
    if not (np.allclose(lo, -hi) and np.allclose(out_lo, -out_hi)):
        raise UnsupportedTaskError("Oracle DNO currently requires symmetric OSC scaling")
    return {
        "translation_gain": ((out_hi - out_lo) / (hi - lo)).astype(np.float32),
        "translation_input_min": lo.astype(np.float32),
        "translation_input_max": hi.astype(np.float32),
    }


def build_task_context(
    env: "LiberoEnv",
    state: MutableMapping | None = None,
    *,
    grasp_distance: float = 0.045,
    lift_height: float = 0.08,
    transport_clearance: float = 0.08,
    placement_tolerance: float = 0.025,
) -> dict[str, np.ndarray]:
    """Read one oracle context; batch its arrays before calling the objective.

    Pass a fresh mutable ``state={}`` at each episode reset. It records source
    object heights from actual observations, plus a lift-completed flag based
    on actual measured elevation. With ``state=None``, the table support height
    is used instead; carrying a state dictionary is preferred for replanning.
    Simulator time going backwards or a task change also clears stale state.

    Returned scalar ``phase`` uses PHASE_NAMES indices; ``goal_pos`` is the
    current end-effector subgoal. ``active=False`` only when all true environment
    placement predicates hold and no task object is currently grasped. No
    inferred phase or geometric distance is substituted for environment reward.
    """
    for name, value in (
        ("grasp_distance", grasp_distance), ("lift_height", lift_height),
        ("transport_clearance", transport_clearance),
        ("placement_tolerance", placement_tolerance),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    domain = env.env.env
    task_name = env.task_name
    predicates = domain.parsed_problem["goal_state"]
    if not predicates or any(len(p) != 3 or p[0].lower() not in {"on", "in"} for p in predicates):
        raise UnsupportedTaskError(
            f"{task_name}: oracle DNO supports only pure On/In placement goals; "
            f"received {predicates!r}. Articulation and switch goals are unsupported."
        )
    if task_name not in SUPPORTED_TASKS:
        raise UnsupportedTaskError(
            f"{task_name}: no validated oracle adapter; supported tasks: {SUPPORTED_TASKS}"
        )
    sources = [p[1] for p in predicates]
    if len(set(sources)) != len(sources):
        raise UnsupportedTaskError("Multiple predicates for one source object require a custom adapter")
    if any(p[2] in sources for p in predicates):
        raise UnsupportedTaskError("Dependent stacking goals require a custom adapter")
    if len(domain.robots) != 1:
        raise UnsupportedTaskError("Oracle DNO currently requires one robot")
    robot = domain.robots[0]
    eef = np.asarray(domain.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64).copy()
    if eef.shape != (3,) or not np.all(np.isfinite(eef)):
        raise ValueError("End-effector position must be a finite 3-vector")
    objects = [domain.get_object(name) for name in sources]
    if any(obj is None for obj in objects):
        raise UnsupportedTaskError("A source BDDL entity is not a manipulable object")
    positions = [_position(domain, name) for name in sources]
    grasped = [bool(domain._check_grasp(robot.gripper, obj.contact_geoms)) for obj in objects]
    satisfied = [bool(domain._eval_predicate(p)) for p in predicates]
    active_indices = [i for i, held in enumerate(grasped) if held]
    if not active_indices:
        active_indices = [i for i, done in enumerate(satisfied) if not done]
    active = bool(active_indices)
    index = active_indices[0] if active else 0
    name, position, source = sources[index], positions[index], objects[index]
    place = _placement_position(domain, predicates[index], source)

    now = float(domain.sim.data.time)
    if state is not None:
        if state.get("task_name", task_name) != task_name or now < state.get("sim_time", now):
            state.clear()
        state["task_name"] = task_name
        state["sim_time"] = now
        initial = state.setdefault("initial_object_z", {})
        for obj_name, obj_pos in zip(sources, positions):
            initial.setdefault(obj_name, float(obj_pos[2]))
        initial_z = initial[name]
    else:
        initial_z = float(np.asarray(domain.workspace_offset)[2]) + _bottom_height(source)
    clearance_z = max(initial_z + lift_height, float(place[2]) + transport_clearance)
    xy_error = float(np.linalg.norm(position[:2] - place[:2]))
    held = grasped[index]
    completed = satisfied[index]
    goal = eef.copy()
    phase = 4
    desired_gripper = -1.0

    if active and not held:
        # Grasp orientation remains supplied by the frozen policy. The target
        # here is the object body's current center, not an invented grasp pose.
        goal = position.copy()
        phase = 0 if np.linalg.norm(eef - goal) > grasp_distance else 1
        desired_gripper = -1.0 if phase == 0 else 1.0
        if state is not None:
            state.pop("lifted_object", None)
    elif active:
        offset = eef - position  # measured grip offset, not predicted attachment
        desired_gripper = 1.0
        if completed:
            phase = 4
            desired_gripper = -1.0
        elif xy_error <= placement_tolerance:
            # Descend only after horizontal alignment; release near the actual
            # support/containment goal, while success still uses LIBERO's predicate.
            goal = place + offset
            phase = 3
            if abs(float(position[2] - place[2])) <= placement_tolerance:
                phase = 4
                desired_gripper = -1.0
        else:
            lifted = position[2] >= clearance_z - placement_tolerance
            if state is not None:
                if lifted:
                    state["lifted_object"] = name
                lifted = lifted or state.get("lifted_object") == name
            if not lifted:
                phase = 2
                goal = eef.copy()
                goal[2] += clearance_z - position[2]
            else:
                phase = 3
                goal = place + offset
                goal[2] = clearance_z + offset[2]
    result = {
        "eef_pos": eef.astype(np.float32),
        "goal_pos": goal.astype(np.float32),
        "desired_gripper": np.asarray(desired_gripper, dtype=np.float32),
        "phase": np.asarray(phase, dtype=np.int64),
        "active": np.asarray(active, dtype=bool),
        "object_pos": position.astype(np.float32),
        "place_pos": place.astype(np.float32),
        "object_index": np.asarray(index, dtype=np.int64),
        "grasped": np.asarray(held, dtype=bool),
        "predicate_satisfied": np.asarray(completed, dtype=bool),
    }
    result.update(_controller_context(robot))
    return result
