from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oat.dno.task_objectives import TaskGeometryObjective
from oat.env.libero.dno_context import (
    PHASE_NAMES, SUPPORTED_TASKS, UnsupportedTaskError, build_task_context,
)


def context(batch=1):
    return {
        "eef_pos": torch.zeros(batch, 3),
        "goal_pos": torch.tensor([[0.4, 0, 0]]).expand(batch, -1),
        "translation_gain": torch.ones(batch),
    }


def test_task_progress_changes_cost_and_gradient_ignores_unexecuted_suffix():
    objective = TaskGeometryObjective(n_action_steps=2, path_weight=0)
    actions = torch.zeros(2, 4, 7)
    actions[0, :2, 0] = 0.2
    actions[1, :2, 0] = -0.2
    actions[:, 2:] = 99
    actions.requires_grad_()
    cost = objective(actions, context(2))
    assert cost.shape == (2,)
    assert cost[0] == 0
    assert cost[1] > 0
    cost.sum().backward()
    assert torch.all(actions.grad[1, :2, 0] < 0)
    assert torch.count_nonzero(actions.grad[:, 2:]) == 0
    assert torch.count_nonzero(actions.grad[..., 3:6]) == 0


def test_inactive_rows_have_zero_cost_and_zero_action_gradient():
    actions = torch.zeros(2, 4, 7, requires_grad=True)
    ctx = context(2)
    ctx["active"] = torch.tensor([True, False])
    cost = TaskGeometryObjective(n_action_steps=2)(actions, ctx)
    assert cost[0] > 0 and cost[1] == 0
    cost.sum().backward()
    assert torch.count_nonzero(actions.grad[1]) == 0


def test_gripper_cost_respects_open_close_command_sign():
    actions = torch.zeros(2, 2, 7)
    actions[0, :, 6] = -1
    actions[1, :, 6] = 1
    objective = TaskGeometryObjective(n_action_steps=2, position_weight=0, gripper_weight=1)
    ctx = context(2)
    ctx["desired_gripper"] = torch.tensor([-1, -1])
    torch.testing.assert_close(objective(actions, ctx), torch.tensor([0.0, 4.0]))
    ctx["desired_gripper"] = torch.tensor([1, 1])
    torch.testing.assert_close(objective(actions, ctx), torch.tensor([4.0, 0.0]))


def test_controller_scale_and_input_clipping_are_applied_before_accumulation():
    actions = torch.zeros(1, 2, 7)
    actions[..., 0] = 5
    ctx = context()
    ctx["goal_pos"] = torch.tensor([[0.2, 0, 0]])
    ctx["translation_gain"] = torch.tensor([[0.05, 0.03, 0.02]])
    ctx["translation_input_max"] = 2.0
    ctx["translation_input_min"] = -2.0
    assert TaskGeometryObjective(n_action_steps=2, path_weight=0)(actions, ctx).item() == 0


@pytest.mark.parametrize("steps,horizon", [(0, 4), (5, 4)])
def test_invalid_executed_prefix_fails(steps, horizon):
    with pytest.raises(ValueError):
        TaskGeometryObjective(n_action_steps=steps)(torch.zeros(1, horizon, 7), context())


class FakeObjectState:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def get_geom_state(self):
        return {"pos": self.position.copy()}


class FakeDomain:
    def __init__(self, predicates=None):
        self.parsed_problem = {
            "goal_state": predicates or [["in", "book", "caddy_site"]]
        }
        self.object_states_dict = {
            "book": FakeObjectState([0, 0, 0.05]),
            "second": FakeObjectState([0.3, 0.1, 0.05]),
            "caddy_site": FakeObjectState([0.5, 0, 0.10]),
            "plate": FakeObjectState([0.5, 0, 0.01]),
        }
        self.objects = {
            name: SimpleNamespace(
                contact_geoms=[name], bottom_offset=np.array([0, 0, -0.05]),
                top_offset=np.array([0, 0, 0.02]),
            )
            for name in ("book", "second", "plate")
        }
        self.object_sites_dict = {
            "caddy_site": SimpleNamespace(size=np.array([0.05, 0.05, 0.05])),
        }
        self.sim = SimpleNamespace(data=SimpleNamespace(
            site_xpos=np.array([[0.3, 0, 0.2]]), time=1.0,
            get_site_xmat=lambda name: np.eye(3),
        ))
        self.robots = [SimpleNamespace(
            eef_site_id=0, gripper=object(), controller=SimpleNamespace(
                control_dim=6, use_delta=True,
                input_min=-np.ones(6), input_max=np.ones(6),
                output_min=-np.array([0.05] * 3 + [0.5] * 3),
                output_max=np.array([0.05] * 3 + [0.5] * 3),
            ),
        )]
        self.workspace_offset = [0, 0, 0]
        self.held = set()
        self.satisfied = set()

    def get_object(self, name):
        return self.objects.get(name)

    def _check_grasp(self, gripper, contact_geoms):
        return contact_geoms[0] in self.held

    def _eval_predicate(self, predicate):
        return predicate[1] in self.satisfied


def fake_env(predicates=None):
    domain = FakeDomain(predicates)
    return SimpleNamespace(task_name=SUPPORTED_TASKS[0], env=SimpleNamespace(env=domain)), domain


def test_context_transitions_use_observed_grasp_position_and_predicates():
    env, domain = fake_env()
    state = {}
    ctx = build_task_context(env, state)
    assert PHASE_NAMES[int(ctx["phase"])] == "reach"
    assert ctx["desired_gripper"] == -1
    np.testing.assert_allclose(ctx["goal_pos"], [0, 0, 0.05])
    np.testing.assert_allclose(ctx["translation_gain"], [0.05] * 3)

    domain.sim.data.site_xpos[0] = [0, 0, 0.07]
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 1 and ctx["desired_gripper"] == 1
    # A closed-command intention alone was insufficient: grasp requires contact.
    domain.held.add("book")
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 2
    assert ctx["goal_pos"][2] > ctx["eef_pos"][2]

    domain.object_states_dict["book"].position[2] = 0.20
    domain.sim.data.site_xpos[0] = [0, 0, 0.22]
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 3 and ctx["desired_gripper"] == 1
    assert ctx["goal_pos"][0] == pytest.approx(0.5)

    domain.object_states_dict["book"].position[:] = [0.5, 0, 0.2]
    domain.sim.data.site_xpos[0] = [0.5, 0, 0.22]
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 3 and ctx["goal_pos"][2] < ctx["eef_pos"][2]

    domain.object_states_dict["book"].position[:] = [0.5, 0, 0.1]
    domain.sim.data.site_xpos[0] = [0.5, 0, 0.12]
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 4 and ctx["desired_gripper"] == -1
    assert ctx["active"]  # Near the goal is not actual predicate success.
    domain.satisfied.add("book")
    assert build_task_context(env, state)["active"]  # Still physically grasped.
    domain.held.clear()
    ctx = build_task_context(env, state)
    assert not ctx["active"]
    np.testing.assert_array_equal(ctx["goal_pos"], ctx["eef_pos"])


def test_completed_object_is_released_before_switching_to_another_object():
    env, domain = fake_env([
        ["in", "book", "caddy_site"], ["in", "second", "caddy_site"],
    ])
    domain.satisfied.add("book")
    domain.held.add("book")
    ctx = build_task_context(env, {})
    assert ctx["object_index"] == 0 and ctx["phase"] == 4
    domain.held.clear()
    ctx = build_task_context(env, {})
    assert ctx["object_index"] == 1
    np.testing.assert_allclose(ctx["object_pos"], [0.3, 0.1, 0.05])


def test_lost_grasp_cannot_keep_using_attached_object_transport_phase():
    env, domain = fake_env()
    state = {"lifted_object": "book"}
    ctx = build_task_context(env, state)
    assert ctx["phase"] == 0
    assert "lifted_object" not in state


def test_on_target_uses_surface_height_and_object_bottom_offset():
    env, _ = fake_env([["on", "book", "plate"]])
    ctx = build_task_context(env, {})
    np.testing.assert_allclose(ctx["place_pos"], [0.5, 0, 0.08])


def test_context_state_resets_after_actual_sim_time_rewinds():
    env, domain = fake_env()
    state = {}
    build_task_context(env, state)
    state["lifted_object"] = "book"
    domain.sim.data.time = 0
    domain.object_states_dict["book"].position[2] = 0.1
    build_task_context(env, state)
    assert state["initial_object_z"]["book"] == pytest.approx(0.1)
    assert "lifted_object" not in state


@pytest.mark.parametrize("goal", [
    [["close", "drawer"], ["in", "book", "caddy_site"]],
    [["turnon", "stove"]],
    [["on", "book", "second"], ["in", "second", "caddy_site"]],
])
def test_unsupported_task_structure_fails_explicitly(goal):
    env, _ = fake_env(goal)
    with pytest.raises(UnsupportedTaskError):
        build_task_context(env)


def test_unvalidated_task_and_absolute_controller_fail_explicitly():
    env, domain = fake_env()
    env.task_name = "new_unvalidated_task"
    with pytest.raises(UnsupportedTaskError, match="no validated oracle adapter"):
        build_task_context(env)
    env.task_name = SUPPORTED_TASKS[0]
    domain.robots[0].controller.use_delta = False
    with pytest.raises(UnsupportedTaskError, match="delta OSC_POSE"):
        build_task_context(env)


def test_numpy_context_batches_into_task_objective_without_renderer():
    env, _ = fake_env()
    ctx = {name: torch.as_tensor(value).unsqueeze(0)
           for name, value in build_task_context(env, {}).items()}
    actions = torch.zeros(1, 4, 7, requires_grad=True)
    cost = TaskGeometryObjective(n_action_steps=2)(actions, ctx)
    assert cost.shape == (1,) and torch.isfinite(cost).all()
    cost.sum().backward()
    assert torch.isfinite(actions.grad).all()


def test_on_tilted_site_uses_world_extent_without_axis_cancellation():
    env, domain = fake_env([["on", "book", "caddy_site"]])
    angle = np.pi / 4
    rotation = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    domain.object_sites_dict["caddy_site"].size = np.array([0.1, 0.2, 0.03])
    domain.sim.data.get_site_xmat = lambda name: rotation
    ctx = build_task_context(env, {})
    expected_z = 0.1 + (0.1 + 0.03) / np.sqrt(2) + 0.05
    assert ctx["place_pos"][2] == pytest.approx(expected_z)
