"""CPU integration checks for learned observation-only source canonicalization.

Run from fm_dno with `python -m unittest discover -s tests -v`.
Tiny encoders exercise the real policy without simulator or vision dependencies.
"""

import contextlib
import copy
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from oat.model.common.normalizer import LinearNormalizer
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.perception.heading_predictor import ObservationHeadingPredictor
from oat.policy.flow_policy_learned_canon import LearnedCanonicalPhaseFlowPolicy
from oat.symmetry.so2_chunk import SO2ChunkSpec, chunk_heading, wrap_angle


class TinyObservationEncoder(BaseObservationEncoder):
    """Trainable image/state encoder with stochastic training behavior."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.projection = nn.Linear(4, 4)
        self.dropout = nn.Dropout(dropout)

    def modalities(self):
        return ["state", "rgb"]

    def output_feature_dim(self):
        return 4

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, obs_dict):
        state = obs_dict["robot0_eef_pos"]
        if "robot0_eef_pos" in self.normalizer.params_dict:
            state = self.normalizer["robot0_eef_pos"].normalize(state)
        image_mean = obs_dict["camera"].mean(dim=(-3, -2, -1))
        features = torch.cat((state, image_mean.unsqueeze(-1)), dim=-1)
        return self.dropout(self.projection(features))


SHAPE_META = {
    "action": {"shape": [7]},
    "obs": {
        "robot0_eef_pos": {"shape": [3], "type": "state"},
        "camera": {"shape": [3, 4, 4], "type": "rgb"},
    },
}
ACTION_SPEC = SO2ChunkSpec.translation_only()


def observations(batch_size=4):
    # The robot is stationary. A visual reference must still be usable.
    return {
        "robot0_eef_pos": torch.zeros(batch_size, 2, 3),
        "camera": torch.linspace(0.0, 1.0, batch_size)[:, None, None, None, None]
        .expand(batch_size, 2, 3, 4, 4)
        .clone(),
    }


def normalizer():
    result = LinearNormalizer()
    result.fit({
        "action": torch.linspace(-1.0, 1.0, 64)[:, None].expand(64, 7).clone(),
        "robot0_eef_pos": torch.linspace(-1.0, 1.0, 64)[:, None].expand(64, 3).clone(),
    })
    return result


def heading_predictor(constant=True, valid_logit=8.0):
    reference = ObservationHeadingPredictor(TinyObservationEncoder(), hidden_dim=8)
    reference.set_normalizer(normalizer(), ACTION_SPEC)
    if constant:
        with torch.no_grad():
            reference.head[-1].weight.zero_()
            reference.head[-1].bias.copy_(torch.tensor([0.0, 1.0, valid_logit]))
    return reference


def make_policy(reference_use="source", reference_model=None, initialize=True, **kwargs):
    if reference_model is None:
        reference_model = heading_predictor()
    with contextlib.ExitStack() as stack:
        if "reference_checkpoint" not in kwargs:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            artifact = Path(directory) / "reference.pt"
            reference_model.save_checkpoint(artifact, action_spec=ACTION_SPEC, horizon=4)
            kwargs["reference_checkpoint"] = str(artifact)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = LearnedCanonicalPhaseFlowPolicy(
                shape_meta=SHAPE_META,
                obs_encoder=TinyObservationEncoder(dropout=0.0),
                reference_model=reference_model,
                reference_use=reference_use,
                horizon=4,
                n_action_steps=2,
                n_obs_steps=2,
                embed_dim=8,
                n_layers=1,
                n_heads=2,
                dropout=0.0,
                num_inference_steps=2,
                action_spec={"block_weights": [1.0, 0.0]},
                coupling={"mode": "iid"},
                kappa=kwargs.pop("kappa", math.inf),
                **kwargs,
            )
            if initialize:
                policy.set_normalizer(normalizer())
                policy.prepare_for_training()
        return policy


class LearnedCanonicalPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(12)

    def assert_aligned(self, source, angle=math.pi / 2):
        theta, _ = chunk_heading(source, ACTION_SPEC)
        torch.testing.assert_close(
            wrap_angle(theta - angle), torch.zeros_like(theta), atol=2e-6, rtol=0
        )

    def test_stationary_robot_can_use_visual_reference(self):
        policy = make_policy(min_ref_speed=100.0)
        obs = observations()
        theta, valid = policy.reference_heading(obs)
        self.assertTrue(valid.all())
        torch.testing.assert_close(theta, torch.full_like(theta, math.pi / 2))
        self.assert_aligned(policy.sample_prior_from_obs(obs))

    def test_source_condition_and_both_ablation_modes(self):
        obs = observations()
        noise = torch.randn(4, 4, 7)
        for mode in ("source", "condition", "both"):
            with self.subTest(mode=mode):
                policy = make_policy(reference_use=mode)
                cond = policy.encode_obs(obs)
                self.assertEqual(cond.shape, (4, 2, 4 if mode == "source" else 7))
                result = policy.canonicalize_source(noise, obs)
                if mode == "condition":
                    torch.testing.assert_close(result, noise, atol=0, rtol=0)
                else:
                    self.assert_aligned(result)
                    torch.testing.assert_close(result.norm(dim=(1, 2)), noise.norm(dim=(1, 2)))
                if mode != "source":
                    expected = torch.tensor([0.0, 1.0, torch.sigmoid(torch.tensor(8.0))])
                    torch.testing.assert_close(cond[..., -3:], expected.expand(4, 2, 3))

    def test_condition_only_prior_is_same_iid_draw(self):
        policy = make_policy(reference_use="condition", kappa=4.0)
        expected = policy.prior_noise_scale * torch.randn(
            4, 4, 7, generator=torch.Generator().manual_seed(42)
        )
        actual = policy.sample_prior_from_obs(
            observations(), generator=torch.Generator().manual_seed(42)
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_low_confidence_reference_leaves_source_unchanged(self):
        policy = make_policy(reference_model=heading_predictor(valid_logit=-8.0))
        obs = observations()
        _, valid = policy.reference_heading(obs)
        self.assertFalse(valid.any())
        noise = torch.randn(4, 4, 7)
        torch.testing.assert_close(policy.canonicalize_source(noise, obs), noise, atol=0, rtol=0)

    def test_finite_dither_preserves_angle_diversity_and_norm(self):
        policy = make_policy(kappa=4.0)
        obs = observations(batch_size=512)
        noise = torch.randn(512, 4, 7)
        source = policy.canonicalize_source(noise, obs)
        theta, _ = chunk_heading(source, ACTION_SPEC)
        residual = wrap_angle(theta - math.pi / 2)
        self.assertGreater(float(residual.std()), 0.1)
        self.assertGreater(float(residual.cos().mean()), 0.6)
        torch.testing.assert_close(source.norm(dim=(1, 2)), noise.norm(dim=(1, 2)))

    def test_entire_reference_stays_frozen_and_deterministic(self):
        reference = heading_predictor(constant=False)
        policy = make_policy(reference_model=reference, reference_use="both")
        obs = observations()
        policy.eval()
        expected = {key: value.clone() for key, value in reference(obs).items()}
        policy.train()
        self.assertFalse(any(module.training for module in reference.modules()))
        self.assertFalse(any(param.requires_grad for param in reference.parameters()))
        for _ in range(3):
            for key, value in reference(obs).items():
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
        optimizer = policy.get_optimizer(1e-3, 1e-3, 1e-3, (0.9, 0.999))
        optimized = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertTrue(optimized.isdisjoint({id(param) for param in reference.parameters()}))
        self.assertTrue({id(param) for param in policy.model.parameters()
                         if param.requires_grad}.issubset(optimized))
        loss = policy({"obs": obs, "action": torch.randn(4, 4, 7)})
        loss.backward()
        trainable_encoder = [
            param for param in policy.obs_encoder.parameters() if param.requires_grad
        ]
        self.assertTrue(trainable_encoder)
        self.assertTrue(any(param.grad is not None and param.grad.abs().sum() > 0
                            for param in trainable_encoder))
        self.assertTrue(all(param.grad is None for param in reference.parameters()))

    def test_training_source_does_not_depend_on_target_actions(self):
        policy = make_policy(reference_use="both", kappa=4.0)
        obs = observations()
        captured = []
        original = policy._canonicalize_with_heading

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            captured.append(result.detach().clone())
            return result

        with mock.patch.object(policy, "_canonicalize_with_heading", side_effect=capture):
            for target in (torch.ones(4, 4, 7), -torch.ones(4, 4, 7)):
                torch.manual_seed(31)
                policy({"obs": obs, "action": target})
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0], captured[1], atol=0, rtol=0)

    def test_predict_from_noise_retains_noise_gradients(self):
        policy = make_policy(reference_use="both")
        noise = torch.randn(4, 4, 7, requires_grad=True)
        result = policy.predict_action_from_noise(observations(), noise, n_steps=2)
        self.assertEqual(result["action"].shape, (4, 2, 7))
        self.assertEqual(result["action_pred"].shape, (4, 4, 7))
        result["action_pred"].square().mean().backward()
        self.assertIsNotNone(noise.grad)
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertGreater(float(noise.grad.abs().sum()), 0.0)

    def test_reference_is_computed_once_per_training_or_prediction(self):
        reference = heading_predictor()
        policy = make_policy(reference_use="both", reference_model=reference)
        obs = observations()
        operations = (
            lambda: policy({"obs": obs, "action": torch.randn(4, 4, 7)}),
            lambda: policy.predict_action(obs),
            lambda: policy.predict_action_from_noise(obs, torch.randn(4, 4, 7)),
        )
        for operation in operations:
            with mock.patch.object(reference, "forward", wraps=reference.forward) as forward:
                operation()
                self.assertEqual(forward.call_count, 1)

    def test_policy_checkpoint_embeds_reference_without_external_artifact(self):
        obs = observations()
        noise = torch.randn(4, 4, 7)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "reference.pt"
            fitted = heading_predictor(constant=False)
            fitted.save_checkpoint(artifact, action_spec=ACTION_SPEC, horizon=4)
            policy = make_policy(reference_checkpoint=str(artifact), reference_use="both")
            policy.eval()
            expected = policy.predict_action_from_noise(obs, noise)["action_pred"].detach()
            checkpoint = copy.deepcopy(policy.state_dict())
            artifact.unlink()
            restored = make_policy(
                reference_checkpoint=str(artifact), reference_use="both", initialize=False
            )
            restored.load_state_dict(checkpoint, strict=True)
            restored.eval()
            actual = restored.predict_action_from_noise(obs, noise)["action_pred"]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_reference_normalizers_survive_policy_normalizer_changes(self):
        reference = heading_predictor(constant=False)
        policy = make_policy(reference_model=reference)
        obs = observations()
        expected = reference(obs)["theta"].clone()
        saved = {key: value.clone() for key, value in reference.state_dict().items()}
        replacement = LinearNormalizer()
        replacement.fit({
            "action": torch.linspace(-2.0, 2.0, 64)[:, None].expand(64, 7).clone(),
            "robot0_eef_pos": torch.linspace(20.0, 40.0, 64)[:, None].expand(64, 3).clone(),
        })
        with contextlib.redirect_stdout(io.StringIO()):
            policy.set_normalizer(replacement)
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
        torch.testing.assert_close(reference(obs)["theta"], expected, atol=0, rtol=0)
        self.assertFalse(any(param.requires_grad for param in reference.parameters()))

    def test_ema_reference_matches_after_separate_lazy_initialization(self):
        from oat.model.diffusion.ema_model import EMAModel

        obs = observations()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "reference.pt"
            fitted = heading_predictor(constant=False)
            fitted.save_checkpoint(artifact, action_spec=ACTION_SPEC, horizon=4)
            policy = make_policy(reference_checkpoint=str(artifact), initialize=False)
            averaged = copy.deepcopy(policy)
            optimizer = policy.get_optimizer(1e-3, 1e-3, 1e-3, (0.9, 0.999))
            with contextlib.redirect_stdout(io.StringIO()):
                policy.set_normalizer(normalizer())
                averaged.set_normalizer(normalizer())
                policy.prepare_for_training()
                averaged.prepare_for_training()
            ema = EMAModel(averaged)
            policy.train()
            averaged.train()
            optimizer.zero_grad()
            policy({"obs": obs, "action": torch.randn(4, 4, 7)}).backward()
            optimizer.step()
            ema.step(policy)
            artifact.unlink()
            self.assertTrue(bool(averaged.obs_encoder.reference_loaded))
            self.assertFalse(any(module.training for module in averaged.reference_model.modules()))
            for key, value in policy.reference_model.state_dict().items():
                torch.testing.assert_close(
                    value, averaged.reference_model.state_dict()[key], atol=0, rtol=0
                )
            torch.testing.assert_close(
                policy.reference_heading(obs)[0], averaged.reference_heading(obs)[0], atol=0, rtol=0
            )

    def test_resume_restores_reference_before_training_preparation(self):
        obs = observations()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "reference.pt"
            fitted = heading_predictor(constant=False)
            fitted.save_checkpoint(artifact, action_spec=ACTION_SPEC, horizon=4)
            policy = make_policy(reference_checkpoint=str(artifact))
            expected = policy.reference_heading(obs)[0].clone()
            checkpoint = copy.deepcopy(policy.state_dict())
            artifact.unlink()
            resumed = make_policy(reference_checkpoint=str(artifact), initialize=False)
            replacement = normalizer()
            with torch.no_grad():
                replacement.params_dict["robot0_eef_pos"]["offset"].add_(10.0)
            # Match TrainPolicyWorkspace.run: fit current data, load resumed state,
            # then prepare the reference before accelerator/DDP wraps the model.
            with contextlib.redirect_stdout(io.StringIO()):
                resumed.set_normalizer(replacement)
            resumed.load_state_dict(checkpoint, strict=True)
            resumed.prepare_for_training()
            torch.testing.assert_close(resumed.reference_heading(obs)[0], expected, atol=0, rtol=0)
            self.assertTrue(bool(resumed.obs_encoder.reference_loaded))
            self.assertFalse(any(param.requires_grad for param in resumed.reference_model.parameters()))

    def test_missing_reference_artifact_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = str(Path(directory) / "missing.pt")
            policy = make_policy(reference_checkpoint=artifact, initialize=False)
            with contextlib.redirect_stdout(io.StringIO()):
                policy.set_normalizer(normalizer())
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                policy.prepare_for_training()


if __name__ == "__main__":
    unittest.main()
