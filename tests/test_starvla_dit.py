"""CPU checks for the GR00T-style action flow backbone."""

import unittest

import torch

from oat.model.flow.starvla_dit import StarVLAFlowTransformer


class StarVLAFlowTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(13)

    def model(self, **kwargs):
        config = dict(
            input_dim=7,
            output_dim=7,
            horizon=4,
            n_obs_steps=2,
            cond_dim=11,
            n_layer=4,
            n_head=2,
            n_emb=16,
            p_drop_emb=0.0,
            p_drop_attn=0.0,
            num_register_tokens=3,
        )
        config.update(kwargs)
        return StarVLAFlowTransformer(**config)

    def test_shape_and_gradients_through_actions_condition_and_continuous_time(self):
        model = self.model()
        actions = torch.randn(2, 4, 7, requires_grad=True)
        cond = torch.randn(2, 2, 11, requires_grad=True)
        time = torch.tensor([120.25, 850.75], requires_grad=True)

        prediction = model(actions, time, cond)
        self.assertEqual(prediction.shape, actions.shape)
        prediction.square().mean().backward()

        for name, value in (("actions", actions), ("condition", cond), ("time", time)):
            with self.subTest(input=name):
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.abs().sum().item(), 0.0)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_condition_and_fractional_time_change_prediction(self):
        model = self.model().eval()
        actions = torch.randn(2, 4, 7)
        cond = torch.randn(2, 2, 11)
        with torch.no_grad():
            baseline = model(actions, 400.1, cond)
            changed_condition = model(actions, 400.1, cond + 0.5)
            changed_time = model(actions, 400.9, cond)
        self.assertFalse(torch.allclose(baseline, changed_condition))
        self.assertFalse(torch.allclose(baseline, changed_time))

    def test_scalar_time_matches_batched_time(self):
        model = self.model().eval()
        actions = torch.randn(2, 4, 7)
        cond = torch.randn(2, 2, 11)
        with torch.no_grad():
            baseline = model(actions, torch.full((2,), 100.25), cond)
            for time in (100.25, torch.tensor(100.25), torch.tensor([100.25])):
                torch.testing.assert_close(model(actions, time, cond), baseline)

    def test_observation_frame_order_changes_prediction(self):
        model = self.model().eval()
        actions = torch.randn(2, 4, 7)
        cond = torch.randn(2, 2, 11)
        with torch.no_grad():
            baseline = model(actions, 300.0, cond)
            swapped_frames = model(actions, 300.0, cond.flip(1))
        self.assertFalse(torch.allclose(baseline, swapped_frames))

    def test_last_action_influences_first_action_through_self_attention(self):
        # Two blocks include one cross-attention and one self-attention block.
        # A first-position output must depend on future input positions.
        model = self.model(n_layer=2).eval()
        actions = torch.randn(1, 4, 7, requires_grad=True)
        cond = torch.randn(1, 2, 11)
        prediction = model(actions, 500.0, cond)
        gradient = torch.autograd.grad(prediction[0, 0].square().sum(), actions)[0]
        self.assertGreater(gradient[0, -1].abs().sum().item(), 0.0)

    def test_odd_depth_shorter_horizon_and_no_registers(self):
        model = self.model(n_layer=3, num_register_tokens=0, output_dim=5)
        output = model(torch.randn(2, 3, 7), 0, torch.randn(2, 2, 11))
        self.assertEqual(output.shape, (2, 3, 5))
        self.assertTrue(torch.isfinite(output).all())

    def test_invalid_architecture_and_input_shapes(self):
        for kwargs in (
            {"n_layer": 1}, {"n_emb": 15}, {"n_head": 3},
            {"cond_dim": 0}, {"num_register_tokens": -1},
        ):
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs)
        model = self.model()
        with self.assertRaisesRegex(ValueError, "timestep"):
            model(torch.randn(2, 4, 7), torch.zeros(2, 1), torch.randn(2, 2, 11))
        with self.assertRaisesRegex(ValueError, "cond"):
            model(torch.randn(2, 4, 7), 0, torch.randn(2, 2, 10))


if __name__ == "__main__":
    unittest.main()
