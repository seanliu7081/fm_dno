"""CPU checks for the time-modulated flow backbone; no vision dependencies."""

import unittest

import torch
from torch import nn

from oat.model.flow.mixed_dit import MixedAdaLNZeroTransformer


def make_model(**kwargs):
    config = dict(
        input_dim=7, output_dim=7, horizon=4, n_obs_steps=2, cond_dim=5,
        n_layer=2, n_head=2, n_emb=16, p_drop_emb=0.0, p_drop_attn=0.0,
    )
    config.update(kwargs)
    return MixedAdaLNZeroTransformer(**config)


def activate_model(model):
    """Open zero gates/output to check conditioning independently of training."""
    with torch.no_grad():
        for block in model.blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.2)
            chunks = block.adaLN_modulation[-1].bias.chunk(9)
            for index in (2, 5, 8):
                chunks[index].fill_(1.0)
        nn.init.normal_(model.final_layer.linear.weight, std=0.2)
        nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.2)


class MixedDiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(12)

    def test_zero_initialization_is_identity_inside_and_zero_velocity_outside(self):
        model = make_model(p_drop_attn=0.2).train()
        tokens = torch.randn(3, 4, 16)
        memory = torch.randn(3, 2, 16)
        time = torch.randn(3, 16)
        for block in model.blocks:
            torch.testing.assert_close(block(tokens, memory, time), tokens, atol=0, rtol=0)
        velocity = model(torch.randn(3, 4, 7), torch.tensor([0, 500, 1000]), torch.randn(3, 2, 5))
        torch.testing.assert_close(velocity, torch.zeros(3, 4, 7), atol=0, rtol=0)

    def test_scalar_time_and_variable_batches_match_explicit_batch_times(self):
        model = make_model().eval()
        activate_model(model)
        for batch_size in (1, 3):
            sample, cond = torch.randn(batch_size, 4, 7), torch.randn(batch_size, 2, 5)
            expected = model(sample, torch.full((batch_size,), 250.5), cond)
            for timestep in (250.5, torch.tensor(250.5), torch.tensor([250.5])):
                torch.testing.assert_close(model(sample, timestep, cond), expected)
            self.assertEqual(model(sample[:, :2], 0, cond[:, :1]).shape, (batch_size, 2, 7))

    def test_training_opens_gates_and_reaches_observation_path(self):
        model = make_model()
        sample, cond = torch.randn(3, 4, 7), torch.randn(3, 2, 5)
        timestep = torch.tensor([0.0, 250.0, 900.0])
        target = torch.randn_like(sample)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = model.cond_obs_emb.weight.detach().clone()
        losses = []
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(sample, timestep, cond) - target).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [param.grad for param in model.parameters() if param.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        self.assertFalse(torch.equal(model.cond_obs_emb.weight, before))
        self.assertGreater(model.cond_obs_emb.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].adaLN_modulation[-1].weight.abs().sum().item(), 0)

    def test_active_model_uses_action_observation_time_and_future_tokens(self):
        model = make_model().eval()
        activate_model(model)
        sample = torch.randn(2, 4, 7, requires_grad=True)
        cond = torch.randn(2, 2, 5, requires_grad=True)
        time = torch.tensor([100.0, 700.0], requires_grad=True)
        velocity = model(sample, time, cond)
        gradients = torch.autograd.grad(velocity.square().sum(), (sample, cond, time))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        changed_action = sample.detach().clone()
        changed_action[:, -1, 0] += 2.0
        # The first action can depend on the last action: no causal self-attention mask.
        changed = model(changed_action, time, cond)
        self.assertGreater((changed[:, 0] - velocity[:, 0]).abs().max().item(), 1e-7)
        for changed in (
            model(sample, time, cond + 3.0), model(sample, time + 137.0, cond),
        ):
            self.assertGreater((changed - velocity).abs().max().item(), 1e-7)

    def test_rejects_invalid_dimensions_and_input_shapes(self):
        for override in ({"n_emb": 15}, {"n_head": 3}, {"horizon": 0}, {"p_drop_attn": -0.1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                make_model(**override)
        model = make_model()
        sample, cond = torch.randn(2, 4, 7), torch.randn(2, 2, 5)
        cases = (
            (sample[:, :, :6], 0, cond), (torch.randn(2, 5, 7), 0, cond),
            (sample, 0, cond[:1]), (sample, 0, torch.randn(2, 3, 5)),
            (sample, torch.zeros(2, 1), cond), (sample, torch.zeros(3), cond),
        )
        for args in cases:
            with self.subTest(shapes=[getattr(arg, "shape", None) for arg in args]):
                with self.assertRaises(ValueError):
                    model(*args)


if __name__ == "__main__":
    unittest.main()
