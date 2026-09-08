"""Task-guided initial-noise optimization for frozen E/F flow policies.

This implementation is independent of OrbitDNO. It optimizes the noise *after*
the policy's own observation-dependent source construction, with observations and
all sampler randomness fixed throughout a solve. Proxy-loss improvement is not a
claim of improved environment success.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np
import torch

from oat.policy.base_policy import BasePolicy


MODES = ('baseline', 'best_of_k', 'dno', 'amortized', 'amortized_dno')


@dataclass
class TaskDNOConfig:
    mode: str = 'dno'
    num_candidates: int = 4
    num_grad_steps: int = 3
    lr: float = 0.05
    trust_weight: float = 0.01
    trust_radius: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if self.num_candidates < 1 or self.num_grad_steps < 0:
            raise ValueError('num_candidates must be positive and num_grad_steps nonnegative')
        if not all(math.isfinite(v) and v >= 0 for v in
                   (self.lr, self.trust_weight, self.trust_radius, self.max_grad_norm)):
            raise ValueError('optimization parameters must be finite and nonnegative')
        if self.mode in ('dno', 'amortized_dno') and (self.num_grad_steps < 1 or self.lr <= 0):
            raise ValueError('DNO requires positive num_grad_steps and lr')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')


class TaskNoisePolicy(BasePolicy):
    def __init__(
        self, base_policy, objective: Optional[Callable] = None, *,
        mode='dno', num_candidates=4, num_grad_steps=3, lr=0.05,
        trust_weight=0.01, trust_radius=0.5, max_grad_norm=1.0,
        seed=42, initializer=None, record_training_data=False,
    ):
        super().__init__()
        for method in ('_prepare_observation', '_sample_prior_with_heading', 'sample_chunk'):
            if not callable(getattr(base_policy, method, None)):
                raise TypeError(f'base_policy must provide {method}; use an E/F checkpoint')
        if getattr(base_policy, 'reference_use', None) not in ('condition', 'both'):
            raise ValueError('TaskNoisePolicy requires E (condition) or F (both)')
        self.config = TaskDNOConfig(mode, num_candidates, num_grad_steps, lr,
                                    trust_weight, trust_radius, max_grad_norm, int(seed))
        if mode != 'baseline' and objective is None:
            raise ValueError('An explicit task objective is required; no geometry-free fallback')
        if mode in ('amortized', 'amortized_dno') and initializer is None:
            raise ValueError('Learned modes require a trained noise initializer')
        self.policy = base_policy
        self.objective = objective
        self.initializer = initializer
        self.record_training_data = bool(record_training_data)
        self.policy.requires_grad_(False).eval()
        if isinstance(self.initializer, torch.nn.Module):
            self.initializer.requires_grad_(False).eval()
        self.info_log = []
        self.last_info = {}
        self._call_index = 0

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    @property
    def n_obs_steps(self):
        return self.policy.n_obs_steps

    @property
    def n_action_steps(self):
        return self.policy.n_action_steps

    def train(self, mode=True):
        super().train(mode)
        self.policy.eval()
        if isinstance(self.initializer, torch.nn.Module):
            self.initializer.eval()
        return self

    def get_policy_name(self):
        return f'task_{self.config.mode}_{self.policy.get_policy_name()}'

    def get_observation_ports(self):
        return self.policy.get_observation_ports()

    def get_observation_modalities(self):
        return self.policy.get_observation_modalities()

    def get_observation_encoder(self):
        return self.policy.get_observation_encoder()

    def set_normalizer(self, normalizer):
        raise RuntimeError('The deployed base policy retains its checkpoint normalizer')

    def reset(self, seed=None):
        if seed is not None:
            if int(seed) < 0:
                raise ValueError('seed must be nonnegative')
            self.config.seed = int(seed)
        self._call_index = 0
        self.policy.reset()

    def clear_log(self):
        self.info_log.clear()
        self.last_info = {}

    def summarize(self):
        if not self.info_log:
            return {}
        result = {f'dno/{key}': float(np.mean([row[key] for row in self.info_log]))
                  for key in self.info_log[0] if isinstance(self.info_log[0][key], (int, float))}
        times = [row['wall_time'] for row in self.info_log]
        result.update({f'dno/wall_time_p{q}': float(np.percentile(times, q)) for q in (50, 95)})
        return result

    def configuration(self):
        return asdict(self.config)

    def _draw_source(self, theta, valid, cond, candidate_index):
        device = cond.device
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == 'cuda' else []
        # Each candidate owns its RNG seed. In particular, candidate zero is
        # identical across baseline / best-of-K / DNO, even for F's von Mises draw.
        seed = (self.config.seed + self._call_index * 1000003 + candidate_index * 10007) % (2**63 - 1)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            return self.policy._sample_prior_with_heading(
                theta, valid, batch_size=cond.shape[0], device=device, dtype=cond.dtype,
            ).detach()

    @staticmethod
    def _batch_context(context, device, dtype, batch_size):
        result = {}
        for key, value in (context or {}).items():
            if torch.is_tensor(value) or isinstance(value, (np.ndarray, int, float, bool)):
                value = torch.as_tensor(value, device=device)
                if value.is_floating_point():
                    value = value.to(dtype=dtype)
            result[key] = value
        # Contexts from a vector runner have leading B, including translation_gain.
        for key in ('eef_pos', 'goal_pos'):
            if key in result and result[key].shape != (batch_size, 3):
                raise ValueError(f'{key} must have shape (B,3), got {result[key].shape}')
        return result

    @staticmethod
    def _repeat_context(context, repeats, batch_size):
        return {key: value.repeat_interleave(repeats, dim=0)
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size
                else value for key, value in context.items()}

    @staticmethod
    def _project(z, anchor, radius):
        delta = z - anchor
        rms = delta.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-12)
        return anchor + delta * (radius / rms).clamp(max=1)

    def predict_action(self, obs_dict, task_context=None, **kwargs):
        if kwargs:
            raise TypeError(f'Unknown TaskNoisePolicy arguments: {sorted(kwargs)}')
        needs_grad = self.config.mode in ('dno', 'amortized_dno')
        if needs_grad and torch.is_inference_mode_enabled():
            raise RuntimeError('Task DNO needs autograd; call outside torch.inference_mode()')
        start = time.perf_counter()
        with torch.no_grad():
            cond, theta, valid = self.policy._prepare_observation(obs_dict)
            cond = cond.detach()
        B = cond.shape[0]
        ctx = self._batch_context(task_context, cond.device, cond.dtype, B)
        K = self.config.num_candidates if self.config.mode in ('best_of_k', 'dno', 'amortized_dno') else 1
        with torch.no_grad():
            source = torch.stack([self._draw_source(theta, valid, cond, i) for i in range(K)], dim=1)
        self._call_index += 1
        anchor = source.flatten(0, 1)
        cond_k = cond.repeat_interleave(K, dim=0)
        ctx_k = self._repeat_context(ctx, K, B)
        counts = {'sampler_calls': 0, 'candidate_evaluations': 0, 'backward_steps': 0}

        def evaluate(z, conditions, context):
            x = self.policy.sample_chunk(conditions, z)
            actions = self.policy.normalizer['action'].unnormalize(x)
            counts['sampler_calls'] += 1
            counts['candidate_evaluations'] += z.shape[0] / B
            if self.objective is None:
                loss = actions.new_zeros(actions.shape[0])
            else:
                loss = self.objective(actions, context)
                if loss.shape != (actions.shape[0],):
                    raise ValueError('Task objective must return one scalar per candidate')
            return actions, loss

        with torch.no_grad():
            original_actions, original_loss = evaluate(anchor, cond_k, ctx_k)
            if not torch.isfinite(original_loss).all() or not torch.isfinite(original_actions).all():
                raise RuntimeError('The initial action / task objective is nonfinite')
            best_z = anchor.clone()
            best_actions = original_actions.clone()
            best_loss = original_loss.clone()
            z_start = anchor.clone()
            if self.config.mode in ('amortized', 'amortized_dno'):
                z_start = self.initializer(cond_k, anchor, ctx_k)
                if z_start.shape != anchor.shape or not torch.isfinite(z_start).all():
                    raise RuntimeError('Initializer must return finite noise with shape (B,H,A)')
                z_start = self._project(z_start, anchor, self.config.trust_radius * self.policy.prior_noise_scale)
                actions, loss = evaluate(z_start, cond_k, ctx_k)
                improved = torch.isfinite(loss) & torch.isfinite(actions).flatten(1).all(-1) \
                    & (loss < best_loss)
                # Keep the raw source as a fallback, including for learned modes.
                best_z[improved], best_actions[improved], best_loss[improved] = \
                    z_start[improved], actions[improved], loss[improved]

        def consider(z, actions, loss):
            nonlocal best_z, best_actions, best_loss
            with torch.no_grad():
                improved = torch.isfinite(loss) & torch.isfinite(actions).flatten(1).all(-1) & (loss < best_loss)
                best_z[improved] = z.detach()[improved]
                best_actions[improved] = actions.detach()[improved]
                best_loss[improved] = loss.detach()[improved]

        if needs_grad:
            with torch.enable_grad():
                z = z_start.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([z], lr=self.config.lr)
                scale = max(float(self.policy.prior_noise_scale), 1e-8)
                for _ in range(self.config.num_grad_steps):
                    actions, loss = evaluate(z, cond_k, ctx_k)
                    consider(z, actions, loss)
                    trust = ((z - anchor) / scale).square().mean(dim=(-1, -2))
                    total = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss)) \
                        + self.config.trust_weight * trust
                    grad, = torch.autograd.grad(total.sum(), z)
                    counts['backward_steps'] += 1
                    finite = torch.isfinite(grad).flatten(1).all(-1)
                    grad = torch.where(finite[:, None, None], grad, torch.zeros_like(grad))
                    if self.config.max_grad_norm > 0:
                        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                        grad = grad * (self.config.max_grad_norm / norm).clamp(max=1)[:, None, None]
                    optimizer.zero_grad(set_to_none=True)
                    z.grad = grad
                    optimizer.step()
                    with torch.no_grad():
                        z.copy_(self._project(z, anchor, self.config.trust_radius * scale))
                with torch.no_grad():
                    actions, loss = evaluate(z, cond_k, ctx_k)
                consider(z, actions, loss)

        with torch.no_grad():
            indices = best_loss.reshape(B, K).argmin(dim=1)
            flat_indices = torch.arange(B, device=cond.device) * K + indices
            final_z = best_z[flat_indices]
            final_actions = best_actions[flat_indices]
            final_loss = best_loss[flat_indices]
            teacher_base = anchor[flat_indices]
            teacher_loss = original_loss[flat_indices]
            initial_loss = original_loss.reshape(B, K)[:, 0]
            info = {
                **counts,
                'nfe': counts['candidate_evaluations'] * self.policy.num_inference_steps,
                'initial_objective': float(initial_loss.mean()),
                'final_objective': float(final_loss.mean()),
                'objective_improvement': float((initial_loss - final_loss).mean()),
                'noise_rms_shift': float((final_z - teacher_base).square().mean((-1, -2)).sqrt().mean()),
                'wall_time': time.perf_counter() - start,
            }
            self.last_info = info
            self.info_log.append(info)
            out = {
                'action': final_actions[:, :self.n_action_steps], 'action_pred': final_actions,
                'z': final_z.detach(), 'info': info,
            }
            if self.record_training_data:
                from oat.dno.noise_initializer import build_context_features
                out.update({
                    'cond': cond.detach(), 'base_noise': teacher_base.detach(),
                    'optimized_noise': final_z.detach(),
                    'context_features': build_context_features(ctx).to(cond),
                    'teacher_improved': (final_loss < teacher_loss - 1e-8).detach(),
                    'initial_objective': teacher_loss.detach(), 'final_objective': final_loss.detach(),
                })
            return out
