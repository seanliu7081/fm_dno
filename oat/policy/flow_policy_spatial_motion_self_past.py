"""SMQ-DiT with ordered command history and depth-one self-generated past.

Training and offline evaluation are stateless. Rollout may assume that the full
returned prefix executes, or use update_history=False and record_executed_actions
with the actual commands. Missing history never contains repeated target actions.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from oat.model.heading.past_action_tokens import PastActionTokenEncoder
from oat.model.heading.segmented_heading_prior import heading_losses, segment_targets
from oat.policy.flow_policy_spatial_motion import SpatialMotionFlowPolicy


class SpatialMotionSelfPastFlowPolicy(SpatialMotionFlowPolicy):
    def __init__(self, *args, past_n=7, self_past_p=1.0,
                 self_past_warmup_steps=500, self_past_ramp_steps=2000, **kwargs):
        if not isinstance(past_n, int) or past_n < 3:
            raise ValueError('past_n must be an integer >= 3')
        if not math.isfinite(self_past_p) or not 0 <= self_past_p <= 1:
            raise ValueError('self_past_p must be in [0,1]')
        for name, value in (('self_past_warmup_steps', self_past_warmup_steps),
                            ('self_past_ramp_steps', self_past_ramp_steps)):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        super().__init__(*args, **kwargs)
        self.past_n = past_n
        self.self_past_p = float(self_past_p)
        self.self_past_warmup_steps = self_past_warmup_steps
        self.self_past_ramp_steps = self_past_ramp_steps
        self.history_encoder = PastActionTokenEncoder(self.action_dim, self.obs_feature_dim, past_n)
        # The original DiT has no condition-position tensor; only its length
        # contract changes. Existing observation and motion positions stay intact.
        if self.model.obs_pos_emb is not None:
            raise ValueError('History tokens require add_cond_pos_emb=False')
        self.model.cond_len += past_n + 2
        self.register_buffer('self_past_optimizer_step', torch.zeros((), dtype=torch.long))
        print(f'  self-past history: {past_n} actions + 2 differences; expanded cond_len={self.model.cond_len}; '
              f'warmup={self_past_warmup_steps}, ramp={self_past_ramp_steps} optimizer updates')
        self.reset()

    def get_policy_name(self):
        return 'spatial_motion_self_past_dit'

    def training_self_past_probability(self):
        step = int(self.self_past_optimizer_step.item())
        if step < self.self_past_warmup_steps:
            return 0.0
        if self.self_past_ramp_steps == 0:
            return self.self_past_p
        fraction = min(1.0, (step - self.self_past_warmup_steps) / self.self_past_ramp_steps)
        return self.self_past_p * fraction

    def _checked_history(self, past_action, past_mask, batch_size):
        if past_action.shape != (batch_size, self.past_n, self.action_dim):
            raise ValueError('past_action must have shape (B,past_n,action_dim)')
        if past_mask.shape != (batch_size, self.past_n) or past_mask.dtype != torch.bool:
            raise ValueError('past_mask must be boolean with shape (B,past_n)')
        return torch.where(past_mask[..., None], past_action.float(), 0.0)

    def encode_condition(self, obs_dict, past_action, past_mask):
        observations = self.obs_encoder({key: value for key, value in obs_dict.items()
                                         if key != '_p2n_context'})
        past_action = self._checked_history(past_action, past_mask, len(observations))
        normalized = self.normalizer['action'].normalize(past_action).float()
        history = self.history_encoder(normalized, past_mask)
        memory = torch.cat((observations, history.to(observations.dtype)), dim=1)
        prediction = self.motion_decoder(memory)
        return torch.cat((memory, prediction['motion_tokens']), dim=1), prediction

    @contextmanager
    def _sampling_mode(self):
        # Preserve heterogeneous module modes exactly, including the caller's
        # training flag. Do not mutate any rollout state or schedule here.
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            yield
        finally:
            for module, mode in modes:
                module.training = mode

    @torch.no_grad()
    def sample_actions(self, obs_dict, past_action, past_mask, noise=None):
        """Pure sampler, shared by self-past, held-out windows and rollout."""
        enabled = self.inference_bf16 and self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        # An inner no-grad pass must not cache detached BF16 weights for the
        # outer training forward, which still needs gradients through them.
        with self._sampling_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16,
                enabled=enabled, cache_enabled=False):
            cond, prediction = self.encode_condition(obs_dict, past_action, past_mask)
            if noise is None:
                noise = torch.randn(len(cond), self.horizon, self.action_dim,
                                    device=cond.device, dtype=torch.float32)
            if noise.shape != (len(cond), self.horizon, self.action_dim):
                raise ValueError('noise must match the action prediction shape')
            x, _ = self.make_source(noise, prediction)
            dt = 1.0 / self.num_inference_steps
            for i in range(self.num_inference_steps):
                t = torch.full((len(cond),), i * dt, device=cond.device)
                x = x + dt * self.model(x, self._scale_t(t), cond).float()
        return self.normalizer['action'].unnormalize(x)

    def _select_history(self, context, *, probability):
        past = context['past_action']
        mask = context['past_mask']
        self._checked_history(past, mask, len(past))
        used = torch.zeros(len(past), device=past.device, dtype=torch.bool)
        if probability <= 0:
            return past, mask, used
        valid = context['prev_window_valid']
        if valid.shape != (len(past),) or valid.dtype != torch.bool:
            raise ValueError('prev_window_valid must be boolean with shape (B,)')
        used = valid.clone()
        if probability < 1:
            used &= torch.rand(len(past), device=past.device) < probability
        indices = used.nonzero(as_tuple=True)[0]
        if not len(indices):
            return past, mask, used
        previous_obs = {key: value[indices] for key, value in context['prev_obs'].items()}
        previous_past = context['prev_past_action'][indices]
        previous_mask = context['prev_past_mask'][indices]
        previous_prediction = self.sample_actions(previous_obs, previous_past, previous_mask)
        executed = previous_prediction[:, :self.n_action_steps]
        combined = torch.cat((previous_past.float(), executed.float()), dim=1)
        combined_mask = torch.cat((previous_mask, torch.ones(
            len(indices), self.n_action_steps, device=mask.device, dtype=torch.bool)), dim=1)
        # P=7/S=8 selects previous_prediction[:,1:8], never its unexecuted tail.
        generated = combined[:, -self.past_n:].detach()
        generated_mask = combined_mask[:, -self.past_n:]
        selected = past.clone()
        selected_mask = mask.clone()
        selected[indices] = generated.to(selected.dtype)
        selected_mask[indices] = generated_mask
        return selected, selected_mask, used

    def loss_components(self, batch, noise=None, t=None):
        actions = batch['action']
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError('Actions must match horizon and action dimension')
        # Validation always measures depth-one generated history, independent
        # of the warmup curriculum; current images remain demonstration images.
        probability = self.training_self_past_probability() if self.training else 1.0
        past, mask, used = self._select_history(batch, probability=probability)
        cond, prediction = self.encode_condition(batch['obs'], past, mask)
        x1 = self.normalizer['action'].normalize(actions).float()
        noise = torch.randn_like(x1) if noise is None else noise
        if noise.shape != x1.shape:
            raise ValueError('Source noise must match actions')
        x0, active = self.make_source(noise, prediction)
        t = torch.rand(len(x1), device=x1.device) if t is None else t
        if t.shape != (len(x1),):
            raise ValueError('Flow time must have shape (B,)')
        tau = t[:, None, None]
        xt = (1 - tau) * x0 + tau * x1
        velocity = self.model(xt, self._scale_t(t), cond)
        flow_loss = (velocity.float() - (x1 - x0)).square().mean()
        target = segment_targets(actions, batch['action_mask'], self.heading_xy_rms,
                                 num_segments=self.num_motion_segments,
                                 threshold=self.min_target_confidence)
        aux = heading_losses(prediction, target)
        total = (flow_loss + self.heading_loss_weight * aux['direction_loss']
                 + self.validity_loss_weight * aux['validity_loss'])
        return {'loss': total, 'flow_loss': flow_loss, 'heading_loss': aux['direction_loss'],
                'validity_loss': aux['validity_loss'], 'prediction': prediction,
                'target': target, 'source': x0, 'source_active': active,
                'self_past_used': used, 'self_past_probability': probability}

    def _cache_metrics(self, result):
        super()._cache_metrics(result)
        used = result['self_past_used']
        count = used.new_tensor(used.numel(), dtype=torch.float32)
        self._training_metrics['self_past_fraction'] = (used.float().sum(), count)
        self._training_metrics['self_past_probability'] = (
            count * result['self_past_probability'], count)

    def get_optimizer(self, policy_lr, obs_enc_lr, weight_decay, betas,
                      heading_lr=1e-4, history_lr=None):
        groups = []
        modules = (('policy', self.model, policy_lr), ('encoder', self.obs_encoder, obs_enc_lr),
                   ('motion', self.motion_decoder, heading_lr),
                   ('history', self.history_encoder, policy_lr if history_lr is None else history_lr))
        for label, module, rate in modules:
            decay, no_decay = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                embedding = any(word in name for word in ('embedding', 'pos_emb', 'queries', 'register_tokens'))
                (decay if param.ndim >= 2 and not embedding else no_decay).append(param)
            for suffix, params, amount in (('decay', decay, weight_decay), ('no_decay', no_decay, 0.0)):
                if params:
                    groups.append({'params': params, 'lr': rate, 'weight_decay': amount,
                                   'name': f'{label}_{suffix}'})
        covered = [id(p) for group in groups for p in group['params']]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(covered) != len(set(covered)) or set(covered) != expected:
            raise RuntimeError('Optimizer must cover every trainable parameter exactly once')
        optimizer = torch.optim.AdamW(groups, betas=betas)
        def completed_update(optimizer, args, kwargs):
            with torch.no_grad():
                self.self_past_optimizer_step.add_(1)
        optimizer.register_step_post_hook(completed_update)
        return optimizer

    def reset(self):
        self._past_buffer = None
        self._past_mask = None

    def record_executed_actions(self, actions):
        """Append actual raw commands; call after predict_action(update_history=False)."""
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError('Executed actions must have shape (B,N,action_dim)')
        if actions.shape[1] < 1:
            return
        self._ensure_history(len(actions), actions.device)
        valid = torch.ones(actions.shape[:2], device=actions.device, dtype=torch.bool)
        self._past_buffer = torch.cat((self._past_buffer, actions.detach().float()), dim=1)[:, -self.past_n:].clone()
        self._past_mask = torch.cat((self._past_mask, valid), dim=1)[:, -self.past_n:].clone()

    def _ensure_history(self, batch_size, device):
        if (self._past_buffer is None or self._past_buffer.shape[0] != batch_size
                or self._past_buffer.device != device):
            self._past_buffer = torch.zeros(batch_size, self.past_n, self.action_dim, device=device)
            self._past_mask = torch.zeros(batch_size, self.past_n, device=device, dtype=torch.bool)

    @torch.no_grad()
    def predict_action(self, obs_dict, *, past_action=None, past_mask=None,
                       noise=None, update_history=True):
        """Context-bearing minibatches never read or mutate online history.

        With ordinary rollout observations, update_history=True assumes the
        returned eight-action prefix executes fully, as in Past2Next. For partial
        or modified execution, disable it and append actual commands explicitly.
        """
        context = obs_dict.get('_p2n_context')
        explicit = past_action is not None or past_mask is not None
        if explicit and (past_action is None or past_mask is None):
            raise ValueError('Supply both past_action and past_mask')
        stateless = explicit or context is not None
        if not explicit:
            if context is not None:
                past_action, past_mask, _ = self._select_history(context, probability=1.0)
            else:
                sample = obs_dict[self.obs_encoder.rgb_ports[0]]
                self._ensure_history(len(sample), sample.device)
                past_action, past_mask = self._past_buffer, self._past_mask
        action_pred = self.sample_actions(obs_dict, past_action, past_mask, noise=noise)
        action = action_pred[:, :self.n_action_steps]
        if not stateless and update_history:
            self.record_executed_actions(action)
        return {'action': action, 'action_pred': action_pred}
