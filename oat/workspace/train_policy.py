if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import random
import warnings
import numpy as np
from collections.abc import Mapping
import hydra
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import copy
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import (
    set_seed as accelerate_set_seed, DistributedDataParallelKwargs)
from typing import Union

from oat.workspace.base_workspace import BaseWorkspace
from oat.dataset.base_dataset import BaseDataset
from oat.env_runner.base_runner import BaseRunner
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.json_logger import JsonLogger
from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply, maybe_to_device
from oat.model.common.lr_scheduler import get_scheduler
from oat.model.common.misc import detect_bf16_support
from oat.policy.base_policy import BasePolicy

register_new_resolvers()


def _configure_models_from_dataset(dataset, *models):
    """Optional train-dataset initialization; resumed buffers are loaded later."""
    for model in models:
        configure = getattr(model, "configure_from_dataset", None)
        if callable(configure):
            configure(dataset)



def _epoch_events(epoch, training, *, lazy_eval, save_rollout_checkpoint=False,
                  save_final_checkpoint=False):
    """Keep legacy cadence by default; optionally count completed rollout epochs."""
    offset = training.get('rollout_epoch_offset', 0)
    if offset not in (0, 1):
        raise ValueError("rollout_epoch_offset must be 0 or 1")
    rollout_every = int(training.rollout_every)
    checkpoint_every = int(training.checkpoint_every)
    if rollout_every < 1 or checkpoint_every < 1:
        raise ValueError("Rollout and checkpoint intervals must be positive")
    rollout_due = not lazy_eval and (epoch + offset) % rollout_every == 0
    checkpoint_due = (
        epoch % checkpoint_every == 0
        or (save_rollout_checkpoint and rollout_due)
        or (save_final_checkpoint and epoch + 1 == training.num_epochs)
    )
    return rollout_due, checkpoint_due


def _clip_grad_norm(accelerator, model, optimizer, max_norm):
    """Honor optional independent clipping groups without repeated AMP unscale."""
    if max_norm is None:
        return None
    policy = accelerator.unwrap_model(model)
    get_groups = getattr(policy, "get_gradient_clip_groups", None)
    if not callable(get_groups):
        return accelerator.clip_grad_norm_(model.parameters(), max_norm)

    groups = get_groups()
    if not isinstance(groups, Mapping):
        raise ValueError("Gradient clip groups must be a mapping of names to parameters")
    groups = {name: list(parameters) for name, parameters in groups.items()}
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    seen = set()
    for name, parameters in groups.items():
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                raise ValueError(f"Gradient clip parameter appears more than once: {name}")
            if identity not in expected:
                raise ValueError(f"Gradient clip group contains a non-trainable or foreign parameter: {name}")
            seen.add(identity)
    if seen != expected:
        raise ValueError("Gradient clip groups must cover every trainable model parameter")

    # Accelerator.clip_grad_norm_ unscales internally. Calling it for each group
    # would unscale the same FP16 optimizer twice, so unscale once before clipping.
    accelerator.unscale_gradients(optimizer)
    return {
        name: torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=True)
        for name, parameters in groups.items()
    }


class TrainPolicyWorkspace(BaseWorkspace):
    # Plain metadata works with BaseWorkspace and inference-only restoration,
    # which intentionally does not instantiate an EMA controller or scheduler.
    include_keys = [
        'global_step', 'epoch', 'resume_state_version', 'next_epoch',
        'next_global_step', 'optimizer_step', 'ema_state',
        'lr_scheduler_state', 'lr_scheduler_config', 'rng_state',
    ]

    def __init__(self, cfg: OmegaConf, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir=output_dir)

        """
        Lazy instantiation allows deferring model, optimizer, and ema creation
        until after the seed has been set and the accelerator device has been chosen.
        1. If lazy_instantiation is False, model, optimizer, and ema are created immediately.
           This is useful for checkpoint loading, where we need to create the model
           before loading the state dict.
        2. If lazy_instantiation is True, model, optimizer, and ema are created in run().
           This is useful for normal training, where we want to seed the creation of these
           objects.
        """
        if lazy_instantiation:
            self.model = None
            self.ema_model = None
            self.optimizer = None
        else:
            self.model = hydra.utils.instantiate(cfg.policy)
            self.ema_model = None
            if cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0
        self.resume_state_version = 1
        self.next_epoch = None
        self.next_global_step = None
        self.optimizer_step = 0
        self.ema_state = None
        self.lr_scheduler_state = None
        self.lr_scheduler_config = None
        self.rng_state = None

    def _resume_training_progress(self, payload, *, legacy_epoch_complete=False):
        """Advance completed training snapshots without changing logged epoch IDs.

        Only run() opts into the legacy convention: its latest.ckpt was written
        after an epoch finished. Loading a policy or an arbitrary checkpoint does
        not itself advance training progress.
        """
        modern = 'resume_state_version' in payload.get('pickles', {})
        if modern:
            if self.next_epoch is not None:
                self.epoch = int(self.next_epoch)
                self.global_step = int(self.next_global_step)
            return
        saved_cfg = payload.get('cfg', {})
        known_workspace = saved_cfg.get('_target_') == 'oat.workspace.train_policy.TrainPolicyWorkspace'
        legacy_complete = (legacy_epoch_complete and known_workspace
                           and 'optimizer' in payload.get('state_dicts', {})
                           and {'epoch', 'global_step'} <= payload.get('pickles', {}).keys())
        # Adam's parameter step counters survive legacy checkpointing and are
        # more accurate than the logging counter with gradient accumulation.
        counts = [int(state['step']) for state in self.optimizer.state.values() if 'step' in state]
        accumulation = int(self.cfg.training.get('gradient_accumulate_every', 1))
        self.optimizer_step = max(counts) if counts else max(
            0, (self.global_step + int(legacy_complete) + accumulation - 1) // accumulation)
        if legacy_complete:
            self.epoch += 1
            self.global_step += 1
        warnings.warn(
            'Legacy training checkpoint has no EMA/scheduler/RNG resume metadata; '
            f'reconstructing optimizer updates as {self.optimizer_step}. '
            'Random-number streams cannot be recovered.', RuntimeWarning)

    def _create_training_dynamics(self, cfg, len_train_dataloader, ema_model=None):
        """Restore EMA and LR schedules after accelerator wraps the optimizer.

        Saved schedule configuration takes precedence over a changed epoch
        budget so extending a run does not silently change its existing curve.
        """
        ema = None
        if ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
            if self.ema_state is not None:
                ema.load_state_dict(self.ema_state)
            else:
                # Legacy checkpoints retain the averaged weights but not their
                # warmup count. Never reset a resumed EMA to decay zero.
                ema.optimization_step = self.optimizer_step
                ema.decay = ema.get_decay(max(self.optimizer_step - 1, 0))
        if self.lr_scheduler_config is None:
            self.lr_scheduler_config = {
                'name': cfg.training.lr_scheduler,
                'num_warmup_steps': cfg.training.lr_warmup_steps,
                'num_training_steps': (len_train_dataloader * cfg.training.num_epochs)
                                     // cfg.training.gradient_accumulate_every,
            }
        # Scheduler construction performs an initial step and changes group LRs.
        # A restored optimizer's LRs must survive that side effect.
        loaded_lrs = [group['lr'] for group in self.optimizer.param_groups]
        scheduler = get_scheduler(
            optimizer=self.optimizer, **self.lr_scheduler_config,
            last_epoch=-1 if self.lr_scheduler_state is not None else self.optimizer_step - 1,
        )
        if self.lr_scheduler_state is not None:
            scheduler.load_state_dict(self.lr_scheduler_state)
            for group, lr in zip(self.optimizer.param_groups, loaded_lrs):
                group['lr'] = lr
        return ema, scheduler

    def _capture_training_state(self, ema, scheduler, accelerator):
        """Capture a completed epoch, including each rank's main-process RNG.

        Persistent DataLoader workers and sampler generators are not serialized.
        Consequently resumed loader shuffling/worker-side augmentation can differ;
        exact numerical continuation requires the same batches and worker states.
        """
        self.next_epoch = self.epoch + 1
        self.next_global_step = self.global_step + 1
        self.ema_state = None if ema is None else copy.deepcopy(ema.state_dict())
        self.lr_scheduler_state = copy.deepcopy(scheduler.state_dict())
        state = {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if accelerator.device.type == 'cuda':
            state['cuda'] = torch.cuda.get_rng_state(accelerator.device)
        if accelerator.num_processes > 1:
            states = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(states, state)
            self.rng_state = states
        else:
            self.rng_state = [state]

    def _restore_training_rng(self, accelerator):
        if self.rng_state is None:
            return
        if len(self.rng_state) != accelerator.num_processes:
            warnings.warn('Process count changed; checkpoint RNG streams cannot be restored.', RuntimeWarning)
            return
        state = self.rng_state[accelerator.process_index]
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if accelerator.device.type == 'cuda' and 'cuda' in state:
            torch.cuda.set_rng_state(state['cuda'].cpu(), accelerator.device)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure accelerator
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(hours=2)), # sim eval can take long time
            ],
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            mixed_precision="bf16" if cfg.training.allow_bf16 and detect_bf16_support() else "no",
        )
        device = accelerator.device

        # set seed
        seed = int(cfg.training.seed)
        accelerate_set_seed(seed, device_specific=True)

        # configure model, ema, and optimizer after seeding
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.policy.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        _configure_models_from_dataset(dataset, self.model, self.ema_model)

        # configure checkpoint
        if accelerator.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"),
                **cfg.checkpoint.topk
            )

        # configure env
        lazy_eval = cfg.task.policy.lazy_eval  # don't eval during training
        if (not lazy_eval) and accelerator.is_main_process:
            env_runner: BaseRunner = hydra.utils.instantiate(
                cfg.task.policy.env_runner,
                output_dir=self.output_dir
            )

        # resume training
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {latest_ckpt_path}")
                payload = self.load_checkpoint(path=latest_ckpt_path)
                self._resume_training_progress(payload, legacy_epoch_complete=True)
                if self.epoch >= cfg.training.num_epochs:
                    accelerator.print(f"Already trained for {self.epoch} epochs. Exiting.")
                    return
                
        # Auxiliary artifacts may already be embedded in a resumed checkpoint.
        # Initialize them only after resume, and before DDP/EMA see module topology.
        self.model.prepare_for_training()
        if self.ema_model is not None:
            self.ema_model.prepare_for_training()

        # prepare with accelerator
        (
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        ) = accelerator.prepare(
            train_dataloader,
            val_dataloader,
            self.model,
            self.optimizer,
        )
        if cfg.training.use_ema:
            self.ema_model = accelerator.prepare(self.ema_model)
        len_train_dataloader = len(train_dataloader)
        ema, lr_scheduler = self._create_training_dynamics(
            cfg, len_train_dataloader,
            accelerator.unwrap_model(self.ema_model) if cfg.training.use_ema else None,
        )

        # configure logging
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        wandb_cfg['dir'] = str(self.output_dir)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        if accelerator.is_main_process:
            accelerator.get_tracker("wandb").run.config.update({
                "output_dir": str(self.output_dir)
            })

        # Restore after construction, device preparation, and tracker setup,
        # which can consume RNG draws unrelated to the continuation.
        self._restore_training_rng(accelerator)

        # training loop
        with JsonLogger(os.path.join(self.output_dir, 'logs.json')) as json_logger:
            while self.epoch < cfg.training.num_epochs:

                if accelerator.is_main_process:
                    step_log = dict()

                # model to train mode
                self.model.train()
                if cfg.training.use_ema:
                    self.ema_model.train()

                loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                with tqdm.tqdm(
                    train_dataloader, 
                    desc=f"Training epoch {self.epoch}",
                    leave=False, 
                    disable=not accelerator.is_local_main_process,
                    mininterval=cfg.training.tqdm_interval_sec
                ) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        with accelerator.accumulate(self.model):
                            # device transfer
                            batch = dict_apply(batch, lambda x: maybe_to_device(x, device))

                            # forward pass
                            with accelerator.autocast():
                                loss = self.model(batch)

                            # backward pass
                            accelerator.backward(loss)

                            # log loss
                            batch_size = batch['action'].shape[0]
                            loss_info[0] += loss.detach() * batch_size
                            loss_info[1] += batch_size

                            # step optimizer
                            if accelerator.sync_gradients:
                                # clip grad norm
                                if cfg.training.max_grad_norm is not None:
                                    _clip_grad_norm(
                                        accelerator, self.model, self.optimizer,
                                        cfg.training.max_grad_norm
                                    )
                            
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                if not getattr(self.optimizer, 'step_was_skipped', False):
                                    self.optimizer_step += 1
                                    lr_scheduler.step()
                                    if cfg.training.use_ema:
                                        ema.step(accelerator.unwrap_model(self.model))

                            # logging
                            is_last_batch = (batch_idx == (len_train_dataloader-1))
                            if accelerator.is_main_process:
                                loss_cpu = loss.item()
                                tepoch.set_postfix(loss=loss_cpu, refresh=False)
                                step_log = {
                                    'train_loss': loss_cpu,
                                    'global_step': self.global_step,
                                    'epoch': self.epoch,
                                    'lr': lr_scheduler.get_last_lr()[0],
                                }
                                if not is_last_batch:
                                    accelerator.log(step_log, step=self.global_step)
                                    json_logger.log(step_log)

                            # increment global step
                            if not is_last_batch:
                                self.global_step += 1

                            # break if reach max training steps
                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break

                # at the end of each epoch
                # replace train_loss with epoch average
                accelerator.wait_for_everyone()
                loss_info = accelerator.reduce(loss_info, reduction='sum')
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    step_log['train_loss'] = (loss_info[0] / loss_info[1]).item()

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = accelerator.unwrap_model(self.ema_model)
                policy.eval()

                rollout_due, checkpoint_due = _epoch_events(
                    self.epoch, cfg.training, lazy_eval=lazy_eval,
                    save_rollout_checkpoint=cfg.checkpoint.get('save_rollout_ckpt', False),
                    save_final_checkpoint=cfg.checkpoint.get('save_final_ckpt', False),
                )
                # A checkpoint-aware runner isolates simulator RNG and returns
                # metrics to this training run after a fixed-policy evaluation.
                if not lazy_eval:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and rollout_due:
                        run_checkpoint = getattr(env_runner, 'run_checkpoint', None)
                        if callable(run_checkpoint):
                            runner_log = run_checkpoint(
                                policy, cfg, epoch=self.epoch, global_step=self.global_step)
                        else:
                            runner_log = env_runner.run(policy)
                        step_log.update(runner_log)
                    accelerator.wait_for_everyone()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader, 
                            desc=f"Validation epoch {self.epoch}",
                            leave=False, 
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:
                            
                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                # forward pass
                                loss = policy(batch).item()

                                # log loss
                                batch_size = batch['action'].shape[0]
                                loss_info[0] += loss * batch_size
                                loss_info[1] += batch_size

                                # break if reach max val steps
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                    
                    # logging
                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['val_loss'] = (loss_info[0] / loss_info[1]).item()

                # action prediction eval
                if self.epoch % cfg.training.sample_every == 0:
                    loss_info = torch.zeros(2, device=device)   # [total loss, total batch_size]
                    with torch.inference_mode():
                        with tqdm.tqdm(
                            val_dataloader, 
                            desc=f"Reconstruction epoch {self.epoch}",
                            leave=False, 
                            disable=not accelerator.is_local_main_process,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:

                            for batch_idx, batch in enumerate(tepoch):
                                # device transfer
                                batch = dict_apply(batch, lambda x: maybe_to_device(x, device, non_blocking=True))

                                # action prediction
                                obs_dict = batch['obs']         # {key: [B, To, *]}
                                gt_action = batch['action']     # [B, Ta, Da]
                                result = policy.predict_action(obs_dict)
                                pred_action = result['action_pred']  # [B, Ta, Da]
                                mse = F.mse_loss(pred_action, gt_action).item()

                                # log loss
                                batch_size = batch['action'].shape[0]
                                loss_info[0] += mse * batch_size
                                loss_info[1] += batch_size

                                # early stop if reach max samples
                                if (cfg.training.max_reconst_steps is not None) \
                                    and batch_idx >= (cfg.training.max_reconst_steps-1):
                                    break

                    # logging
                    accelerator.wait_for_everyone()
                    loss_info = accelerator.reduce(loss_info, reduction='sum')
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        step_log['test_reconst_mse'] = (loss_info[0] / loss_info[1]).item()

                # Every rank contributes its RNG state before the main rank saves.
                if checkpoint_due:
                    self._capture_training_state(ema, lr_scheduler, accelerator)

                # checkpoint
                if accelerator.is_main_process and checkpoint_due:
                    # unwrap
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        ema_model_ddp = self.ema_model
                        self.ema_model = accelerator.unwrap_model(self.ema_model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    if rollout_due and cfg.checkpoint.get('save_rollout_ckpt', False):
                        archive = pathlib.Path(self.output_dir) / 'checkpoints' / f'epoch-{self.epoch + 1:04d}.ckpt'
                        if archive.exists():
                            raise FileExistsError(f"Refusing to overwrite rollout checkpoint: {archive}")
                        self.save_checkpoint(path=archive, use_thread=False)

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # restore
                    self.model = model_ddp
                    if cfg.training.use_ema:
                        self.ema_model = ema_model_ddp

                # end of epoch
                # log of last step is combined with validation and rollout
                if accelerator.is_main_process:
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                # increment epoch and global step
                self.epoch += 1
                self.global_step += 1

        # Ensure the final asynchronous latest checkpoint is complete on exit.
        if accelerator.is_main_process and self._saving_thread is not None:
            self._saving_thread.join()

        # clean up
        if not lazy_eval:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and (not lazy_eval):
                env_runner.close()
            accelerator.wait_for_everyone()
        accelerator.end_training()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()