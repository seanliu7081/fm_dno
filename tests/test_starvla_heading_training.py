"""Trainer gates, exact sampler/RNG resume, and atomic checkpoint contracts."""
import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DistributedSampler

from oat.starvla_heading.evaluation import validate_metadata
from oat.starvla_heading.prior import ActionStatistics
from oat.starvla_heading.training import (
    ResumableBatchSampler, capture_rng_state, check_resume_contract, config_sha256,
    deepspeed_configuration, evaluate_original_libero, export_checkpoint, file_sha256,
    isolated_rng, load_training_state, normalized_error, optimizer_groups,
    representative_gradient_parameters, restore_rng_state, save_training_state, publish_checkpoint_links,
    seed_everything, validate_configuration, validation_schedule, verify_statistics_buffers,
    warmup_multiplier, repair_completed_status,
)


@pytest.fixture
def config():
    return OmegaConf.to_container(OmegaConf.load(Path(__file__).resolve().parents[1] /
                                                'oat/config/starvla_heading_gaussian.yaml'), resolve=True)


@pytest.fixture
def manifest():
    return {'suites': ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
            'tasks': [{}] * 40, 'episodes': [{}] * 2000, 'require_complete': True,
            'expected_demos_per_task': 50, 'missing_tasks': [], 'dataset_origin': 'original_libero',
            'train_episode_count': 1800, 'val_episode_count': 200, 'sha256': 'a' * 64}


@pytest.fixture
def statistics():
    return {'action': {'scale': [1.1] * 7, 'offset': [0.03] * 7},
            'state': {'scale': [1.01] * 9, 'offset': [0.11] * 9}, 'heading_xy_rms': 0.362203887}


def test_production_is_six_gpu_full_finetuning_with_safe_zero3(config, manifest):
    assert validate_configuration(config, manifest, 6) == 60
    ds = deepspeed_configuration(config)
    assert ds['train_micro_batch_size_per_gpu'] == 10
    assert ds['gradient_accumulation_steps'] == 1
    assert ds['bf16']['enabled'] and ds['zero_optimization']['stage'] == 3
    assert ds['zero_optimization']['stage3_gather_16bit_weights_on_model_save']
    assert not any('offload' in key for key in ds['zero_optimization'])
    assert ds['zero_optimization']['reduce_bucket_size'] == 5_000_000
    assert config['training']['optimizer']['weight_decay'] == 1e-6
    with pytest.raises(ValueError, match='six torchrun processes'):
        validate_configuration(config, manifest, 5)
    broken = copy.deepcopy(config)
    broken['training']['full_finetuning'] = False
    with pytest.raises(ValueError, match='parameters must train'):
        validate_configuration(broken, manifest)
    incomplete = {**manifest, 'tasks': [{}] * 10}
    with pytest.raises(ValueError, match='all 40'):
        validate_configuration(config, incomplete)
    broken = copy.deepcopy(config)
    broken['validation']['benchmark'] = 'libero_plus'
    with pytest.raises(ValueError, match='original LIBERO only'):
        validate_configuration(broken, manifest)


def test_resumed_sampler_exactly_matches_uninterrupted_rank_order():
    for rank_id in range(6):
        sampler = DistributedSampler(range(127), num_replicas=6, rank=rank_id,
                                      seed=42, shuffle=True, drop_last=True)
        for epoch in (0, 1, 3):
            sampler.set_epoch(epoch)
            full = list(ResumableBatchSampler(sampler, 2))
            for consumed in (0, 1, len(full) - 1, len(full)):
                resumed = ResumableBatchSampler(sampler, 2, consumed)
                assert list(resumed) == full[consumed:]
                assert len(resumed) == len(full) - consumed
    with pytest.raises(ValueError, match='cursor'):
        ResumableBatchSampler(sampler, 2, 500)


def test_validation_schedule_equal_collectives_and_no_duplicate_metric_weight():
    schedules = [list(validation_schedule(97, 17, seed=1729, rank_id=rank_id, world=6,
                                         micro_batch_size=2)) for rank_id in range(6)]
    assert len({len(schedule) for schedule in schedules}) == 1
    included = [index for schedule in schedules for batch in schedule for index, active in batch if active]
    assert len(included) == len(set(included)) == 17
    assert schedules == [list(validation_schedule(97, 17, seed=1729, rank_id=rank_id, world=6,
                                                  micro_batch_size=2)) for rank_id in range(6)]
    assert sum(not active for schedule in schedules for batch in schedule for _, active in batch) == 7


def sample(action, mask):
    return {'action': np.asarray(action, dtype=np.float32), 'action_mask': np.asarray(mask, dtype=bool)}


def test_validation_mse_excludes_action_padding_and_collective_fillers():
    stats = {'action': {'scale': [2.] * 7, 'offset': [1.] * 7}}
    examples = [sample(np.ones((3, 7)), [True, False, False]),
                sample(np.ones((3, 7)), [True, True, True])]
    predicted = torch.full((2, 3, 7), 1000.)
    predicted[0, 0] = 5  # normalized target=3, squared error=4.
    numerator, denominator = normalized_error(predicted, examples, stats, [True, False])
    assert denominator.item() == 7
    assert (numerator / denominator).item() == 4


def rng_draw():
    return random.random(), float(np.random.rand()), torch.rand(3)


def assert_same_rng(left, right):
    assert left[:2] == right[:2]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def test_validation_uses_engine_forward_and_restores_training_rng(statistics):
    class Data:
        def __init__(self):
            self.statistics = statistics
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return sample(np.zeros((16, 7)), [True] * 16)
    class Engine(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, examples, inference=False):
            assert inference
            self.calls += 1
            # Simulate the model's stochastic Gaussian action source.
            torch.rand(5)
            return {'normalized_actions': torch.full((len(examples), 16, 7), 0.03)}
    seed_everything(7)
    before = capture_rng_state()
    expected = rng_draw()
    restore_rng_state(before)
    engine = Engine().train()
    error = evaluate_original_libero(engine, Data(), {'seed': 9, 'num_samples': 3,
                                                     'micro_batch_size': 2}, torch.device('cpu'))
    assert error == pytest.approx(0)
    assert engine.calls == 2 and engine.training
    assert_same_rng(rng_draw(), expected)


class FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.value = 1
    def save_16bit_model(self, directory, save_filename):
        torch.save({'weight': torch.tensor([self.value])}, Path(directory) / save_filename)
        return True
    def save_checkpoint(self, directory, tag, client_state, save_latest):
        target = Path(directory) / tag
        target.mkdir(parents=True, exist_ok=True)
        (target / 'state.json').write_text(json.dumps(client_state))
        torch.rand(5)  # Saving must not alter future training randomness.
        if self.fail:
            raise RuntimeError('simulated disk failure')
    def load_checkpoint(self, directory, tag, load_module_strict):
        target = Path(directory) / tag
        return str(target), json.loads((target / 'state.json').read_text())


def test_export_has_hashed_raw_weights_metadata_and_retains_only_best_latest(tmp_path, config, manifest, statistics):
    config['statistics'] = statistics
    engine = FakeEngine()
    first = export_checkpoint(engine, tmp_path, config, manifest, 1, 0.5, True)
    metadata = json.loads((first / 'metadata.json').read_text())
    validate_metadata(metadata)
    assert metadata['checkpoint_sha256'] == file_sha256(first / 'weights.pt')
    assert metadata['config_sha256'] == file_sha256(first / 'config.json')
    assert metadata['dataset_manifest_sha256'] == manifest['sha256']
    assert metadata['camera_render_size'] == 128 and metadata['image_resize'] == 'bicubic'
    assert metadata['starvla_revision'] == '3422b9f2387b6f682cf02802904a77b23ab13afd'
    assert set(torch.load(first / 'weights.pt', weights_only=True)) == {'weight'}
    second = export_checkpoint(engine, tmp_path, config, manifest, 2, 0.6, False)
    assert (tmp_path / 'best').resolve() == first
    assert (tmp_path / 'latest').resolve() == second
    third = export_checkpoint(engine, tmp_path, config, manifest, 3, 0.4, True)
    assert not first.exists() and not second.exists()
    assert (tmp_path / 'best').resolve() == third == (tmp_path / 'latest').resolve()
    assert len(list((tmp_path / 'exports').glob('step_*'))) == 1


def test_interrupted_optimizer_checkpoint_cannot_be_resumed(tmp_path):
    state = {'step': 2, 'epoch': 0, 'batch_cursor': 20, 'world_size': 1,
             'config_sha256': 'c' * 64, 'dataset_manifest_sha256': 'a' * 64}
    with pytest.raises(RuntimeError, match='simulated disk failure'):
        save_training_state(FakeEngine(fail=True), tmp_path, state)
    assert (tmp_path / 'resume/INCOMPLETE').is_file()
    with pytest.raises(RuntimeError, match='INCOMPLETE'):
        load_training_state(FakeEngine(), tmp_path, 'c' * 64, 'a' * 64)


def test_committed_optimizer_resume_restores_exact_rank_rng_and_rejects_drift(tmp_path):
    state = {'step': 2, 'epoch': 0, 'batch_cursor': 20, 'world_size': 1,
             'config_sha256': 'c' * 64, 'dataset_manifest_sha256': 'a' * 64}
    seed_everything(17)
    before = capture_rng_state()
    expected = rng_draw()
    restore_rng_state(before)
    engine = FakeEngine()
    save_training_state(engine, tmp_path, state)
    assert not (tmp_path / 'resume/INCOMPLETE').exists()
    assert_same_rng(rng_draw(), expected)
    seed_everything(500)
    assert load_training_state(engine, tmp_path, 'c' * 64, 'a' * 64) == state
    assert_same_rng(rng_draw(), expected)
    for cfg, data, world in [('d' * 64, 'a' * 64, 1), ('c' * 64, 'b' * 64, 1),
                             ('c' * 64, 'a' * 64, 6)]:
        with pytest.raises(ValueError, match='differs'):
            check_resume_contract(state, cfg, data, world)


class TinyComponents(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = nn.Module()
        self.qwen.visual = nn.Module()
        self.qwen.visual.blocks = nn.ModuleList([nn.LayerNorm(2)])
        self.qwen.visual.merger = nn.LayerNorm(2)
        self.qwen.language_model = nn.Module()
        self.qwen.language_model.layers = nn.ModuleList([nn.LayerNorm(2)])
        self.qwen.language_model.embedding = nn.Linear(2, 3)
        self.action_model = nn.Module()
        self.action_model.model = nn.Module()
        self.action_model.model.transformer_blocks = nn.ModuleList([nn.LayerNorm(2)])
        self.action_model.heading_head = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 3))


def test_optimizer_updates_every_component_at_requested_learning_rates(config):
    model = TinyComponents()
    groups, counts = optimizer_groups(model, config['training']['optimizer'])
    assert sum(counts.values()) == sum(parameter.numel() for parameter in model.parameters())
    included = [id(parameter) for group in groups for parameter in group['params']]
    assert len(included) == len(set(included)) == len(list(model.parameters()))
    rates = {'vision': 5e-6, 'qwen': 1e-5, 'expert': 1e-4}
    for group in groups:
        assert group['lr'] == rates[group['name'].split('_')[0]]
    assert set(representative_gradient_parameters(model)) == {'vision', 'visual_merger', 'language', 'dit', 'heading'}
    model.qwen.visual.blocks[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match='unexpectedly froze'):
        optimizer_groups(model, config['training']['optimizer'])


def test_statistics_must_keep_exact_fp32_not_bf16_rounded_values(statistics):
    holder = SimpleNamespace(action_model=SimpleNamespace(statistics=ActionStatistics(statistics)))
    verify_statistics_buffers(holder, statistics, expected_device="cpu")
    with pytest.raises(ValueError, match="wrong device"):
        verify_statistics_buffers(holder, statistics, expected_device="meta")
    holder.action_model.statistics.action_scale = holder.action_model.statistics.action_scale.bfloat16().float()
    with pytest.raises(ValueError, match='FP32 fidelity'):
        verify_statistics_buffers(holder, statistics)


def test_constant_warmup_and_configuration_hash_are_stable(config):
    assert warmup_multiplier(0, 500) == 1 / 500
    assert warmup_multiplier(499, 500) == 1
    assert warmup_multiplier(30000, 500) == 1
    assert config_sha256(config) == config_sha256(copy.deepcopy(config))
    changed = copy.deepcopy(config)
    changed['training']['gradient_accumulation_steps'] = 2
    assert config_sha256(config) != config_sha256(changed)


def test_export_optimizer_commit_order_recovers_without_collision(tmp_path, config, manifest, statistics):
    config['statistics'] = statistics
    engine = FakeEngine()
    first = export_checkpoint(engine, tmp_path, config, manifest, 1, 0.5, True)
    state = {'step': 1, 'epoch': 0, 'batch_cursor': 10, 'world_size': 1,
             'config_sha256': 'c' * 64, 'dataset_manifest_sha256': manifest['sha256'],
             'latest_export_step': 1, 'best_export_step': 1}
    save_training_state(engine, tmp_path, state)
    future = export_checkpoint(engine, tmp_path, config, manifest, 2, 0.4, True, publish=False)
    assert first.exists() and future.exists()
    assert (tmp_path / 'best').resolve() == first
    # Interruption before optimizer commit leaves the step-one state authoritative.
    load_training_state(engine, tmp_path, 'c' * 64, manifest['sha256'])
    assert not future.exists() and first.exists()
    future = export_checkpoint(engine, tmp_path, config, manifest, 2, 0.4, True, publish=False)
    state.update(step=2, batch_cursor=20, latest_export_step=2, best_export_step=2)
    save_training_state(engine, tmp_path, state)
    # Interruption after optimizer commit but before symlink publication recovers too.
    load_training_state(engine, tmp_path, 'c' * 64, manifest['sha256'])
    assert (tmp_path / 'best').resolve() == future
    assert (tmp_path / 'latest').resolve() == future
    assert not first.exists()


def test_unvalidated_export_cannot_claim_zero_shot_checkpoint_selection(tmp_path, config, manifest):
    path = export_checkpoint(FakeEngine(), tmp_path, config, manifest, 1, None, False)
    metadata = json.loads((path / 'metadata.json').read_text())
    assert metadata['selection_benchmark'] is None
    with pytest.raises(ValueError, match='selection_benchmark'):
        validate_metadata(metadata)


def test_deepspeed_reserved_metadata_is_filtered_before_resume_resave(tmp_path):
    class MetadataEngine(FakeEngine):
        def save_checkpoint(self, directory, tag, client_state, save_latest):
            assert 'checkpoint_parallel_dimensions' not in client_state
            assert 'module' not in client_state
            assert 'global_steps' not in client_state
            return super().save_checkpoint(directory, tag, client_state, save_latest)
        def load_checkpoint(self, directory, tag, load_module_strict):
            path, state = super().load_checkpoint(directory, tag, load_module_strict)
            state.update(checkpoint_parallel_dimensions={'dp': 1, 'tp': 1},
                         module={'weight': torch.ones(3)}, global_steps=state['step'])
            return path, state
    original = {'step': 2, 'epoch': 0, 'batch_cursor': 20, 'world_size': 1,
                'config_sha256': 'c' * 64, 'dataset_manifest_sha256': 'a' * 64,
                'best_validation_mse': 1.25}
    engine = MetadataEngine()
    save_training_state(engine, tmp_path, original)
    loaded = load_training_state(engine, tmp_path, 'c' * 64, 'a' * 64)
    assert loaded == original
    loaded.update(step=3, batch_cursor=30)
    # Defensive filtering also protects callers accidentally passing DS metadata.
    save_training_state(engine, tmp_path, {**loaded, 'checkpoint_parallel_dimensions': {'dp': 1}})
    assert json.loads((tmp_path / 'resume/commit.json').read_text()) == loaded
    assert not (tmp_path / 'resume/INCOMPLETE').exists()
    assert load_training_state(engine, tmp_path, 'c' * 64, 'a' * 64) == loaded


def test_loaded_application_state_must_match_authoritative_commit(tmp_path):
    class DriftedEngine(FakeEngine):
        def load_checkpoint(self, directory, tag, load_module_strict):
            path, state = super().load_checkpoint(directory, tag, load_module_strict)
            state['batch_cursor'] += 1
            return path, state
    state = {'step': 2, 'epoch': 0, 'batch_cursor': 20, 'world_size': 1,
             'config_sha256': 'c' * 64, 'dataset_manifest_sha256': 'a' * 64}
    save_training_state(FakeEngine(), tmp_path, state)
    with pytest.raises(ValueError, match='authoritative optimizer commit'):
        load_training_state(DriftedEngine(), tmp_path, 'c' * 64, 'a' * 64)


def committed_final_fixture(tmp_path, config, manifest, statistics):
    config['statistics'] = statistics
    config['training']['max_steps'] = 3
    engine = FakeEngine()
    export_checkpoint(engine, tmp_path, config, manifest, 2, 0.4, True)
    export_checkpoint(engine, tmp_path, config, manifest, 3, 0.5, False)
    state = {'step': 3, 'epoch': 0, 'batch_cursor': 30, 'world_size': 6,
             'config_sha256': config_sha256(config), 'dataset_manifest_sha256': manifest['sha256'],
             'latest_export_step': 3, 'best_export_step': 2, 'best_validation_mse': 0.4}
    save_training_state(engine, tmp_path, state)
    return state


@pytest.mark.parametrize('stale_status', [False, True])
def test_resume_repairs_missing_or_stale_final_status_without_another_update(tmp_path, config, manifest, statistics, stale_status):
    state = committed_final_fixture(tmp_path, config, manifest, statistics)
    if stale_status:
        (tmp_path / 'status.json').write_text(json.dumps({'status': 'interrupted', 'step': 2}))
    optimizer_before = (tmp_path / 'resume/state/state.json').read_bytes()
    assert repair_completed_status(tmp_path, config, state)
    assert json.loads((tmp_path / 'status.json').read_text()) == {
        'status': 'completed', 'step': 3, 'best_validation_mse': 0.4}
    assert (tmp_path / 'resume/state/state.json').read_bytes() == optimizer_before
    assert (tmp_path / 'best').resolve().name == 'step_00000002'
    assert (tmp_path / 'latest').resolve().name == 'step_00000003'


def test_final_status_repair_requires_exact_budget_and_committed_final_export(tmp_path, config, manifest, statistics):
    state = committed_final_fixture(tmp_path, config, manifest, statistics)
    with pytest.raises(ValueError, match='exceeds'):
        repair_completed_status(tmp_path, config, {**state, 'step': 4})
    assert not repair_completed_status(tmp_path, config, {**state, 'step': 2})
    altered = {**state, 'latest_export_step': 2}
    (tmp_path / 'resume/commit.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='committed final export'):
        repair_completed_status(tmp_path, config, altered)
    assert not (tmp_path / 'status.json').exists()


def test_final_status_repair_requires_validated_final_and_intact_best(tmp_path, config, manifest, statistics):
    state = committed_final_fixture(tmp_path, config, manifest, statistics)
    final_metadata = tmp_path / 'exports/step_00000003/metadata.json'
    original = json.loads(final_metadata.read_text())
    final_metadata.write_text(json.dumps({**original, 'selection_benchmark': None}))
    with pytest.raises(ValueError, match='selection_benchmark'):
        repair_completed_status(tmp_path, config, state)
    final_metadata.write_text(json.dumps(original))
    with (tmp_path / 'exports/step_00000002/weights.pt').open('ab') as output:
        output.write(b'corrupt')
    with pytest.raises(ValueError, match='SHA256'):
        repair_completed_status(tmp_path, config, state)
    assert not (tmp_path / 'status.json').exists()
