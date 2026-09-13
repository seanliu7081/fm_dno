"""Protocol, benchmark accounting, and corruption-preserving rollout tests."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from oat.starvla_heading.evaluation import (
    PROMPT_PROTOCOL, PolicyClient, array_hash, audit_prompt_mapping, decode_image, encode_image, extract_observation,
    json_hash, load_instruction_catalog, resolve_task_prompt,
    freeze_manifest, load_official_states, load_results, make_plan, rollout_episode,
    smoke_metadata, summarize_results, validate_actions, validate_metadata,
)


def observation(value):
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) + value
    return {"agentview_image": image, "robot0_eye_in_hand_image": image + 20,
            "robot0_eef_pos": np.array([value, 2, 3]),
            "robot0_eef_quat": np.array([0, 0, 0, 1]),
            "robot0_gripper_qpos": np.array([0.03, -0.03])}


class FakeSuite:
    def __init__(self, names=("task_view_1", "task_language_2")):
        self.tasks = [SimpleNamespace(name=name, language=f"Rewritten instruction {i}.") for i, name in enumerate(names)]
        self.resolved = []

    def get_num_tasks(self):
        return len(self.tasks)

    def get_task(self, i):
        return self.tasks[i]

    def get_task_init_states(self, i):
        self.resolved.append(i)
        return np.arange(12, dtype=np.float64).reshape(2, 6) + i


def labels(suite):
    return {"libero_spatial": [{"id": i + 1, "name": task.name, "category": ("Camera Viewpoints", "Language Instructions")[i % 2],
                                "difficulty_level": i + 1} for i, task in enumerate(suite.tasks)]}


def catalog():
    instructions = {"libero_spatial": {"task": "pick up the black bowl and place it on the plate"}}
    return {"protocol": PROMPT_PROTOCOL, "training_manifest_sha256": "a" * 64,
            "instruction_catalog_sha256": json_hash(instructions), "instructions": instructions}


def prompt_inputs():
    return {"instruction_catalog": catalog(),
            "language_loader": lambda task: {"language": task.language, "language_bddl_sha256": "b" * 64}}


def plan():
    suite = FakeSuite()
    return make_plan({"libero_spatial": suite}, labels(suite), "libero_plus", 1, 42, **prompt_inputs())[0]


def test_orientation_state_and_lossless_wire_format():
    raw = observation(1)
    converted = extract_observation(raw)
    np.testing.assert_array_equal(converted["images"][0], raw["agentview_image"][::-1])
    assert not np.array_equal(converted["images"][0], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(decode_image(encode_image(converted["images"][1])), converted["images"][1])
    np.testing.assert_allclose(converted["state"], [1, 2, 3, 0, 0, 0, 1, .03, -.03])
    raw["agentview_image"][:] = 99
    assert not np.all(converted["images"][0] == 99)


def test_official_resolver_dynamic_counts_language_and_seed_independence():
    suite = FakeSuite()
    actual, scope = make_plan({"libero_spatial": suite}, labels(suite), "libero_plus", 1, 42, **prompt_inputs())
    assert scope["suite_counts"] == {"libero_spatial": 2}
    assert scope["available_episodes"] == 2
    assert suite.resolved == [0, 1]
    assert actual[1]["language"] == "Rewritten instruction 1."
    assert actual[1]["official_state_sha256"] == array_hash(suite.get_task_init_states(1)[0])
    subset, _ = make_plan({"libero_spatial": suite}, labels(suite), "libero_plus", 1, 42, 1, 1, **prompt_inputs())
    assert subset == actual[1:]
    with pytest.raises(ValueError, match="exactly one"):
        make_plan({"libero_spatial": suite}, labels(suite), "libero_plus", 50, 42, **prompt_inputs())
    classification = labels(suite)
    classification["libero_spatial"].pop()
    with pytest.raises(ValueError, match="exactly cover"):
        make_plan({"libero_spatial": suite}, classification, "libero_plus", 1, 42, **prompt_inputs())


def test_bad_official_states_rejected():
    suite = FakeSuite()
    suite.get_task_init_states = lambda i: np.array([1., 2., 3.])
    with pytest.raises(ValueError, match="Official state resolver"):
        load_official_states(suite, 0)


def test_zero_shot_metadata_and_action_contract():
    metadata = smoke_metadata()
    with pytest.raises(ValueError, match="training_benchmark"):
        validate_metadata(metadata)
    metadata.update(training_benchmark="libero", selection_benchmark="libero")
    validate_metadata(metadata)
    bad = {**metadata, "selection_benchmark": "libero_plus"}
    with pytest.raises(ValueError, match="selection_benchmark"):
        validate_metadata(bad)
    with pytest.raises(ValueError, match="orientation"):
        validate_metadata({**metadata, "image_orientation": "rotate180"})
    for bad_action in (np.zeros((8, 7)), np.full((16, 7), np.nan)):
        with pytest.raises(ValueError, match="raw actions"):
            validate_actions(bad_action)


class FakeEnv:
    def __init__(self):
        self.steps = 0
        self.actions = []

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return observation(0)

    def set_init_state(self, state):
        return observation(0)

    def step(self, action):
        self.steps += 1
        self.actions.append(action)
        # These are wrapper-corrupted frames, not clean simulator observations.
        return observation(self.steps), 0, False, {}

    def check_success(self):
        return False


class FakeClient:
    def __init__(self):
        self.calls = []

    def predict(self, history, language, seed, request_id):
        self.calls.append((deepcopy(list(history)), language, seed, request_id))
        actions = np.zeros((16, 7), dtype=np.float32)
        actions[:, -1] = .25
        return actions


def test_rollout_uses_latest_wrapper_obs_two_frames_execute_eight_and_raw_gripper():
    episode = {**plan()[0], "max_episode_steps": 9}
    env, client = FakeEnv(), FakeClient()
    result = rollout_episode(client, env, episode, np.zeros(6), {"settle_steps": 10})
    assert result["executed_steps"] == 9
    assert result["control_cycles"] == 2
    first, second = client.calls
    assert first[0][0]["state"][0] == first[0][1]["state"][0] == 10
    assert [frame["state"][0] for frame in second[0]] == [17, 18]
    assert first[1] == episode["language"]
    assert first[2] != second[2]
    assert first[3].endswith("control-0")
    assert env.actions[:10] == [[0.] * 6 + [-1.]] * 10
    assert all(action[-1] == .25 for action in env.actions[10:])
    np.testing.assert_array_equal(first[0][0]["images"][0], observation(10)["agentview_image"][::-1])


def test_smoke_never_calls_model_or_reports_benchmark_success():
    episode = plan()[0]
    result = rollout_episode(None, FakeEnv(), episode, np.zeros(6), {"settle_steps": 10, "smoke": True, "smoke_max_steps": 2})
    assert result["executed_steps"] == 2
    summary = summarize_results([episode], [result], official_scope=True, smoke=True)
    assert summary["complete"]
    assert summary["overall"]["success_rate"] is None
    assert summary["official_success_rate"] is None
    assert not summary["official_benchmark_complete"]


def test_summary_coverage_errors_and_official_category_difficulty():
    planned = plan()
    results = [{**entry, "success": i == 0, "error": None} for i, entry in enumerate(planned)]
    partial = summarize_results(planned, results[:1], official_scope=True)
    assert not partial["complete"]
    assert partial["official_success_rate"] is None
    full = summarize_results(planned, results, official_scope=True)
    assert full["official_success_rate"] == .5
    assert full["breakdowns"]["category"]["Camera Viewpoints"]["success_rate"] == 1
    assert full["breakdowns"]["category_by_difficulty"]["Language Instructions"]["2"]["success_rate"] == 0
    assert summarize_results(planned, results, official_scope=False)["official_success_rate"] is None
    broken = [{**results[0], "error": "renderer failure"}, results[1]]
    assert summarize_results(planned, broken, official_scope=True)["official_success_rate"] is None
    with pytest.raises(ValueError, match="duplicate"):
        summarize_results(planned, results + [results[0]])
    with pytest.raises(ValueError, match="changed its planned inputs"):
        summarize_results(planned, [{**results[0], "language": "task ID leakage"}])


def test_resume_rejects_changed_checkpoint_settings_and_results(tmp_path):
    manifest = {"policy": {"checkpoint_sha256": "a" * 64}, "plan": plan()}
    sha = freeze_manifest(tmp_path, manifest)
    assert freeze_manifest(tmp_path, manifest) == sha
    with pytest.raises(RuntimeError, match="Resume refused"):
        freeze_manifest(tmp_path, {**manifest, "policy": {"checkpoint_sha256": "b" * 64}})
    result_path = tmp_path / "results.jsonl"
    result_path.write_text(json.dumps({"manifest_sha256": sha, "episode_id": "test"}) + "\n")
    assert len(load_results(result_path, sha)) == 1
    with pytest.raises(ValueError, match="different evaluation manifest"):
        load_results(result_path, "c" * 64)


def test_real_http_protocol_four_images_temporal_order_seed_and_checkpoint():
    metadata = smoke_metadata()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(metadata).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"checkpoint_sha256": metadata["checkpoint_sha256"],
                                        "actions": np.zeros((16, 7)).tolist()}).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyClient(f"http://127.0.0.1:{server.server_port}", metadata)
        history = [extract_observation(observation(2)), extract_observation(observation(3))]
        actions = client.predict(history, "Actual rewritten language!", 123, "episode/control-0")
        assert actions.shape == (16, 7)
        payload = received[0]
        assert payload["language"] == "Actual rewritten language!"
        assert payload["seed"] == 123
        assert len(payload["images"]) == 4
        assert np.asarray(payload["state"]).shape == (2, 9)
        for i, image in enumerate(payload["images"]):
            np.testing.assert_array_equal(decode_image(image), history[i // 2]["images"][i % 2])
        metadata["checkpoint_sha256"] = "1" * 64
        with pytest.raises(RuntimeError, match="Checkpoint changed"):
            client.predict(history, "language", 456, "episode/control-1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_server_batches_requests_and_preserves_per_request_rng():
    from oat.starvla_heading.server import BatchingPredictor
    batches = []

    def fake_model(payloads):
        batches.append(len(payloads))
        return [np.random.default_rng(p["seed"]).normal(size=3).tolist() for p in payloads]

    predictor = BatchingPredictor(fake_model, max_batch_size=4, max_wait_ms=50)
    try:
        futures = [predictor.submit({"seed": seed}) for seed in range(4)]
        outputs = [future.result(timeout=2) for future in futures]
        assert batches == [4]
        singleton = predictor.submit({"seed": 2}).result(timeout=2)
        assert singleton == outputs[2]
    finally:
        predictor.close()


def test_server_validates_observation_size_and_checkpoint():
    from oat.starvla_heading.server import request_examples, validate_request
    metadata = smoke_metadata(image_size=2)
    history = [extract_observation(observation(2)), extract_observation(observation(3))]
    payload = {"protocol_version": 1, "checkpoint_sha256": metadata["checkpoint_sha256"],
               "request_id": "test-0", "seed": 1, "language": "move the bowl",
               "state": [frame["state"].tolist() for frame in history],
               "images": [encode_image(image) for frame in history for image in frame["images"]]}
    validate_request(payload, metadata)
    example = request_examples([payload])[0]
    assert example["lang"] == payload["language"]
    assert len(example["image"]) == 4
    assert example["state"].shape == (2, 9)
    for changes, match in [({"seed": True}, "unsigned"), ({"checkpoint_sha256": "1" * 64}, "checkpoint"),
                           ({"state": [[0]]}, "state"), ({"language": ""}, "language")]:
        with pytest.raises(ValueError, match=match):
            validate_request({**payload, **changes}, metadata)
    with pytest.raises(ValueError, match="shape"):
        validate_request(payload, {**metadata, "image_size": 224})


def test_full_server_client_http_round_trip_and_concurrent_batch():
    from concurrent.futures import ThreadPoolExecutor
    from http.server import ThreadingHTTPServer
    from oat.starvla_heading.server import BatchingPredictor, make_handler
    metadata = smoke_metadata(image_size=2)
    batches = []

    def model(payloads):
        batches.append(len(payloads))
        return [{"checkpoint_sha256": metadata["checkpoint_sha256"],
                 "actions": np.full((16, 7), p["seed"], dtype=float).tolist()} for p in payloads]

    predictor = BatchingPredictor(model, max_batch_size=4, max_wait_ms=100)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(metadata, predictor))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = PolicyClient(f"http://127.0.0.1:{server.server_port}", metadata)
        history = [extract_observation(observation(2)), extract_observation(observation(3))]
        with ThreadPoolExecutor(max_workers=4) as pool:
            calls = [pool.submit(client.predict, history, "put bowl on plate", seed, f"test-{seed}") for seed in range(4)]
            for seed, call in enumerate(calls):
                np.testing.assert_array_equal(call.result(), np.full((16, 7), seed))
        assert batches == [4]
    finally:
        server.shutdown()
        server.server_close()
        predictor.close()
        thread.join()


def test_error_attempts_resume_without_repeating_completed_trials(tmp_path):
    path = tmp_path / "results.jsonl"
    failed = {"episode_id": "task/0", "manifest_sha256": "f" * 64, "attempt": 1,
              "success": False, "error": "missing simulator asset"}
    completed = {**failed, "attempt": 2, "success": False, "error": None}
    path.write_text(json.dumps(failed) + "\n" + json.dumps(completed) + "\n")
    assert load_results(path, "f" * 64) == [completed]
    with path.open("a") as stream:
        stream.write(json.dumps({**completed, "attempt": 3}) + "\n")
    with pytest.raises(ValueError, match="cannot be retried"):
        load_results(path, "f" * 64)


def test_model_predictor_assigns_torch_noise_per_request_independent_of_batch():
    import torch
    from oat.starvla_heading.server import ModelPredictor

    class NoiseModel:
        def predict_action(self, examples, noise):
            assert len(examples) == len(noise)
            return {"actions": noise.numpy()}

    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.device = torch.device("cpu")
    predictor.metadata = smoke_metadata(image_size=2)
    predictor.model = NoiseModel()
    base = {"images": [encode_image(observation(1)["agentview_image"])] * 4,
            "state": np.zeros((2, 9)).tolist(), "language": "test language"}
    payloads = [{**base, "seed": i, "request_id": str(i)} for i in range(4)]
    together = predictor(payloads)
    alone = predictor([payloads[2]])[0]
    assert together[2] == alone
    assert together[0]["actions"] != alone["actions"]


def test_eval_resize_matches_training_flip_then_pil_bicubic():
    from PIL import Image
    raw = observation(5)
    extracted = extract_observation(raw, image_size=224)
    expected = np.asarray(Image.fromarray(np.ascontiguousarray(raw["agentview_image"][::-1])).resize(
        (224, 224), Image.Resampling.BICUBIC))
    np.testing.assert_array_equal(extracted["images"][0], expected)
    assert extracted["images"][1].shape == (224, 224, 3)



def test_nonlanguage_variants_use_exact_original_instruction_and_never_filename_metadata():
    instructions = catalog()
    canonical = instructions["instructions"]["libero_spatial"]["task"]
    suffixes = ("view_0_0_100_0_0_initstate_0_noise_40", "table_3", "tb_10", "add_10",
                "level1_sample1", "level5_sample6", "moved_level1_sample1", "light_9")
    for suffix in suffixes:
        task = SimpleNamespace(name="task_" + suffix, language="polluted instruction " + suffix.replace("_", " "))
        prompt = resolve_task_prompt(task, "libero_spatial", "libero_plus", instructions)
        assert prompt["language"] == canonical
        assert prompt["prompt_source"] == "original_training_manifest"
        assert prompt["prompt_sha256"] == json_hash(canonical)
        assert prompt["language_bddl_sha256"] is None
    original = SimpleNamespace(name="task", language="different base BDDL wording with akita")
    assert resolve_task_prompt(original, "libero_spatial", "libero", instructions)["language"] == canonical


def test_language_variant_preserves_full_rewrite_with_natural_metadata_words():
    language = "In my view, move the bowl; ignore the background noise and table 40."
    task = SimpleNamespace(name="task_language_14_view_0_0_100_0_0_initstate_0", language=language)
    loader = lambda task: {"language": language, "language_bddl_sha256": "c" * 64}
    prompt = resolve_task_prompt(task, "libero_spatial", "libero_plus", catalog(), loader)
    assert prompt["language"] == language
    assert prompt["prompt_source"] == "libero_plus_language_bddl"
    assert prompt["language_bddl_sha256"] == "c" * 64
    task.language += " view 0 0 initstate 0"
    with pytest.raises(ValueError, match="differs from its variant BDDL"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", catalog(), loader)


def test_unknown_ambiguous_or_unbound_prompt_sources_fail_closed():
    task = SimpleNamespace(name="task_newperturbation_1", language="instruction")
    with pytest.raises(ValueError, match="Unknown Plus"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", catalog())
    with pytest.raises(ValueError, match="frozen original"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", None)
    task.name = "unseen_view_1"
    with pytest.raises(ValueError, match="no unique original"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", catalog())
    task.name = "task_view_1"
    ambiguous = catalog()
    ambiguous["instructions"]["libero_spatial"]["task_view"] = "another instruction"
    ambiguous["instruction_catalog_sha256"] = json_hash(ambiguous["instructions"])
    with pytest.raises(ValueError, match="no unique original"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", ambiguous)
    broken = catalog()
    broken["instructions"]["libero_spatial"]["task"] = "changed instruction"
    with pytest.raises(ValueError, match="catalog changed"):
        resolve_task_prompt(task, "libero_spatial", "libero_plus", broken)
    with pytest.raises(ValueError, match="cannot contain"):
        resolve_task_prompt(task, "libero_spatial", "libero", catalog())


def test_frozen_catalog_verifies_policy_manifest_identity_and_all_original_tasks(tmp_path):
    from oat.starvla_heading.data import manifest_sha256
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    manifest = {"version": 1, "dataset_origin": "original_libero",
                "tasks": [{"suite": suite, "task": f"task{i}", "lang": f"Move item {i}."}
                          for suite in suites for i in range(10)]}
    manifest["sha256"] = manifest_sha256(manifest)
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = load_instruction_catalog(path, manifest["sha256"])
    assert loaded["training_manifest_sha256"] == manifest["sha256"]
    assert sum(map(len, loaded["instructions"].values())) == 40
    with pytest.raises(ValueError, match="policy's training manifest"):
        load_instruction_catalog(path, "0" * 64)
    manifest["tasks"][0]["lang"] = "Tampered instruction."
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_instruction_catalog(path)


def test_prompt_audit_is_order_independent_and_resume_rejects_previous_prompt_protocol(tmp_path):
    planned = plan()
    audit = audit_prompt_mapping(planned, catalog())
    assert audit == audit_prompt_mapping(list(reversed(planned)), catalog())
    assert audit["source_counts"] == {"original_training_manifest": 1, "libero_plus_language_bddl": 1}
    assert audit["mapped_tasks"] == 2
    previous = {"plan": planned}
    freeze_manifest(tmp_path, previous)
    with pytest.raises(RuntimeError, match="Resume refused"):
        freeze_manifest(tmp_path, {**previous, "prompt_audit": audit})
    changed = deepcopy(planned)
    changed[1]["language"] += " changed"
    with pytest.raises(ValueError, match="instruction hash mismatch"):
        audit_prompt_mapping(changed, catalog())
