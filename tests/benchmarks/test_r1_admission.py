"""CPU integration against the actual Scheduler methods, with native stubs."""
import copy
import json
import os
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_request_collector as native
import test_r1_shadow as existing

A = native.load_module("isolated_r1_admission", native.ROOT / "vllm/v1/core/sched/r1_admission.py")
S = existing.S


class Writer:
    instance_id = "test-writer"
    def __init__(self):
        self.rows = []
    def emit(self, kind, values):
        self.rows.append((kind, copy.deepcopy(values)))
        return True
    def close(self):
        return True


class Predictor:
    def __init__(self, scores):
        self.scores, self.calls = scores, 0
    def assemble(self, candidates, running, kv):
        return [r.request_id for r in candidates]
    def dynamic(self, rows):
        self.calls += 1
        return [[0., float(self.scores.get(r, 1)), float(self.scores.get(r, 1))] for r in rows]


def req(name, status=None, lengths=(1., 2., 3.)):
    return existing.request(name, status=status, lengths=lengths)


class AdmissionTests(unittest.TestCase):
    def setup_scheduler(self, mode="apply", names="abcd", scores=None, every=1, limit=32,
                        skip_single_slot=False):
        requests = [req(x) for x in names]
        obj = native.scheduler(waiting=requests)
        predictor = Predictor(scores or {"a": 9, "b": 3, "c": 1, "d": 1})
        writer = Writer()
        shadow = S.Shadow(S.ShadowConfig(Path("unused"), "run", Path(".")), predictor, writer)
        obj.r1_shadow = shadow
        obj.r1_admission_mode = mode
        obj.r1_admission = (None if mode == "off" else
                            A.AdmissionController(A.AdmissionConfig(mode, every, limit, skip_single_slot),
                                                  shadow, S.valid_lengths))
        return obj, predictor, writer

    def event(self, writer):
        return [v for k, v in writer.rows if k == "r1_admission_v1"][-1]

    def test_off_observe_apply(self):
        for mode, expected in (("off", ["a", "b", "c"]),
                               ("observe", ["a", "b", "c"]),
                               ("apply", ["a", "c", "d"])):
            with self.subTest(mode=mode):
                obj, pred, writer = self.setup_scheduler(mode)
                if mode != "off":
                    obj.r1_shadow.config = S.ShadowConfig(Path("unused"), "run", Path("."), sort=True)
                obj.max_num_running_reqs = 3
                obj.schedule()
                self.assertEqual([r.request_id for r in obj.running], expected)
                self.assertEqual(pred.calls, 1)
                if mode != "off":
                    row = self.event(writer)
                    self.assertEqual(row["sorted_ids"], ("a", "c", "d", "b"))
                    self.assertEqual(row["window_A"], 3)
                    self.assertEqual(row["window_M"], 3)
                    self.assertEqual(row["applied_order_changed"], mode == "apply")
                    self.assertFalse(any(k == "shadow_candidates" for k, _ in writer.rows))
                    self.assertFalse(any(k == "shadow_sort" for k, _ in writer.rows))
                else:
                    self.assertTrue(any(k == "shadow_candidates" for k, _ in writer.rows))
                    self.assertFalse(any(k == "r1_admission_v1" for k, _ in writer.rows))
                self.assertEqual([r.request_id for r in obj.waiting],
                                 [x for x in "abcd" if x not in expected])

    def test_single_slot_success_and_failure_preserve_native_resources(self):
        for mode in ("observe", "apply"):
            for fail in (False, True):
                outcomes = []
                for enabled in (False, True):
                    with self.subTest(mode=mode, fail=fail, enabled=enabled):
                        obj, pred, writer = self.setup_scheduler(mode=mode, skip_single_slot=enabled)
                        running = req("running", native.N["RequestStatus"].RUNNING)
                        # This running request receives no tokens this step. It still
                        # occupies a slot; scheduled_running_reqs is therefore zero.
                        running.num_computed_tokens = running.num_tokens
                        obj.running.append(running)
                        obj.requests[running.request_id] = running
                        obj.max_num_running_reqs = 2
                        if fail:
                            obj.kv_cache_manager.fail.add("a")
                        obj.kv_cache_manager.free = Mock(wraps=obj.kv_cache_manager.free)
                        obj.kv_cache_manager.free_block_hashes = Mock(wraps=obj.kv_cache_manager.free_block_hashes)
                        pred.assemble = Mock(wraps=pred.assemble)
                        obj.r1_shadow.sweep = Mock(side_effect=AssertionError("old shadow sweep"))
                        with patch.object(native.time, "monotonic", return_value=123.0):
                            output = obj.schedule()
                        row = self.event(writer)
                        outcomes.append(([r.request_id for r in obj.running],
                            [r.request_id for r in obj.waiting], output.num_scheduled_tokens,
                            list(obj.kv_cache_manager.allocations), list(obj.kv_cache_manager.cache_queries),
                            obj.kv_cache_manager.free.call_count, obj.kv_cache_manager.free_block_hashes.call_count,
                            obj.kv_cache_manager.usage_reads))
                        self.assertEqual(pred.calls, 0 if enabled else 1)
                        self.assertEqual(pred.assemble.call_count, 0 if enabled else 1)
                        self.assertEqual(row["available_running_slots"], 1)
                        self.assertEqual(row["skip_single_slot"], enabled)
                        self.assertEqual(row["reason"], "single_slot_skip" if enabled else "success")
                        if not fail:
                            admission = next(v for k, v in writer.rows if k == "shadow_first_admission")
                            self.assertEqual(admission["predicted_in_this_batch"], not enabled)
                            self.assertEqual(admission["snapshot"], {
                                "decision_batch_id": f"{row['instance_id']}:{row['schedule_id']}",
                                "scheduled_timestamp": 123.0, "running_count": 1,
                                "kv_usage": 0.0, "token_budget": 20})
                        if enabled:
                            self.assertEqual(row["candidate_ids"], tuple("abcd"))
                            self.assertEqual((row["prediction_ns"], row["decision_ns"],
                                              row["apply_ns"], row["restore_ns"]), (0, 0, 0, 0))
                            self.assertEqual(row["prediction_successes"], 0)
                            self.assertEqual((row["scores_p90"], row["sorted_ids"]), ((), ()))
                            if not fail:
                                head = obj.requests["a"]
                                self.assertFalse(head.r1_shadow_predicted)
                                obj.finish_requests("a", native.N["RequestStatus"].FINISHED_STOPPED)
                                terminal = next(v for k, v in writer.rows if k == "shadow_terminal")
                                self.assertTrue(terminal["was_admitted"])
                                self.assertFalse(terminal["was_candidate"])
                                self.assertEqual(terminal["status"], "FINISHED_STOPPED")
                        self.assertEqual(outcomes[-1][3], [("a", 3)])
                        self.assertEqual(outcomes[-1][4], ["a"])
                        if fail:
                            self.assertFalse(any(k == "shadow_first_admission" for k, _ in writer.rows))
                self.assertEqual(outcomes[0], outcomes[1])

    def test_single_slot_does_not_change_every_phase(self):
        obj, pred, writer = self.setup_scheduler(every=8, skip_single_slot=True)
        obj.kv_cache_manager.fail.add("a")
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(self.event(writer)["reason"], "single_slot_skip")
        obj.max_num_running_reqs = 2
        for _ in range(16):
            obj.schedule()
        rows = [v for k, v in writer.rows if k == "r1_admission_v1"]
        self.assertEqual([r["opportunity_id"] for r in rows if r["reason"] == "success"], [9, 17])
        self.assertEqual([r["reason"] for r in rows[1:8]], ["interval_skip"] * 7)
        self.assertTrue(all(r["checked"] == 0 for r in rows[1:8]))
        self.assertEqual(pred.calls, 2)
        self.assertEqual(obj.r1_admission.opportunities, 17)

    def test_single_slot_multiple_slots_and_special_head(self):
        obj, pred, writer = self.setup_scheduler(skip_single_slot=True)
        obj.max_num_running_reqs = 2
        obj.schedule()
        self.assertEqual(pred.calls, 1)
        self.assertEqual([r.request_id for r in obj.running], ["a", "c"])
        self.assertEqual([r.request_id for r in obj.waiting], ["b", "d"])
        self.assertEqual(self.event(writer)["available_running_slots"], 2)
        obj, pred, writer = self.setup_scheduler(skip_single_slot=True)
        obj.max_num_running_reqs = 1
        special = obj.waiting[0]
        special.status = native.N["RequestStatus"].WAITING_FOR_FSM
        special.structured_output_request = None
        obj.schedule()
        self.assertEqual(pred.calls, 0)
        self.assertEqual(self.event(writer)["reason"], "empty_prefix")
        self.assertEqual([r.request_id for r in obj.running], ["b"])
        self.assertEqual([r.request_id for r in obj.waiting], ["a", "c", "d"])
        obj, pred, writer = self.setup_scheduler(skip_single_slot=True)
        obj.max_num_running_reqs = 1
        obj.waiting[1].status = native.N["RequestStatus"].WAITING_FOR_FSM
        obj.waiting[1].structured_output_request = None
        obj.schedule()
        row = self.event(writer)
        self.assertEqual((row["reason"], row["candidate_ids"], row["checked"]),
                         ("single_slot_skip", ("a",), 2))
        self.assertEqual(pred.calls, 0)
        self.assertEqual([r.request_id for r in obj.waiting], ["b", "c", "d"])

    def test_single_slot_snapshot_after_running_scheduling(self):
        obj, predictor, writer = self.setup_scheduler(skip_single_slot=True)
        running = req("running", native.N["RequestStatus"].RUNNING)
        obj.running.append(running)
        obj.requests[running.request_id] = running
        obj.max_num_running_reqs = 2
        with patch.object(native.time, "monotonic", return_value=456.0):
            obj.schedule()
        admission = next(v for k, v in writer.rows if k == "shadow_first_admission")
        snapshot = admission["snapshot"]
        self.assertEqual((snapshot["scheduled_timestamp"], snapshot["running_count"],
                          snapshot["kv_usage"], snapshot["token_budget"]), (456.0, 1, .03, 17))
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(obj.kv_cache_manager.allocations, [("running", 3), ("a", 3)])

    def test_single_slot_invalid_snapshot_falls_back_without_prediction(self):
        obj, predictor, writer = self.setup_scheduler(skip_single_slot=True)
        obj.max_num_running_reqs = 1
        obj.kv_cache_manager.allocated = 101  # usage is outside [0, 1]
        obj.schedule()
        self.assertEqual(self.event(writer)["reason"], "snapshot_failed")
        self.assertEqual(predictor.calls, 0)
        admission = next(v for k, v in writer.rows if k == "shadow_first_admission")
        self.assertIsNone(admission["snapshot"])
        self.assertFalse(admission["predicted_in_this_batch"])

    def test_single_slot_real_writer_keeps_snapshot_terminal_and_footer(self):
        with tempfile.TemporaryDirectory(prefix="r1-single-slot-", dir=native.ROOT) as tmp:
            path = Path(tmp)
            writer = native.C.RecordWriter(native.C.CollectionConfig("single-slot", path), "shadow-core")
            obj, predictor, _ = self.setup_scheduler(skip_single_slot=True)
            shadow = S.Shadow(S.ShadowConfig(path / "unused", "single-slot", path), predictor, writer)
            obj.r1_shadow = shadow
            obj.r1_admission = A.AdmissionController(A.AdmissionConfig("apply", 1, 32, True),
                                                     shadow, S.valid_lengths)
            obj.max_num_running_reqs = 1
            obj.schedule()
            obj.finish_requests("a", native.N["RequestStatus"].FINISHED_STOPPED)
            self.assertTrue(shadow.close())
            rows = [json.loads(line) for line in writer.path.read_text("utf-8").splitlines()]
            event = next(r for r in rows if r["record_type"] == "r1_admission_v1")
            admission = next(r for r in rows if r["record_type"] == "shadow_first_admission")
            terminal = next(r for r in rows if r["record_type"] == "shadow_terminal")
            self.assertEqual(event["reason"], "single_slot_skip")
            self.assertEqual(admission["snapshot"]["running_count"], 0)
            self.assertFalse(admission["predicted_in_this_batch"])
            self.assertTrue(terminal["was_admitted"])
            self.assertFalse(terminal["was_candidate"])
            self.assertTrue(rows[-1]["clean_shutdown"])
            self.assertTrue(rows[-1]["records_complete"])

    def test_single_slot_config_and_off_mode(self):
        for value, expected in ((None, False), ("0", False), ("1", True)):
            env = {} if value is None else {"VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT": value}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(A.AdmissionConfig.from_env().skip_single_slot, expected)
        for invalid in ("", "true", "2", "-1"):
            with patch.dict(os.environ, {"VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT": invalid}, clear=True):
                with self.assertRaisesRegex(ValueError, "SKIP_SINGLE_SLOT"):
                    A.AdmissionConfig.from_env()
        obj, pred, writer = self.setup_scheduler(mode="off", skip_single_slot=True)
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(pred.calls, 1)  # Old shadow sweep, unaffected by RC1 flag.
        self.assertFalse(any(k == "r1_admission_v1" for k, _ in writer.rows))

    def test_opportunity_frequency_empty_failure_and_no_stale(self):
        obj, pred, writer = self.setup_scheduler(names="a", every=8)
        obj.kv_cache_manager.fail.add("a")
        for _ in range(17): obj.schedule()
        self.assertEqual(pred.calls, 3)
        rows = [v for k, v in writer.rows if k == "r1_admission_v1"]
        self.assertEqual([r["opportunity_id"] for r in rows if r["reason"] == "success"], [1, 9, 17])
        self.assertEqual([r["reason"] for r in rows[1:8]], ["interval_skip"] * 7)
        self.assertEqual([r.request_id for r in obj.waiting], ["a"])
        empty, p, w = self.setup_scheduler(names="a", every=1)
        empty.waiting[0].r1_shadow_lengths = None
        empty.kv_cache_manager.fail.add("a")
        empty.schedule()
        self.assertEqual(self.event(w)["reason"], "empty_prefix")
        self.assertEqual(empty.r1_admission.opportunities, 1)
        self.assertEqual(p.calls, 0)
        empty.waiting[0].r1_shadow_lengths = (1., 2., 3.)
        p.dynamic = Mock(side_effect=ValueError("prediction"))
        empty.schedule()
        self.assertEqual(self.event(w)["reason"], "prediction_failed")
        self.assertEqual(empty.r1_admission.opportunities, 2)

    def test_identity_mismatch_and_bad_prediction_fall_back(self):
        for fault in ("instance", "schedule", "ids", "duplicate", "nan", "negative", "raise"):
            with self.subTest(fault=fault):
                obj, pred, writer = self.setup_scheduler()
                original = obj.r1_admission.make_decision
                def make(frame, candidates, running, kv):
                    if fault == "raise": raise RuntimeError("predict")
                    d = original(frame, candidates, running, kv)
                    return A.Decision("other" if fault == "instance" else d.instance_id,
                        d.schedule_id + (fault == "schedule"),
                        (("a", "c", "b", "d") if fault == "ids" else
                         ("a", "b", "b", "d") if fault == "duplicate" else d.candidate_ids),
                        ((0., float("nan"), 1., 1.) if fault == "nan" else
                         (0., -1., 1., 1.) if fault == "negative" else d.scores))
                obj.r1_admission.make_decision = make
                obj.max_num_running_reqs = 2
                obj.schedule()
                self.assertEqual([r.request_id for r in obj.running], ["a", "b"])
                self.assertIn(self.event(writer)["reason"], ("invalid_decision", "prediction_failed"))

    def test_invalid_predictor_output_and_native_exception(self):
        for output in ([[0., float("nan"), 3.]] * 4, [[0., 1., 2.]],
                       [[0., -1., 2.]] * 4, [[0., True, 2.]] * 4):
            obj, predictor, writer = self.setup_scheduler()
            predictor.dynamic = Mock(return_value=output)
            obj.max_num_running_reqs = 2
            obj.schedule()
            self.assertEqual([r.request_id for r in obj.running], ["a", "b"])
            self.assertEqual(self.event(writer)["reason"], "prediction_failed")
            self.assertEqual(obj.r1_admission.prediction_successes, 0)
            predictor.dynamic.assert_called_once()

    def test_disabled_shadow_and_admission_make_no_predictions(self):
        obj, predictor, writer = self.setup_scheduler(mode="off")
        obj.r1_shadow = None
        obj.schedule()
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(writer.rows, [])
        self.assertEqual([r.request_id for r in obj.running], list("abcd"))

    def test_prefix_sizes_special_truncation_and_tail_skip(self):
        for n in (0, 1, 2):
            obj, pred, writer = self.setup_scheduler(names="ab"[:n])
            obj.schedule()
            self.assertEqual(pred.calls, int(n > 0))
            self.assertEqual(self.event(writer)["candidate_count"], n)
        obj, pred, writer = self.setup_scheduler(names="abcd")
        blocked = obj.waiting[2]
        blocked.status = native.N["RequestStatus"].WAITING_FOR_FSM
        blocked.structured_output_request = None
        obj.schedule()
        self.assertEqual(self.event(writer)["candidate_ids"], ("a", "b"))
        self.assertEqual(self.event(writer)["tail_A"], 2)
        self.assertEqual(self.event(writer)["tail_admitted"], ["d"])
        self.assertEqual([r.request_id for r in obj.waiting], ["c"])
        self.assertEqual([r.request_id for r in obj.running], ["a", "b", "d"])
        head, p, w = self.setup_scheduler(names="ab")
        head.waiting[0].r1_shadow_lengths = None
        head.schedule()
        self.assertEqual(self.event(w)["reason"], "empty_prefix")
        self.assertEqual(p.calls, 0)

    def test_request_eligibility_is_fail_closed(self):
        obj, _, _ = self.setup_scheduler()
        check = obj.r1_admission.eligible
        self.assertTrue(check(req("valid")))
        cases = []
        cases.append(req("preempted", native.N["RequestStatus"].PREEMPTED))
        cases.append(req("fsm", native.N["RequestStatus"].WAITING_FOR_FSM))
        cases.append(req("remote", native.N["RequestStatus"].WAITING_FOR_REMOTE_KVS))
        cases.append(req("no_static", lengths=None))
        for field, value in (("num_computed_tokens", 1), ("lora_request", object()),
                             ("has_encoder_inputs", True), ("mm_inputs", [1]),
                             ("mm_positions", [1]), ("kv_transfer_params", {}),
                             ("r1_shadow_admitted", True)):
            r = req(field)
            setattr(r, field, value)
            cases.append(r)
        guided = req("guided")
        guided.sampling_params.guided_decoding = object()
        cases.append(guided)
        emitted = req("output")
        emitted._output_token_ids.append(9)
        cases.append(emitted)
        missing_flag = req("missing_flag")
        del missing_flag.r1_shadow_admitted
        cases.append(missing_flag)
        for request in cases:
            with self.subTest(request=request.request_id):
                self.assertFalse(check(request))

    def test_capacity_token_and_kv_failure_restoration(self):
        for fail, slots, budget, admitted, remaining in (
            (None, 2, 20, ["a", "c"], ["b", "d"]),
            (None, 8, 6, ["a", "c"], ["b", "d"]),
            ("a", 8, 20, [], ["a", "b", "c", "d"]),
            ("c", 8, 20, ["a"], ["b", "c", "d"])):
            obj, pred, writer = self.setup_scheduler()
            obj.max_num_running_reqs, obj.max_num_scheduled_tokens = slots, budget
            if fail: obj.kv_cache_manager.fail.add(fail)
            obj.schedule()
            self.assertEqual([r.request_id for r in obj.running], admitted)
            self.assertEqual([r.request_id for r in obj.waiting], remaining)
            self.assertEqual(sorted(admitted + remaining), list("abcd"))
            self.assertEqual(pred.calls, 1)

    def test_preemption_and_resource_exceptions(self):
        obj, pred, writer = self.setup_scheduler(names="a")
        running = req("running", native.N["RequestStatus"].RUNNING)
        obj.running.append(running)
        obj.requests["running"] = running
        obj.kv_cache_manager.fail.add("running")
        obj.schedule()
        self.assertEqual(pred.calls, 0)
        self.assertEqual(self.event(writer)["reason"], "preempted")
        self.assertEqual([r.request_id for r in obj.waiting], ["running", "a"])
        bad, _, _ = self.setup_scheduler()
        bad.kv_cache_manager.allocate_slots = Mock(side_effect=RuntimeError("native KV"))
        with self.assertRaisesRegex(RuntimeError, "native KV"): bad.schedule()

    def test_long_tail_never_fully_scanned(self):
        obj, _, _ = self.setup_scheduler(names="abcd", limit=3)
        tail = [req("tail-" + str(i)) for i in range(3000)]
        class BoundedDeque(deque):
            def __iter__(self):
                for i, value in enumerate(super().__iter__()):
                    if i >= 3: raise AssertionError("scanned beyond window")
                    yield value
        obj.waiting = BoundedDeque([*obj.waiting, *tail])
        obj.max_num_running_reqs = 1
        obj.schedule()
        self.assertEqual(len(obj.waiting), 3003)

    def test_many_native_skips_use_bounded_attempt_logging(self):
        obj, pred, writer = self.setup_scheduler(names="")
        special = [req("special-" + str(i), native.N["RequestStatus"].WAITING_FOR_FSM)
                   for i in range(100)]
        for r in special:
            r.structured_output_request = None
        tail = req("tail")
        obj.waiting = deque([*special, tail])
        obj.requests = {r.request_id: r for r in [*special, tail]}
        obj.max_num_running_reqs = 1
        obj.schedule()
        event = self.event(writer)
        self.assertEqual(event["reason"], "empty_prefix")
        self.assertEqual(event["tail_A"], 101)
        self.assertEqual(event["actual_admission_order"], ["tail"])
        self.assertNotIn("tail_attempts", event)
        self.assertEqual([r.request_id for r in obj.waiting], [r.request_id for r in special])
        self.assertEqual(pred.calls, 0)

    def test_startup_configuration(self):
        for values in ({"VLLM_R1_ADMISSION_MODE": "bad"},
                       {"VLLM_R1_ADMISSION_EVERY": "0"},
                       {"VLLM_R1_ADMISSION_PREFIX_LIMIT": "0"},
                       {"VLLM_R1_ADMISSION_PREFIX_LIMIT": "bad"}):
            with patch.dict(os.environ, values, clear=True), self.assertRaises(ValueError):
                A.AdmissionConfig.from_env()
        cfg = types.SimpleNamespace(
            model_config=types.SimpleNamespace(is_multimodal_model=False, is_encoder_decoder=False,
                model="Meta-Llama-3-8B-Instruct", tokenizer="Meta-Llama-3-8B-Instruct",
                hf_config=types.SimpleNamespace(model_type="llama", hidden_size=4096,
                                                num_hidden_layers=32, vocab_size=128256),
                encoder_config=None),
            parallel_config=types.SimpleNamespace(data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1),
            scheduler_config=types.SimpleNamespace(policy="fcfs", scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler"),
            lora_config=None, kv_transfer_config=None, speculative_config=None)
        self.assertIsNone(A.unsupported_reason(cfg))
        for target, field, value, reason in (
                (cfg, "lora_config", object(), "lora_config"),
                (cfg, "kv_transfer_config", object(), "kv_connector"),
                (cfg, "speculative_config", object(), "speculative_decoding"),
                (cfg.model_config, "is_multimodal_model", True, "not_text_decoder"),
                (cfg.model_config, "is_encoder_decoder", True, "not_text_decoder"),
                (cfg.parallel_config, "data_parallel_size", 2, "parallelism_not_one"),
                (cfg.parallel_config, "pipeline_parallel_size", 2, "parallelism_not_one"),
                (cfg.parallel_config, "tensor_parallel_size", 2, "parallelism_not_one"),
                (cfg.scheduler_config, "policy", "priority", "scheduler_not_native_fcfs"),
                (cfg.scheduler_config, "scheduler_cls", "other", "scheduler_not_native_fcfs")):
            original = getattr(target, field)
            setattr(target, field, value)
            self.assertEqual(A.unsupported_reason(cfg), reason)
            setattr(target, field, original)
        cfg.model_config.model = "other"
        self.assertEqual(A.unsupported_reason(cfg), "shadow_model_or_tokenizer")
        # In active RC1, unsupported configurations return before loading the
        # old shadow model for both roles, even if no model path is configured.
        with patch.dict(os.environ, {"VLLM_R1_ADMISSION_MODE": "apply",
                                     "VLLM_R1_SHADOW": "1"}, clear=True), \
             patch.dict(sys.modules, {"vllm.v1.core.sched.r1_admission": A}), \
             patch.object(S, "Predictor", side_effect=AssertionError("model loaded")):
            for role in ("frontend", "core"):
                self.assertIsNone(S.create_shadow(cfg, role))

    def test_constructor_mode_gating_with_real_scheduler_init(self):
        ns = native.scheduler_namespace()
        ns.update(EventPublisherFactory=types.SimpleNamespace(create=lambda *args: None),
                  compute_encoder_budget=lambda **kwargs: (0, 0),
                  EncoderCacheManager=lambda **kwargs: types.SimpleNamespace(),
                  KVCacheManager=lambda **kwargs: native.FakeKV())
        cfg = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=20,
                max_model_len=100, policy="fcfs", scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler"),
            cache_config=types.SimpleNamespace(num_gpu_blocks=100, block_size=16,
                enable_prefix_caching=False, prefix_caching_hash_algo="builtin"),
            model_config=types.SimpleNamespace(is_multimodal_model=False, is_encoder_decoder=False,
                model="Meta-Llama-3-8B-Instruct", tokenizer="Meta-Llama-3-8B-Instruct",
                hf_config=types.SimpleNamespace(model_type="llama", hidden_size=4096,
                                                num_hidden_layers=32, vocab_size=128256), encoder_config=None),
            lora_config=None, kv_events_config=None, kv_transfer_config=None, speculative_config=None,
            parallel_config=types.SimpleNamespace(data_parallel_rank=0, data_parallel_size=1,
                pipeline_parallel_size=1, tensor_parallel_size=1))
        with patch.dict(sys.modules, {"vllm.v1.core.sched.r1_admission": A}):
            with patch.dict(os.environ, {"VLLM_R1_ADMISSION_MODE": "apply", "VLLM_R1_SHADOW": "0"}, clear=True):
                with self.assertLogs(ns["logger"], level="WARNING") as logs:
                    obj = ns["Scheduler"](cfg, types.SimpleNamespace(), types.SimpleNamespace())
                self.assertIsNone(obj.r1_admission)
                self.assertIn("shadow_disabled", str(logs.output))
            with patch.dict(os.environ, {"VLLM_R1_ADMISSION_MODE": "invalid"}, clear=True):
                with self.assertRaises(ValueError):
                    ns["Scheduler"](cfg, types.SimpleNamespace(), types.SimpleNamespace())
            with patch.dict(os.environ, {"VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT": "invalid"}, clear=True):
                with self.assertRaisesRegex(ValueError, "SKIP_SINGLE_SLOT"):
                    ns["Scheduler"](cfg, types.SimpleNamespace(), types.SimpleNamespace())
            with patch.dict(os.environ, {"VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT": "1"}, clear=True):
                obj = ns["Scheduler"](cfg, types.SimpleNamespace(), types.SimpleNamespace())
                self.assertEqual(obj.r1_admission_mode, "off")
                self.assertIsNone(obj.r1_admission)

    def test_timing_and_outcome_records_are_separate(self):
        obj, _, writer = self.setup_scheduler()
        obj.max_num_running_reqs = 2
        obj.schedule()
        row = self.event(writer)
        for key in ("trigger_prefix_ns", "prediction_ns", "decision_ns", "apply_ns", "restore_ns"):
            self.assertGreaterEqual(row[key], 0)
        self.assertEqual((row["waiting_size"], row["checked"], row["candidate_count"]), (4, 4, 4))
        self.assertEqual(row["window_admitted"], ["a", "c"])
        self.assertEqual(row["exit_reason"], "capacity")
        report = obj.r1_shadow.duration_report()["overall"]
        self.assertEqual(report["admission_schedule_foreground_v1"]["n"], 1)
        self.assertEqual(report["admission_log_foreground_v1"]["n"], 1)
        self.assertGreaterEqual(report["admission_schedule_foreground_v1"]["min_ns"],
                                sum(row[k] for k in ("trigger_prefix_ns", "prediction_ns",
                                    "decision_ns", "apply_ns", "restore_ns")))

    def test_real_async_writer_round_trip_and_footer(self):
        with tempfile.TemporaryDirectory(prefix="r1-admission-test-", dir=native.ROOT) as tmp:
            path = Path(tmp)
            writer = native.C.RecordWriter(native.C.CollectionConfig("admission-test", path), "shadow-core")
            obj, predictor, _ = self.setup_scheduler()
            shadow = S.Shadow(S.ShadowConfig(path / "unused", "admission-test", path), predictor, writer)
            obj.r1_shadow = shadow
            obj.r1_admission = A.AdmissionController(A.AdmissionConfig("apply", 1, 32), shadow, S.valid_lengths)
            obj.max_num_running_reqs = 2
            obj.schedule()
            self.assertTrue(shadow.close())
            rows = [json.loads(line) for line in writer.path.read_text("utf-8").splitlines()]
            row = next(row for row in rows if row["record_type"] == "r1_admission_v1")
            self.assertEqual(row["actual_admission_order"], ["a", "c"])
            summary = next(row for row in rows if row["record_type"] == "shadow_summary")
            self.assertEqual(summary["foreground_distributions"]["overall"]
                             ["admission_schedule_foreground_v1"]["n"], 1)
            self.assertTrue(rows[-1]["clean_shutdown"])
            self.assertTrue(rows[-1]["records_complete"])


if __name__ == "__main__":
    unittest.main()
