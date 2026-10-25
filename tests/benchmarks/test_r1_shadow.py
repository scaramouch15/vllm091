# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of actual shadow code/native methods with resource stubs.

Real sklearn B-model and Msgpack/process checks have separate server commands;
these local tests never install packages, load an LLM or initialize CUDA.
"""
import ast
import copy
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_request_collector as native

ROOT = native.ROOT
S = native.load_module("isolated_r1_shadow", ROOT / "vllm/v1/metrics/r1_shadow.py")
NS = types.SimpleNamespace


class FakePredictor:
    def __init__(self):
        self.static_calls = []
        self.dynamic_calls = []

    def static(self, text, tokens):
        self.static_calls.append((text, tokens))
        return (1., 2., 3.), {"tfidf_ns": 1, "svd_ns": 1,
                             "text_transform_ns": 2, "length_head_ns": 3}

    def assemble(self, candidates, running_count, kv_usage):
        return [[r.num_prompt_tokens, running_count, kv_usage,
                 *r.r1_shadow_lengths] for r in candidates]

    def dynamic(self, matrix):
        self.dynamic_calls.append(copy.deepcopy(matrix))
        return [[float(row[0]), float(row[0] + 1), float(row[0] + 2)]
                for row in matrix]


def completion_request(**kw):
    values = dict(prompt="  x\n    y\n", prompt_embeds=None, n=1,
                  use_beam_search=False, truncate_prompt_tokens=None,
                  add_special_tokens=False, kv_transfer_params=None)
    return NS(**dict(values, **kw))


def request(name, status=None, lengths=(1., 2., 3.)):
    r = native.request(name, status=status)
    r.max_tokens = r.sampling_params.max_tokens = 2048
    r.r1_shadow_lengths = lengths
    if lengths is not None:
        r.r1_shadow_admitted = r.r1_shadow_predicted = False
    return r


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r1-shadow-test-", dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def shadow(self, every=1, cap=32, scan=128):
        config = S.ShadowConfig(self.directory / "unused", "shadow-test",
                                self.directory, cap, scan, every)
        writer = native.C.RecordWriter(
            native.C.CollectionConfig(config.run_id, config.directory), "shadow-core")
        shadow = S.Shadow(config, FakePredictor(), writer)
        self.addCleanup(shadow.close)
        return shadow

    def records(self, shadow):
        shadow.close()
        return [__import__("json").loads(line)
                for line in shadow.writer.path.read_text("utf-8").splitlines()]

    def test_disabled_no_import_model_writer_or_id(self):
        with patch.dict(os.environ, {"VLLM_R1_SHADOW": "0"}, clear=True), \
             patch.object(S, "Predictor", side_effect=AssertionError("model")), \
             patch.object(native.C, "RecordWriter", side_effect=AssertionError("writer")):
            before = set(sys.modules)
            self.assertIsNone(S.create_shadow(None, "frontend"))
            self.assertFalse((set(sys.modules) - before) & {"sklearn", "joblib", "numpy"})

    def test_config_gates_and_no_global_rng(self):
        for env in ({"VLLM_R1_SHADOW": "bad"}, {"VLLM_R1_SHADOW": "1"},
                    {"VLLM_R1_SHADOW": "1", "VLLM_R1_SHADOW_MODEL": "x",
                     "VLLM_R1_SHADOW_RUN_ID": "../x", "VLLM_R1_SHADOW_DIR": "x"}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                S.ShadowConfig.from_env()
        tree = ast.parse((ROOT / "vllm/v1/metrics/r1_shadow.py").read_text())
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
        self.assertNotIn("random", S.__dict__)

    def test_startup_wrong_model_rejected_before_deserialization(self):
        wrong = self.directory / "bad.joblib"
        wrong.write_bytes(b"not the B model")
        with patch.object(S.platform, "python_version", return_value="3.10.22"), \
             self.assertRaisesRegex(ValueError, "SHA256"):
            S.Predictor(wrong, "frontend")

    def test_factory_loads_once_and_collector_switch_is_independent(self):
        predictor = FakePredictor()
        predictor.threadpools = [{"num_threads": 1}]
        predictor.environment = {"python": "fixture"}
        config = NS(model_config=NS(model="fixture-model", tokenizer="fixture-tokenizer",
                                   revision=None, tokenizer_revision=None, dtype="float16"))
        env = {"VLLM_R1_SHADOW": "1", "VLLM_R1_COLLECT": "0",
               "VLLM_R1_SHADOW_MODEL": str(self.directory / "model"),
               "VLLM_R1_SHADOW_RUN_ID": "factory-test",
               "VLLM_R1_SHADOW_DIR": str(self.directory)}
        with patch.dict(os.environ, env, clear=True), \
             patch.dict(sys.modules, {"vllm.v1.metrics.request_collector": native.C}), \
             patch.object(S, "validate_service"), \
             patch.object(S, "Predictor", return_value=predictor) as loader:
            shadow = S.create_shadow(config, "frontend")
            self.addCleanup(shadow.close)
            for name in ("ordinary-1", "ordinary-2"):
                req, params = completion_request(), native.params(max_tokens=2048)
                shadow.prepare(req, params, {"prompt": req.prompt, "prompt_token_ids": [1]},
                               {"prompt_token_ids": [1]}, name)
            loader.assert_called_once()
            self.assertEqual(len(predictor.static_calls), 2)
        rows = self.records(shadow)
        self.assertEqual(rows[0]["role"], "shadow-frontend")
        self.assertEqual(rows[0]["metadata"]["environment"], predictor.environment)
        self.assertGreaterEqual(rows[0]["metadata"]["startup_load_ns"], 0)

    def test_static_once_per_request_no_text_cache_or_whitespace_changes(self):
        shadow = self.shadow()
        req = completion_request()
        for name in ("ordinary-curl", "same-text-new-request"):
            params = native.params(max_tokens=2048, extra_args={"other": 42})
            shadow.prepare(req, params, {"prompt": req.prompt, "prompt_token_ids": [1, 2]},
                           {"prompt_token_ids": [1, 2]}, name)
            self.assertEqual(params.extra_args["other"], 42)
            cloned = copy.deepcopy(params)
            self.assertEqual(S.consume_static(cloned, 2, shadow), (1., 2., 3.))
            self.assertNotIn(S.INTERNAL_KEY, cloned.extra_args)
        self.assertEqual(shadow.predictor.static_calls, [(req.prompt, 2), (req.prompt, 2)])
        records = self.records(shadow)
        self.assertEqual(sum(r["record_type"] == "shadow_static" for r in records), 2)
        self.assertNotIn(req.prompt, shadow.writer.path.read_text("utf-8"))
        self.assertEqual(records[-1]["metrics"].get("dropped_records", 0), 0)

    def test_unsupported_inputs_preserve_native_params(self):
        cases = [dict(prompt=["a"]), dict(prompt=[1, 2]), dict(prompt_embeds="embeds"),
                 dict(n=2), dict(use_beam_search=True), dict(truncate_prompt_tokens=-1),
                 dict(add_special_tokens=True), dict(kv_transfer_params={"x": 1})]
        shadow = self.shadow()
        for change in cases:
            req = completion_request(**change)
            params = native.params(max_tokens=2048, extra_args={"other": 1})
            shadow.prepare(req, params, {}, {}, "id")
            self.assertEqual(params.extra_args, {"other": 1})
        self.assertFalse(shadow.predictor.static_calls)
        for max_tokens in (None, 2047, 2049):
            params = native.params(max_tokens=max_tokens)
            shadow.prepare(completion_request(), params, {}, {}, "id")
            self.assertEqual(params.max_tokens, max_tokens)
        self.assertFalse(shadow.predictor.static_calls)

    def test_forgery_token_and_limit_mismatch_discarded(self):
        shadow = self.shadow()
        for value, tokens, max_tokens, n in (
            ({"lengths": [1, 2, 3]}, 2, 2048, 1),
            (S.StaticEnvelope(2, 2048, (1., 2., 3.)), 3, 2048, 1),
            (S.StaticEnvelope(2, 2048, (1., 2., 3.)), 2, 8, 1),
            (S.StaticEnvelope(2, 2048, (1., 2., 3.)), 2, 2048, 2),
            (S.StaticEnvelope(2, 2048, (1., float("nan"), 3.)), 2, 2048, 1),
        ):
            params = native.params(max_tokens=max_tokens, n=n,
                                   extra_args={S.INTERNAL_KEY: value, "kv_transfer_params": {"a": 1}})
            self.assertIsNone(S.consume_static(params, tokens, shadow))
            self.assertEqual(params.extra_args, {"kv_transfer_params": {"a": 1}})

    def test_static_exception_falls_back_without_zero_prediction(self):
        shadow = self.shadow()
        shadow.predictor.static = Mock(side_effect=RuntimeError("private prompt"))
        req = completion_request()
        params = native.params(max_tokens=2048)
        shadow.prepare(req, params, {"prompt": req.prompt, "prompt_token_ids": [1]},
                       {"prompt_token_ids": [1]}, "id")
        self.assertNotIn(S.INTERNAL_KEY, params.extra_args or {})
        rows = self.records(shadow)
        error = next(r for r in rows if r["record_type"] == "shadow_static_error")
        self.assertEqual(error["error_type"], "RuntimeError")
        self.assertNotIn("private prompt", shadow.writer.path.read_text())

    def test_actual_scheduler_shadow_on_off_identical(self):
        baseline = subprocess.check_output([
            "git", "show", "09c35e97b0605bd437c725562ca05afaacdb82e1:vllm/v1/core/sched/scheduler.py"],
            cwd=ROOT, text=True)
        cls = native.scheduler_namespace(baseline)["Scheduler"]
        outputs = []
        for mode in ("baseline", "off", "on"):
            obj = native.scheduler([request("running", native.N["RequestStatus"].RUNNING)],
                                   [request("a"), request("b")], cls=cls if mode == "baseline" else None)
            if mode == "on":
                obj.r1_shadow = self.shadow()
            with patch.object(native.time, "monotonic", return_value=10.):
                output = obj.schedule()
            outputs.append((vars(output), [r.request_id for r in obj.running],
                            list(obj.waiting), obj.kv_cache_manager.allocations,
                            obj.kv_cache_manager.cache_queries,
                            [(r.request_id, r.num_computed_tokens, r.status,
                              [(e.type, e.timestamp) for e in r.events]) for r in obj.running]))
        def normalize(x):
            if isinstance(x, NS): return normalize(vars(x))
            if isinstance(x, dict): return {k: normalize(v) for k, v in x.items()}
            if isinstance(x, (tuple, list)): return [normalize(v) for v in x]
            return x
        self.assertEqual(normalize(outputs[0]), normalize(outputs[1]))
        self.assertEqual(normalize(outputs[0]), normalize(outputs[2]))

    def test_failed_allocation_candidate_is_not_admission_later_resume_excluded(self):
        shadow = self.shadow()
        r = request("failed")
        obj = native.scheduler(waiting=[r])
        obj.r1_shadow = shadow
        obj.kv_cache_manager.fail.add("failed")
        obj.schedule()
        self.assertFalse(r.r1_shadow_admitted)
        obj.schedule()
        self.assertEqual(len(shadow.predictor.dynamic_calls), 2)
        self.assertEqual(shadow.predictor.static_calls, [])
        obj.kv_cache_manager.fail.clear()
        obj.schedule()
        self.assertTrue(r.r1_shadow_admitted)
        obj.running.remove(r)
        r.status = native.N["RequestStatus"].PREEMPTED
        r.num_computed_tokens = 0
        obj.kv_cache_manager.create_empty_block_list = lambda: NS()
        obj.waiting.appendleft(r)
        obj.schedule()
        r.status = native.N["RequestStatus"].WAITING
        r.num_computed_tokens = 0
        obj.running.remove(r)
        obj.waiting.appendleft(r)
        obj.schedule()
        rows = self.records(shadow)
        self.assertEqual(sum(x["record_type"] == "shadow_first_admission" for x in rows), 1)
        self.assertEqual(shadow.counts["candidate_predictions"], 3)

    def test_bounded_order_shared_snapshot_and_frequency(self):
        shadow = self.shadow(every=2, cap=2, scan=4)
        missing, a, b, c = request("missing", lengths=None), request("a"), request("b"), request("c")
        waiting = native.deque([missing, a, b, c])
        kv = NS(usage=0.25)
        snapshot, ids = shadow.sweep(waiting, [], kv, 20, 3.)
        self.assertEqual(ids, ("a", "b"))
        self.assertEqual([r.request_id for r in waiting], ["missing", "a", "b", "c"])
        self.assertEqual(shadow.predictor.dynamic_calls[0], [[3, 0, .25, 1., 2., 3.]] * 2)
        self.assertEqual(shadow.sweep(waiting, [], kv, 20, 3.)[1], ())
        kv.usage = .9
        a.num_prompt_tokens = 999
        rows = self.records(shadow)
        record = next(x for x in rows if x["record_type"] == "shadow_candidates")
        self.assertEqual(record["snapshot"]["kv_usage"], .25)
        self.assertEqual(record["candidates"][0]["prompt_tokens"], 3)
        self.assertEqual(record["candidate_count"], 2)

    def test_native_gates_no_candidates_no_prediction_and_dynamic_failure(self):
        for budget, slots, wait in ((0, 8, [request("a")]), (20, 0, [request("a")]), (20, 8, [])):
            obj = native.scheduler(waiting=wait)
            obj.max_num_scheduled_tokens, obj.max_num_running_reqs = budget, slots
            obj.r1_shadow = self.shadow()
            obj.schedule()
            self.assertEqual(obj.r1_shadow.batch, 0)
        obj = native.scheduler(waiting=[request("missing", lengths=None)])
        obj.r1_shadow = self.shadow()
        obj.schedule()
        self.assertEqual(obj.r1_shadow.predictor.dynamic_calls, [])
        obj = native.scheduler(waiting=[request("a")])
        obj.r1_shadow = self.shadow()
        obj.r1_shadow.predictor.dynamic = Mock(side_effect=RuntimeError("oops"))
        obj.schedule()
        self.assertEqual([r.request_id for r in obj.running], ["a"])
        rows = self.records(obj.r1_shadow)
        self.assertFalse(any(r["record_type"] == "shadow_candidates" for r in rows))
        admission = next(r for r in rows if r["record_type"] == "shadow_first_admission")
        self.assertFalse(admission["predicted_in_this_batch"])

    def test_terminal_and_close_and_bounded_writer_failure(self):
        shadow = self.shadow()
        r = request("cancelled")
        r.status = native.N["RequestStatus"].FINISHED_ABORTED
        obj = native.scheduler(waiting=[])
        obj.requests[r.request_id] = r
        obj.r1_shadow = shadow
        obj._free_request(r)
        self.assertNotIn(r.request_id, obj.requests)
        obj.shutdown()
        self.assertFalse(shadow.writer._thread.is_alive())
        rows = self.records(shadow)
        terminal = next(r for r in rows if r["record_type"] == "shadow_terminal")
        self.assertFalse(terminal["was_admitted"])
        self.assertTrue(rows[-1]["clean_shutdown"])
        broken = self.shadow()
        broken.writer.emit = Mock(side_effect=OSError("disk"))
        broken.emit("shadow_test", {})
        self.assertEqual(broken.counts["log_errors"], 1)

    def test_request_transfer_default_and_numbers(self):
        kwargs = dict(request_id="ordinary-id", prompt_token_ids=[1, 2],
                      mm_inputs=None, mm_hashes=None, mm_placeholders=None,
                      sampling_params=native.params(max_tokens=2048),
                      eos_token_id=None, client_index=0, lora_request=None, cache_salt=None)
        for lengths in (None, (1., 2., 3.)):
            ecr = NS(**kwargs, r1_shadow_lengths=lengths)
            r = native.N["Request"].from_engine_core_request(ecr)
            self.assertEqual(r.request_id, "ordinary-id")
            self.assertEqual(r.r1_shadow_lengths, lengths)
            self.assertEqual(r.num_prompt_tokens, 2)
            self.assertEqual(hasattr(r, "r1_shadow_admitted"), lengths is not None)

    def test_actual_processor_consumes_only_internal_key_before_ecr(self):
        shadow = self.shadow()
        namespace = dict(native.N, SamplingParams=NS, EngineCoreRequest=NS,
                         MULTIMODAL_REGISTRY=None,
                         split_enc_dec_inputs=lambda x: (None, x))
        native.load_defs("vllm/v1/engine/processor.py", ["Processor"], namespace)
        obj = namespace["Processor"].__new__(namespace["Processor"])
        obj.r1_shadow = shadow
        obj.model_config = NS(max_model_len=8192)
        obj.vllm_config = NS(parallel_config=NS(data_parallel_size=1))
        obj._validate_lora = obj._validate_params = obj._validate_model_inputs = Mock()
        obj.use_hash = False
        obj.generation_config_fields = {}
        obj.tokenizer = NS(get_lora_tokenizer=lambda _: None)
        obj.input_preprocessor = NS(preprocess=lambda *a, **kw:
            {"type": "token", "prompt_token_ids": [1, 2]}, get_eos_token_id=lambda _: 0)
        value = S.StaticEnvelope(2, 2048, (1., 2., 3.))
        params = native.params(max_tokens=2048, extra_args={S.INTERNAL_KEY: value, "other": 9})
        def clone():
            return native.params(max_tokens=params.max_tokens,
                extra_args=copy.deepcopy(params.extra_args),
                update_from_generation_config=Mock(), update_from_tokenizer=Mock())
        params.clone = clone
        platform_module = types.ModuleType("vllm.platforms")
        platform_module.current_platform = NS(validate_request=Mock())
        with patch.dict(sys.modules, {"vllm.platforms": platform_module,
                                     "vllm.v1.metrics.r1_shadow": S}):
            _, ecr = obj.process_inputs("not-a-required-prefix", {"prompt_token_ids": [1, 2]}, params)
        self.assertEqual(ecr.r1_shadow_lengths, (1., 2., 3.))
        self.assertEqual(ecr.sampling_params.extra_args, {"other": 9})
        self.assertEqual(params.extra_args[S.INTERNAL_KEY], value)
        r = native.N["Request"].from_engine_core_request(NS(**vars(ecr), client_index=0))
        self.assertEqual(r.r1_shadow_lengths, (1., 2., 3.))
        self.assertFalse(r.r1_shadow_admitted)

    def test_actual_completion_calls_static_once_and_ordinary_id_is_accepted(self):
        import typing
        shadow = self.shadow()
        req = completion_request(suffix=None, echo=False, stream=True, best_of=None,
                                 model="model", priority=0)
        params = native.params(max_tokens=2048)
        req.to_sampling_params = lambda *a: params
        namespace = dict(native.N, cast=typing.cast, Union=typing.Union,
            EmbedsPrompt=dict, TokensPrompt=dict, SamplingParams=NS,
            BeamSearchParams=type("Beam", (), {}),
            RequestResponseMetadata=lambda **kw: NS(**kw),
            is_embeds_prompt=lambda p: "prompt_embeds" in p,
            is_tokens_prompt=lambda p: "prompt_token_ids" in p,
            merge_async_iterators=lambda *args: object())
        ns = dict(namespace, OpenAIServing=object)
        native.load_defs("vllm/entrypoints/openai/serving_completion.py",
                         ["OpenAIServingCompletion"], ns)
        cls = ns["OpenAIServingCompletion"]
        obj = cls.__new__(cls)
        obj.engine_client = NS(errored=False, r1_shadow=shadow,
                               generate=Mock(return_value=object()))
        async def tokenizer(*a): return None
        async def check(*a): return None
        async def preprocess(*a, **kw):
            return ([{"prompt": req.prompt, "prompt_token_ids": [1, 2]}],
                    [{"prompt_token_ids": [1, 2]}])
        obj.engine_client.get_tokenizer = tokenizer
        obj._check_model = check
        obj._base_request_id = lambda _: "ordinary-curl"
        obj._maybe_get_adapters = lambda _: (None, None)
        obj._preprocess_completion = preprocess
        obj._log_inputs = Mock()
        obj._get_model_name = lambda *a: "model"
        obj.completion_stream_generator = Mock(return_value="stream")
        obj.max_model_len = 8192
        obj.default_sampling_params = {}
        obj.model_config = NS(logits_processor_pattern=None)
        self.assertEqual(native.run_async(obj.create_completion(req)), "stream")
        self.assertEqual(len(shadow.predictor.static_calls), 1)
        self.assertEqual(obj.engine_client.generate.call_args.args[2], "cmpl-ordinary-curl-0")

    def test_numeric_runtime_matches_reference_feature_order_and_postprocessing(self):
        import numpy as np
        predictor = S.Predictor.__new__(S.Predictor)
        predictor.np = np
        class Model:
            def __init__(self, offset): self.offset = offset
            def predict(self, x):
                self.x = x.copy()
                return x[:, 0] * .5 + self.offset
        models = [Model(5), Model(-8), Model(2)]
        predictor.head = {"models": models}
        candidates = [request("a"), request("b")]
        candidates[1].num_prompt_tokens = 7
        x = predictor.assemble(candidates, 4, .2)
        self.assertEqual(x.dtype, np.float64)
        np.testing.assert_array_equal(x, [[3, 4, .2, 1, 2, 3], [7, 4, .2, 1, 2, 3]])
        actual = predictor.dynamic(x)
        raw = np.column_stack([m.predict(x) for m in models])
        np.testing.assert_allclose(actual, np.sort(np.maximum(raw, 0), axis=1), rtol=1e-10, atol=1e-10)
        for i in range(len(x)):
            np.testing.assert_array_equal(predictor.dynamic(x[i:i+1])[0], actual[i])

    def test_actual_static_numeric_path_caps_after_clipping_sorting(self):
        import numpy as np
        predictor = S.Predictor.__new__(S.Predictor)
        predictor.np = np
        vectorizer = NS(transform=Mock(return_value="sparse-placeholder"))
        svd = NS(transform=Mock(return_value=np.arange(32, dtype=np.float64)[None, :]))
        predictor.encoder = dict(vectorizer=vectorizer, svd=svd)
        models = [NS(predict=Mock(return_value=np.array([v]))) for v in (3000, -2, 9)]
        predictor.head = dict(models=models)
        values, timings = predictor.static(" x\n  y ", 17)
        self.assertEqual(values, (0., 9., 2048.))
        vectorizer.transform.assert_called_once_with([" x\n  y "])
        x = models[0].predict.call_args.args[0]
        self.assertEqual(x.dtype, np.float64)
        np.testing.assert_array_equal(x, [[17, *range(32)]])
        self.assertGreaterEqual(timings["text_transform_ns"], 0)

    def test_model_service_configuration_gate(self):
        config = NS(model_config=NS(model="/models/Meta-Llama-3-8B-Instruct",
            tokenizer="/models/Meta-Llama-3-8B-Instruct",
            hf_config=NS(model_type="llama", hidden_size=4096,
                         num_hidden_layers=32, vocab_size=128256),
            is_multimodal_model=False, is_encoder_decoder=False, encoder_config=None),
            kv_transfer_config=None, speculative_config=None,
            parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1,
                               tensor_parallel_size=1),
            scheduler_config=NS(policy="fcfs",
                scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler"))
        S.validate_service(config)
        config.model_config.hf_config.hidden_size = 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            S.validate_service(config)

    def test_preempted_round_does_not_sweep(self):
        r = request("running", native.N["RequestStatus"].RUNNING)
        obj = native.scheduler([r], [request("waiting")])
        obj.r1_shadow = self.shadow()
        obj.kv_cache_manager.fail.add("running")
        obj.schedule()
        self.assertEqual(obj.r1_shadow.batch, 0)

    def test_real_frontend_core_shutdown_close_shadow(self):
        namespace = dict(native.N, EngineClient=object,
            UsageContext=NS(ENGINE_CONTEXT=0), MULTIMODAL_REGISTRY=None,
            shutdown_prometheus=lambda: None)
        native.load_defs("vllm/v1/engine/async_llm.py", ["AsyncLLM"], namespace)
        obj = namespace["AsyncLLM"].__new__(namespace["AsyncLLM"])
        obj.r1_shadow = self.shadow()
        obj.engine_core = NS(shutdown=Mock())
        obj.shutdown()
        self.assertFalse(obj.r1_shadow.writer._thread.is_alive())
        ns = dict(native.N, MULTIMODAL_REGISTRY=None)
        native.load_defs("vllm/v1/engine/core.py", ["EngineCore"], ns)
        core = ns["EngineCore"].__new__(ns["EngineCore"])
        shadow = self.shadow()
        core.scheduler = NS(r1_shadow=shadow, shutdown=Mock())
        core.structured_output_manager = NS(clear_backend=Mock())
        core.model_executor = NS(shutdown=Mock())
        core.shutdown()
        self.assertFalse(shadow.writer._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
