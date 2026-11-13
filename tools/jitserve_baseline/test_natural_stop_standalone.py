import importlib.util, sys, types, unittest
from pathlib import Path
for name in ('vllm','vllm.lora','vllm.lora.request','vllm.sampling_params','vllm.sequence','vllm.transformers_utils','vllm.transformers_utils.tokenizer'):
    sys.modules[name] = types.ModuleType(name)
sys.modules['vllm.lora.request'].LoRARequest = object
sys.modules['vllm.sampling_params'].SamplingParams = object
sys.modules['vllm.transformers_utils.tokenizer'].AnyTokenizer = object
class Status:
    FINISHED_LENGTH_CAPPED = 'length'
    FINISHED_STOPPED = 'stop'
sys.modules['vllm.sequence'].Sequence = object
sys.modules['vllm.sequence'].SequenceStatus = Status
spec = importlib.util.spec_from_file_location('stop_tested', Path(__file__).resolve().parents[1] / 'third_party/vllm/vllm/engine/output_processor/stop_checker.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
class Seq:
    eos_token_id = 42
    output_text = 'x'
    def __init__(self, n, last, target):
        self.n, self.last, self.real_output_len = n, last, target
        self.data = types.SimpleNamespace(_output_token_ids=[last])
        self.status = self.stop_reason = None
    def get_output_len(self): return self.n
    def get_len(self): return self.n + 10
    def get_last_token_id(self): return self.last
class Stops(unittest.TestCase):
    def run_one(self, n, last, target):
        seq = Seq(n, last, target)
        params = types.SimpleNamespace(min_tokens=0, ignore_eos=False, include_stop_str_in_output=False, stop_token_ids=[], stop=[], max_tokens=4)
        module.StopChecker(100, None).maybe_stop_sequence(seq, 0, params)
        return seq
    def test_natural_eos(self): self.assertEqual(self.run_one(2, 42, None).status, 'stop')
    def test_natural_cap(self): self.assertEqual(self.run_one(4, 1, None).status, 'length')
    def test_fixed_replay(self):
        early = self.run_one(2, 42, 3)
        self.assertIsNone(early.status)
        self.assertEqual(early.data._output_token_ids[-1], 43)
        self.assertEqual(self.run_one(3, 1, 3).status, 'stop')
if __name__ == '__main__': unittest.main()
