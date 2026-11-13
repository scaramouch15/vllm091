"""CPU test shim: only unused vLLM multimodal/LoRA imports are replaced."""
import sys, types, typing
for name in ('vllm', 'vllm.lora', 'vllm.lora.request', 'vllm.lora.utils',
             'vllm.multimodal', 'vllm.multimodal.image', 'vllm.transformers_utils',
             'vllm.transformers_utils.tokenizer'):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
sys.modules['vllm.lora.request'].LoRARequest = type('LoRARequest', (), {})
sys.modules['vllm.lora.utils'].get_adapter_absolute_path = lambda x: x
sys.modules['vllm.multimodal'].MultiModalDataDict = dict
sys.modules['vllm.multimodal.image'].convert_image_mode = lambda image, mode: image.convert(mode)
sys.modules['vllm.transformers_utils.tokenizer'].AnyTokenizer = typing.Any
sys.modules['vllm.transformers_utils.tokenizer'].get_lora_tokenizer = lambda *args, **kwargs: None
