"""Offline CPU interface probe for the authors' OPT-125M ranking checkpoint.

This tool does not integrate with Scheduler, start a service, or download files.
Use only locally obtained, trusted safetensors and tokenizer files.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path

CHECKPOINT_SHA256 = 'e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab'
CHECKPOINT_REVISION = '39df2b41ffe88d5ed967c6035d3838b5b5960379'
TOKENIZER_REVISION = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
SAMPLES = ('Say hello.', 'Explain why the sky is blue in three sentences.',
           'Write a detailed step-by-step guide to sorting a list of numbers.')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_descending(scores):
    if not scores or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
        raise ValueError('scores must be nonempty finite numbers')
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic-token-ids', action='store_true',
                        help='check model weights without testing tokenizer')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; use a new path')
    weights = args.model_dir / 'model.safetensors'
    actual = sha256_file(weights)
    if actual != CHECKPOINT_SHA256:
        raise ValueError('checkpoint SHA256 mismatch')
    config = json.loads((args.model_dir / 'config.json').read_text(encoding='utf-8'))
    if config.get('architectures') != ['OPTForSequenceClassification'] or config.get('num_labels') != 1:
        raise ValueError('unexpected model schema')
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    started = time.perf_counter_ns()
    tokenizer = (None if args.synthetic_token_ids else AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False,
        use_safetensors=True).to('cpu').eval()
    load_ns = time.perf_counter_ns() - started
    start = time.perf_counter_ns()
    if args.synthetic_token_ids:
        ids = torch.tensor([[2, 123, 234, 1], [2, 123, 234, 345], [2, 123, 234, 1]])
        inputs = {'input_ids': ids, 'attention_mask': (ids != 1).long()}
    else:
        inputs = tokenizer(list(SAMPLES), max_length=2048, padding=True,
                           truncation=True, return_tensors='pt')
    tokenization_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    with torch.inference_mode():
        logits = model(**inputs).logits
    inference_ns = time.perf_counter_ns() - start
    if tuple(logits.shape) != (len(SAMPLES), 1):
        raise ValueError('unexpected logits shape: ' + str(tuple(logits.shape)))
    scores = [float(x) for x in logits[:, 0].tolist()]
    order = stable_descending(scores)
    # A synthetic tied score checks Python stable ordering, not model accuracy.
    assert stable_descending([1., 3., 3., -1.]) == [1, 2, 0, 3]
    result = {'checkpoint_sha256': actual, 'checkpoint_revision': CHECKPOINT_REVISION,
              'tokenizer_revision_expected': TOKENIZER_REVISION,
              'model_config_sha256': sha256_file(args.model_dir / 'config.json'),
              'tokenizer_files_sha256': {p.name: sha256_file(p) for p in sorted(args.tokenizer_dir.iterdir()) if p.is_file()},
              'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'safetensors': importlib.metadata.version('safetensors'),
              'device': 'cpu', 'input_kind': ('synthetic_token_ids' if args.synthetic_token_ids else 'text'),
              'max_length': 2048, 'scores': scores, 'order_descending': order,
              'sample_count': len(SAMPLES), 'load_ns': load_ns, 'tokenization_ns': tokenization_ns,
              'inference_ns': inference_ns, 'limitation': 'interface probe, not accuracy or online overhead'}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
