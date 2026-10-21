# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local raw-source requests for the original benchmark_serving.py entry."""

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchmark_dataset import BenchmarkDataset, SampleRequest

TASKS = ("chat", "code", "summary")
DATASETS = {
    "chat": "lmsys/lmsys-chat-1m",
    "code": "m-a-p/CodeFeedback-Filtered-Instruction",
    "summary": "abisee/cnn_dailymail",
}
REQUIRED_FIELDS = {
    "chat": ("conversation_id", "language", "conversation"),
    "code": ("query", "resource", "lang"),
    "summary": ("id", "article"),
}
SUMMARY_PREFIX = (
    "Summarize the following news article, covering its main points.\n\nArticle:\n"
)


def add_mixed_dataset_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("mixed raw dataset options")
    for task in TASKS:
        group.add_argument(
            f"--mixed-{task}-path", help=f"Explicit local raw {DATASETS[task]} path."
        )
        group.add_argument(
            f"--mixed-{task}-count",
            type=int,
            default=0,
            help=f"Number of {task} requests; three counts must sum to num-prompts.",
        )
    group.add_argument(
        "--mixed-output-len",
        type=int,
        default=None,
        help="Positive fixed max_tokens; otherwise use each reference's token count.",
    )
    group.add_argument(
        "--mixed-dry-run",
        action="store_true",
        help="Prepare and inspect mixed data, then return before any server request.",
    )
    group.add_argument(
        "--mixed-export-path",
        default=None,
        help="Optional JSON inspection manifest, written before sending requests.",
    )


def mixed_dataset_from_args(args: argparse.Namespace) -> "MixedWorkloadDataset":
    if args.backend not in ("vllm", "openai") or args.endpoint != "/v1/completions":
        raise ValueError(
            "mixed supports --backend vllm/openai with --endpoint /v1/completions"
        )
    dataset = MixedWorkloadDataset(
        **{f"{task}_path": getattr(args, f"mixed_{task}_path") for task in TASKS},
        **{f"{task}_count": getattr(args, f"mixed_{task}_count") for task in TASKS},
        output_len=args.mixed_output_len,
        random_seed=args.seed,
    )
    dataset.validate_num_requests(args.num_prompts)
    return dataset


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class MixedSampleRequest(SampleRequest):
    metadata: dict[str, Any] = field(default_factory=dict)


class MixedWorkloadDataset(BenchmarkDataset):
    """LMSYS train, CodeFeedback train and CNN/DM 3.0.0 validation.

    Task labels identify source groups, including unreviewed chat candidates.
    Internal source whitespace is preserved. No model context filter is applied.
    """

    def __init__(
        self,
        *,
        chat_path: Optional[str] = None,
        code_path: Optional[str] = None,
        summary_path: Optional[str] = None,
        chat_count: int = 0,
        code_count: int = 0,
        summary_count: int = 0,
        output_len: Optional[int] = None,
        random_seed: int = BenchmarkDataset.DEFAULT_SEED,
    ) -> None:
        super().__init__(random_seed=random_seed)
        self.paths = dict(zip(TASKS, (chat_path, code_path, summary_path)))
        self.counts = dict(zip(TASKS, (chat_count, code_count, summary_count)))
        for task, count in self.counts.items():
            if type(count) is not int or count < 0:
                raise ValueError(f"--mixed-{task}-count must be nonnegative")
            if count and not self.paths[task]:
                raise ValueError(f"--mixed-{task}-path is required for count > 0")
        if output_len is not None and (type(output_len) is not int or output_len <= 0):
            raise ValueError("--mixed-output-len must be a positive integer or omitted")
        self.output_len = output_len
        self.files: dict[str, list[dict]] = {}
        self.stats: dict[str, dict] = {}
        self.requests: list[MixedSampleRequest] = []
        self.tokenizer_info: dict = {}

    def validate_num_requests(self, num_requests: int) -> None:
        if type(num_requests) is not int or num_requests <= 0:
            raise ValueError("mixed --num-prompts must be positive")
        if sum(self.counts.values()) != num_requests:
            raise ValueError(
                f"mixed counts {self.counts} must sum to --num-prompts={num_requests}"
            )

    def load_data(self) -> None:
        """Resolve the small set of supported raw layouts in stable file order."""
        self.files = {task: [] for task in TASKS}
        for task in TASKS:
            if not self.counts[task]:
                continue
            path = Path(self.paths[task]).expanduser().resolve()
            if not path.exists():
                raise ValueError(f"{task}: local source does not exist: {path}")
            if task == "chat":
                directory = path / "data" if (path / "data").is_dir() else path
                files = (
                    [path]
                    if path.is_file()
                    else list(directory.glob("train-*.parquet"))
                )
                valid = lambda p: p.suffix == ".parquet" and p.name.startswith("train-")
            elif task == "code":
                files = (
                    [path]
                    if path.is_file()
                    else [path / "CodeFeedback-Filtered-Instruction.jsonl"]
                )
                valid = lambda p: p.suffix == ".jsonl"
            else:
                directory = path / "3.0.0" if (path / "3.0.0").is_dir() else path
                files = (
                    [path]
                    if path.is_file()
                    else list(directory.glob("validation-*.parquet"))
                )
                valid = lambda p: (
                    p.parent.name == "3.0.0"
                    and p.suffix == ".parquet"
                    and p.name.startswith("validation-")
                )
            if not files or any(not p.is_file() or not valid(p) for p in files):
                raise ValueError(
                    f"{task}: unsupported raw source layout: {path}; expected "
                    "LMSYS data/train-*.parquet, CodeFeedback original .jsonl, "
                    "or CNN/DM 3.0.0/validation-*.parquet, respectively"
                )
            base = path.parent if path.is_file() else path
            for file in sorted(files, key=lambda p: p.relative_to(base).as_posix()):
                self.files[task].append(
                    {
                        "file_id": file.relative_to(base).as_posix(),
                        "source_file": str(file),
                        "size_bytes": file.stat().st_size,
                    }
                )

    @staticmethod
    def _require(row: dict, fields: tuple[str, ...], location: str) -> None:
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected a raw record object")
        missing = [name for name in fields if name not in row]
        if missing:
            raise ValueError(f"{location}: missing required fields/columns: {missing}")

    def _rows(self, task: str, file: dict):
        path = Path(file["source_file"])
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{number}: invalid JSON: {exc}"
                        ) from exc
                    self._require(row, REQUIRED_FIELDS[task], f"{path}:{number}")
                    yield number, row
        else:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            columns = parquet.schema_arrow.names
            self._require(dict.fromkeys(columns), REQUIRED_FIELDS[task], str(path))
            # Do not materialize unrelated columns such as moderation payloads.
            selected = list(REQUIRED_FIELDS[task])
            if (
                task == "summary"
                and self.output_len is None
                and "highlights" in columns
            ):
                selected.append("highlights")
            number = 0
            for batch in parquet.iter_batches(batch_size=1024, columns=selected):
                for row in batch.to_pylist():
                    number += 1
                    yield number, row

    def _exclude(self, task: str, reason: str):
        self.stats[task]["excluded"] += 1
        self.stats[task]["exclusion_reasons"][reason] += 1
        return None

    @staticmethod
    def _duplicate(seen, digest: str, task: str) -> bool:
        # Keep all source types as a bit mask, even for within-type duplicates.
        task_bit = 1 << TASKS.index(task)
        previous = seen.get(digest, 0)
        seen[digest] = previous | task_bit
        return bool(previous & task_bit)

    def _input(self, task: str, row: dict, location: str):
        for key in REQUIRED_FIELDS[task]:
            if key != "conversation" and not isinstance(row[key], str):
                return self._exclude(task, f"invalid_type:{key}")
        if task == "chat":
            self.stats[task]["language_values"][row["language"]] += 1
            conversation = row["conversation"]
            if not isinstance(conversation, list):
                return self._exclude(task, "invalid_type:conversation")
            if not conversation:
                return self._exclude(task, "empty_conversation")
            if not isinstance(conversation[0], dict):
                return self._exclude(task, "invalid_type:first_message")
            first = conversation[0]
            self._require(first, ("role", "content"), location + ":conversation[0]")
            if not isinstance(first["role"], str):
                return self._exclude(task, "invalid_type:first_role")
            if not isinstance(first["content"], str):
                return self._exclude(task, "invalid_type:first_content")
            if row["language"] != "English":
                return self._exclude(task, "language_mismatch")
            if first["role"] != "user":
                return self._exclude(task, "first_role_not_user")
            text, sample_id = first["content"], row["conversation_id"]
            extra = {
                "language": row["language"],
                "chat_review_status": "unreviewed_candidate",
            }
        elif task == "code":
            if row["lang"].strip().lower() != "python":
                return self._exclude(task, "language_mismatch")
            text, sample_id = row["query"], None
            extra = {"resource": row["resource"], "lang": row["lang"]}
        else:
            text, sample_id = row["article"], row["id"]
            extra = {"source_config": "3.0.0"}
        if not text.strip():
            return self._exclude(task, "empty_input")
        if sample_id is not None and not sample_id.strip():
            return self._exclude(task, "empty_source_id")
        return text.strip(), sample_id, extra

    def _max_tokens(self, task: str, row: dict, tokenizer) -> Optional[int]:
        if self.output_len is not None:
            return self.output_len
        if task == "chat":
            conversation = row["conversation"]
            if len(conversation) < 2:
                return self._exclude(task, "missing_reference")
            assistant = conversation[1]
            if not isinstance(assistant, dict):
                return self._exclude(task, "invalid_type:reference_message")
            if assistant.get("role") != "assistant":
                return self._exclude(task, "reference_not_assistant")
            reference = assistant.get("content")
        else:
            reference = row.get("answer" if task == "code" else "highlights")
        if reference is None:
            return self._exclude(task, "missing_reference")
        if not isinstance(reference, str):
            return self._exclude(task, "invalid_type:reference")
        if not reference.strip():
            return self._exclude(task, "empty_reference")
        # Match native ShareGPTDataset: tokenize the original reference using
        # tokenizer defaults, including its default special-token handling.
        length = len(tokenizer(reference).input_ids)
        if length <= 0:
            return self._exclude(task, "nonpositive_reference_tokens")
        return length

    def sample(self, tokenizer, num_requests: int) -> list[MixedSampleRequest]:
        self.validate_num_requests(num_requests)
        try:
            template = tokenizer.get_chat_template()
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                "mixed requires an available tokenizer chat template"
            ) from exc
        if not template:
            raise ValueError("mixed requires an available tokenizer chat template")
        self.requests = []
        self.tokenizer_info = {}
        self.load_data()
        self.stats = {
            task: {
                "read": 0,
                "excluded": 0,
                "duplicate": 0,
                "eligible": 0,
                "sampled": 0,
                "exclusion_reasons": Counter(),
                "language_values": Counter(),
            }
            for task in TASKS
        }
        seen_inputs: dict[str, int] = {}
        seen_prompts: dict[str, int] = {}
        input_duplicates: Counter = Counter()
        # Discover conflicts before any sampling. Spool prompts instead of
        # retaining the full raw datasets in memory or rendering them twice.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as candidates:
            for task in TASKS:
                for file in self.files[task]:
                    for number, row in self._rows(task, file):
                        self.stats[task]["read"] += 1
                        location = f"{file['source_file']}:{number}"
                        parsed = self._input(task, row, location)
                        if parsed is None:
                            continue
                        text, sample_id, extra = parsed
                        max_tokens = self._max_tokens(task, row, tokenizer)
                        if max_tokens is None:
                            continue
                        input_hash = _hash(text)
                        if self._duplicate(seen_inputs, input_hash, task):
                            input_duplicates[task, input_hash] += 1
                            continue
                        body = SUMMARY_PREFIX + text if task == "summary" else text
                        prompt = tokenizer.apply_chat_template(
                            [{"role": "user", "content": body}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        if not isinstance(prompt, str) or not prompt:
                            raise ValueError(
                                f"{location}: chat template must render text"
                            )
                        prompt_hash = _hash(prompt)
                        self._duplicate(seen_prompts, prompt_hash, task)
                        metadata = {
                            "source_sample_id": sample_id
                            if sample_id is not None
                            else f"{file['file_id']}:row:{number}",
                            "source_dataset": DATASETS[task],
                            "task_type": task,
                            "source_split": "validation"
                            if task == "summary"
                            else "train",
                            "prompt_hash": prompt_hash,
                            "input_hash": input_hash,
                            "prompt": prompt,
                            "prompt_len": len(
                                tokenizer(prompt, add_special_tokens=False).input_ids
                            ),
                            "max_tokens": max_tokens,
                            "source_file": file["source_file"],
                            "file_id": file["file_id"],
                            "source_row": number,
                            **extra,
                        }
                        candidates.write(
                            json.dumps(metadata, ensure_ascii=False) + "\n"
                        )

            # More than one set bit means this hash occurs in multiple types.
            conflicting_inputs = {
                digest for digest, mask in seen_inputs.items() if mask & (mask - 1)
            }
            conflicting_prompts = {
                digest for digest, mask in seen_prompts.items() if mask & (mask - 1)
            }
            del seen_inputs, seen_prompts
            rngs = {
                task: random.Random(f"{self.random_seed}:mixed:sample:{task}")
                for task in TASKS
            }
            reservoirs: dict[str, list[MixedSampleRequest]] = {t: [] for t in TASKS}
            eligible_prompts: set[str] = set()
            candidates.seek(0)
            for line in candidates:
                metadata = json.loads(line)
                task = metadata["task_type"]
                copies = 1 + input_duplicates[task, metadata["input_hash"]]
                if (
                    metadata["input_hash"] in conflicting_inputs
                    or metadata["prompt_hash"] in conflicting_prompts
                ):
                    # Conflicts take precedence over same-type deduplication:
                    # count every valid source row once, including its copies.
                    self.stats[task]["excluded"] += copies
                    self.stats[task]["exclusion_reasons"]["cross_type_conflict"] += (
                        copies
                    )
                    continue
                self.stats[task]["duplicate"] += copies - 1
                if metadata["prompt_hash"] in eligible_prompts:
                    self.stats[task]["duplicate"] += 1
                    continue
                eligible_prompts.add(metadata["prompt_hash"])
                self.stats[task]["eligible"] += 1
                request = MixedSampleRequest(
                    prompt=metadata["prompt"],
                    prompt_len=metadata["prompt_len"],
                    expected_output_len=metadata["max_tokens"],
                    metadata=metadata,
                )
                reservoir = reservoirs[task]
                if len(reservoir) < self.counts[task]:
                    reservoir.append(request)
                else:
                    index = rngs[task].randrange(self.stats[task]["eligible"])
                    if index < self.counts[task]:
                        reservoir[index] = request
        for task in TASKS:
            reservoir = reservoirs[task]
            if len(reservoir) < self.counts[task]:
                raise ValueError(
                    f"mixed {task}: insufficient valid unique samples; requested "
                    f"{self.counts[task]}, available {len(reservoir)}; "
                    f"stats={self.stats[task]}"
                )
        for task in TASKS:
            reservoir = reservoirs[task]
            self.requests.extend(reservoir)
            self.stats[task]["sampled"] = len(reservoir)
        random.Random(f"{self.random_seed}:mixed:shuffle").shuffle(self.requests)
        for index, request in enumerate(self.requests):
            request.metadata["request_index"] = index
        self.tokenizer_info = {
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "class": type(tokenizer).__name__,
            "chat_template_sha256": _hash(template),
            "add_special_tokens": False,
        }
        return self.requests

    @staticmethod
    def _distribution(values: list[int]) -> dict:
        if not values:
            return {"count": 0}
        values = sorted(values)

        def percentile(p):
            position = (len(values) - 1) * p
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)

        return {
            "count": len(values),
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / len(values),
            **{f"p{p}": percentile(p / 100) for p in (50, 90, 95, 99)},
        }

    def report(self) -> dict:
        groups = {
            "all": self.requests,
            **{
                task: [
                    request
                    for request in self.requests
                    if request.metadata["task_type"] == task
                ]
                for task in TASKS
            },
        }
        return {
            "seed": self.random_seed,
            "counts": self.counts,
            "files": self.files,
            "output_length_mode": "reference_tokens"
            if self.output_len is None
            else "fixed",
            "mixed_output_len": self.output_len,
            "tokenizer": self.tokenizer_info,
            "source_configs": {
                "chat": "default/train",
                "code": "train",
                "summary": "3.0.0/validation",
            },
            "chat_english_language_value": "English",
            "stats": self.stats,
            "distributions": {
                name: {
                    "prompt_len": self._distribution([r.prompt_len for r in requests]),
                    "max_tokens": self._distribution(
                        [r.expected_output_len for r in requests]
                    ),
                }
                for name, requests in groups.items()
            },
        }

    def print_inspection(self) -> None:
        print(json.dumps(self.report(), ensure_ascii=False, indent=2))
        for task in TASKS:
            example = next(
                (r for r in self.requests if r.metadata["task_type"] == task), None
            )
            if example is not None:
                print(
                    f"\n--- {task} final prompt "
                    f"(max_tokens={example.expected_output_len}) ---"
                )
                print(example.prompt)
        print(
            "Dry-run checks data construction only; "
            "server context limits were not checked."
        )

    def export(
        self, output_path: str, *, tokenizer_id: str, tokenizer_mode: str
    ) -> None:
        path = Path(output_path).expanduser().resolve()
        if any(
            path == Path(file["source_file"])
            for files in self.files.values()
            for file in files
        ):
            raise ValueError("mixed manifest must not overwrite a raw source")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": {
                **self.report(),
                "tokenizer_id": tokenizer_id,
                "tokenizer_mode": tokenizer_mode,
            },
            "requests": [request.metadata for request in self.requests],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
