# 最小混合数据集扩展

本批在原仓库、`research-v0.9.1` 分支完成；HEAD 与 `v0.9.1` 均为 `b6553be1bc75f046b00046a4ad7576364d03c835`。开始时只有用户新增的未跟踪 `AGENTS.md`，该文件保留未改。没有创建 worktree、恢复 stash、回滚、提交或推送。

## 修改与调用链

| 文件 | 改动及理由 |
| --- | --- |
| `benchmarks/mixed_workload_dataset.py`（新增） | 直接导入原生 BenchmarkDataset / SampleRequest；集中实现原始文件读取、字段校验、长度解析、去重、抽样、元数据和检查清单 |
| `benchmarks/benchmark_serving.py` | 增加 mixed 分发、参数、同入口 dry-run；用既有 extra_body 传递特殊 token 设置 |
| `tests/benchmarks/test_mixed_workload_dataset.py`（新增） | CPU 小样例单测；隔离未安装的依赖，明确不作为真实模型/服务端验证 |
| `benchmarks/MIXED_WORKLOAD.md`（新增） | 参数、数据语义、命令和验证说明 |

实际运行路径：

```text
benchmark_serving.py 原生 CLI / main
  -> mixed_dataset_from_args
  -> MixedWorkloadDataset.sample（继承 benchmark_dataset.py 的原生接口）
  -> 原生 benchmark（初始探测与正式批次）
  -> 原生 RequestFuncInput.output_len
  -> 原生 async_request_openai_completions -> payload.max_tokens
```

本地源码确认 `vllm`、`openai` 均映射到同一 `async_request_openai_completions`。mixed 支持这两个 backend 与 `/v1/completions` 的组合；其他组合对 mixed 报错，不限制其他数据集。

原生 `benchmark_dataset.py` 的类位置和构造接口完全保留；没有新增基类模块或独立 benchmark 入口。`get_request()`、`benchmark()`、`calculate_metrics()`、HTTP/SSE 发送解析、结果格式、`vllm/benchmarks/` 和 `vllm/v1/` 均未修改。

## 原始数据与 prompt

| 类型 | 本地路径与格式 | 使用规则 |
| --- | --- | --- |
| chat | LMSYS 根目录、其 `data/` 目录，或单个 `train-*.parquet` | train；`language == "English"`；首条 user.content；conversation_id 作为原始 ID |
| code | 原始 `.jsonl`，或包含 `CodeFeedback-Filtered-Instruction.jsonl` 的目录 | train；`lang.strip().lower() == "python"`；输入 query；保留原始 resource、lang |
| summary | CNN/DailyMail 根目录、`3.0.0/`，或该目录下单个 `validation-*.parquet` | 固定 3.0.0 / validation；输入 article；保留 id |

发布文件格式依据 [LMSYS 数据卡](https://huggingface.co/datasets/lmsys/lmsys-chat-1m/blob/main/README.md)、[CodeFeedback 原始文件目录](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/tree/main)和 [CNN/DailyMail 3.0.0 目录](https://huggingface.co/datasets/abisee/cnn_dailymail/tree/main/3.0.0)核对。没有读取其他数据源或统一 prompt JSONL。

本机未发现真实数据；LMSYS 官方预览也未能访问，因此**真实文件的 language 取值尚未验证**。实现明确按 `English` 匹配，不猜测 `en` 等别名；dry-run 会输出 `stats.chat.language_values` 和语言不匹配计数供核对。chat 标签只代表来源分组，元数据明确为 `unreviewed_candidate`，不表示已排除代码或摘要任务。

源输入只去除首尾空白，保留内部换行、缩进和代码块。summary 正文固定为：

```text
Summarize the following news article, covering its main points.

Article:
{article}
```

三类均作为单条 user 消息，使用同一个目标 tokenizer 的聊天模板，`tokenize=False, add_generation_prompt=True`。模板只应用一次，缺失时明确报错；参考回答不进入 prompt。

最终 prompt 按 `add_special_tokens=False` 计长。原生 Completions 默认会再次添加特殊 token，因此 mixed 在既有 extra_body 中补入 `add_special_tokens=False`，原生初始探测与正式请求共用该设置；temperature、top_p、ignore_eos 等原有参数继续沿用。

## 参数与输出上限

| 参数 | 来源与语义 |
| --- | --- |
| `--dataset-name mixed` | 原生选项新增一个取值 |
| `--num-prompts`、`--seed` | 原生参数；总数与主种子，默认值不变 |
| `--model`、`--tokenizer`、`--backend`、`--endpoint` | 原生参数，原有语义不变；model 仍必填 |
| `--mixed-chat-path` / `--mixed-code-path` / `--mixed-summary-path` | 新增，显式本地路径；数量为 0 的来源无需填写，也不读取 |
| `--mixed-chat-count` / `--mixed-code-count` / `--mixed-summary-count` | 新增，非负整数，默认 0；之和必须等于正数 num-prompts |
| `--mixed-output-len` | 新增，`type=int, default=None`；省略时逐条取参考回答 token 数；显式值必须为正整数 |
| `--mixed-dry-run` | 新增，数据准备后、任何 benchmark 调用前返回 |
| `--mixed-export-path` | 新增，可选；在发送前写出单个 JSON 检查清单，dry-run 和正式运行共用 |

默认模式的参考回答分别是首条 user 后**紧随的** assistant.content、answer、highlights。计长与当前原生 ShareGPTDataset 一致：`len(tokenizer(reference).input_ids)`，使用 tokenizer 默认特殊 token 行为，按原始参考文本计长；仅用 strip 检查是否空白。缺失、非字符串、空白或计长非正时排除并分类计数。

显式覆盖模式不要求参考回答存在，也不依据它筛选；所有请求使用指定的正整数。最终 expected_output_len 一定是具体正整数，沿原生路径变成 max_tokens。**参考答案 token 数只是默认生成上限的依据；生成上限不是实际输出长度，也不是预测结果。** 该默认模式不代表无限生成或自然完成长度。

没有新增 `--mixed-context-len`，不使用 tokenizer.model_max_length，不套用 is_valid_sequence 的 1024/2048 采样阈值，不截断或逐条缩小上限。请求上下文合法性由服务端校验；dry-run 成功仅证明数据构造成功。

## 抽样、计数与检查清单

文件按相对路径排序。输入类型错误、空输入、语言不符、缺失/非法参考回答分别计入 exclusion_reasons；结构不支持、缺少必需字段或列（包括空 Parquet 文件）、非法 JSON 报出来源位置。

对首尾处理后的源输入及最终 prompt 做精确 SHA256 去重，不归一化内部空白、大小写或 Unicode。跨来源发现相同输入但类型冲突时报告两处位置并失败。有效唯一候选不足时报错，不调用原生重复补齐。

每类使用由 seed 派生的局部 RNG 做无放回 reservoir 抽样，再用独立局部 RNG 混排整个样本对象；不会消耗全局 Python/NumPy RNG。扫描所有启用来源，但只保留所需数量的最终 prompt；去重索引仍随唯一候选数量增长。复现需保持文件内容及路径集合、tokenizer/模板/设置、数量、输出长度模式、seed 和 Python 随机实现一致。

每个请求的 metadata 保存混排后的 request_index、source_sample_id、source_dataset、task_type、prompt_hash、input_hash、最终 prompt、prompt_len、解析后的 max_tokens、source_split、来源文件与从 1 开始的原始行号，以及 code 的 resource/lang 和 chat 审核状态。code 无原生 ID，以文件标识和行号定位。它们只是客户端标识，未关联服务端 request_id。

清单为 `{"run": {...}, "requests": [...]}`；run 记录 seed、数量、文件清单、来源配置、tokenizer 标识/类/模板指纹、输出长度模式、计数和长度分布；requests 与混排后的请求逐条对应。它只用于检查追溯，没有回放、版本平台或训练划分功能。

## 同入口 dry-run 命令

在仓库根目录使用已具备原生依赖的环境，替换以下路径占位符。tokenizer 必须为本地目录（也可省略 tokenizer 并令 model 指向本地目录）；dry-run 额外传入 local_files_only，不下载模型或 tokenizer，不查询推理服务、不加载权重、不启动服务、不做 GPU 推理。正常原生依赖导入仍保留。

```bash
python benchmarks/benchmark_serving.py \
  --model "YOUR_INSTRUCT_MODEL_NAME" \
  --tokenizer "/ABS/PATH/TO/LOCAL_TOKENIZER" \
  --backend vllm --endpoint /v1/completions \
  --dataset-name mixed --num-prompts 6 --seed 42 \
  --mixed-chat-path "/ABS/PATH/TO/LMSYS_CHAT_1M" \
  --mixed-code-path "/ABS/PATH/TO/CodeFeedback-Filtered-Instruction.jsonl" \
  --mixed-summary-path "/ABS/PATH/TO/CNN_DAILYMAIL/3.0.0" \
  --mixed-chat-count 2 --mixed-code-count 2 --mixed-summary-count 2 \
  --mixed-dry-run --mixed-export-path "./mixed-inspection.json"
```

此命令采用参考长度模式；增加 `--mixed-output-len 256` 即为显式覆盖模式，数值可修改。dry-run 输出各类读取/排除/重复/有效/抽样计数，整体与分类的输入长度、输出上限分布，并各展示一条完整最终 prompt。删除 mixed-dry-run 后仍沿同一数据准备路径进入原生 benchmark；本批未执行真实运行，也未制作 bash 实验脚本。

## 验证与未验证事项

- **13 项轻量单测通过**：实际 Parquet/JSONL 小样例，三类映射与答案隔离，两种输出模式及覆盖时缺参考，模板/代码空白，真实 HF 合成 tokenizer 的 BOS 对照，数量校验、无放回与复现、全局 RNG 不变、去重/冲突、错误定位、元数据/导出顺序、同入口 dry-run 返回点，以及原生非 mixed ShareGPT 行为。
- 原生 benchmark 定义的隔离测试使用模拟发送函数，确认 1 次初始探测与 6 次正式调用共享 max_tokens 与特殊 token 参数；没有发 HTTP。发送函数内 max_tokens 的既有映射也作了静态核对。
- Ruff lint/格式检查、Python 语法编译、`git diff --check` 通过；原生受保护文件与基线无差异，7 个非 main 的 serving 定义保持一致；49 个原生 CLI 参数声明只在 dataset-name 的 choices 增加 mixed，原默认值未改。
- 本轮使用现有的 Python 3.12 临时测试环境及其已安装的 NumPy/PyArrow/Transformers，没有安装或升级包、改动依赖文件或环境配置。
- **完整 CLI 导入未通过**：`python benchmarks/benchmark_serving.py --help` 在现有测试环境报缺少 aiohttp；该环境还缺少 pandas、datasets、Pillow、PyTorch。未改原生导入或移动类来绕过。单测通过 AST 隔离当前源码中的原生接口/main/benchmark 定义和模拟依赖，不能当作完整 CLI、真实模型或服务器验证。
- 未验证真实原始数据及 LMSYS 真实语言分布、目标模型 tokenizer、目标服务端上下文/真实推理、V1 引擎运行或大规模数据性能。没有下载完整数据集或模型，没有运行正式 1200 条实验，没有开发预测、调度、时间标签或逐类型时延统计。

测试命令：`python -m unittest discover -s tests/benchmarks -p test_mixed_workload_dataset.py -v`（使用已经具备测试所需库的环境）。

最终 diff：`benchmark_serving.py` 增加 36 行、删除 2 行；新增一个数据集模块、一个测试文件和本说明。原生类、发送器、引擎、另一套 benchmark、环境与锁文件均保持原样，用户的 AGENTS.md 不计入本次改动。下一步需要在已有完整 vLLM 环境中填写真实本地路径执行上述 dry-run，本批不自动安装依赖或启动真实 benchmark。
