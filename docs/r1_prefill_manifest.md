# R1 prefill 身份清单、回放和采集后校验

本实验工具不改 Scheduler，也不加载预测模型。清单与采集记录含原始 prompt；仅在受控目录保存。原始测试集不得用于调参，本工具不会验证使用者声称的旧测试身份清单是否完整。

## 输入与划分

先用现有 mixed dry-run 和 `--mixed-export-path` 生成候选清单。候选清单应来自本地 tokenizer，源文件与 tokenizer 需要在划分及回放环境可读；当前工具要求相同的绝对路径和文件内容。采集前准备仅含 `input_hash`/`prompt_hash` 的 train、validation、诊断及可选旧测试身份 JSON 或 JSONL（缺少的一类哈希不会凭空补全）。只使用不含标签和结果的文件。路径中的原始源文件、清单和 tokenizer 均不可覆盖。

```bash
python benchmarks/r1_prefill_manifest.py split \
  --candidate "$CANDIDATE_MANIFEST" \
  --train-exclusion "$OLD_TRAIN_IDENTITIES" \
  --validation-exclusion "$OLD_VALID_IDENTITIES" \
  --diagnostic-exclusion "$DIAGNOSTIC_IDENTITIES" \
  --old-test-identities "$OLD_TEST_HASH_ONLY" \
  --seed "$SPLIT_SEED" --target 1800 --output "$NEW_APPROVED_JSON"
```

旧测试身份不完整时省略该参数或提供现有部分；summary 为 `unverified`。`--old-test-complete` 仅记录操作者声明，工具无法独立证明旧测试身份集完整。批准前必须检查 exclusion 的实际覆盖、summary 的各类与来源数量、长度、重复组和 shortfall。采样不足不会补齐，也不会弱化隔离。`input_hash` 或 `prompt_hash` 任一相同就传递合并为一组，且排除以整组进行。该工具不会推断旧记录的诊断身份。清单的候选 run、源文件 SHA256、tokenizer 文件 SHA256、seed、候选和 split 均保存于批准 JSON；请另行保存其 `sha256sum` 并审核内容。对于自定义 tokenizer 或未识别的辅助资源，已列出的 tokenizer 文件哈希不等于整个环境依赖指纹，需另外固定运行环境。

## 回放

在已有 `benchmark_serving.py --dataset-name mixed` 命令基础上，继续传入**生成候选清单时的**三类 counts、`--num-prompts`、seed、原文件路径、output-len、tokenizer 和 tokenizer-mode，并增加：

```bash
  --mixed-approved-manifest "$NEW_APPROVED_JSON" \
  --mixed-approved-split train
```

`--num-prompts` 此处是原候选总数；实际发出的是选定 split。回放先重新执行原 mixed 构造流程、校验源文件和 tokenizer 哈希、tokenizer/配置以及整个候选列表，再按批准顺序选取 split；任一不一致失败，不退回随机采样。服务采集前请审查数目、hash 和日志后再执行。`--mixed-export-path` 为原有接口，可能覆盖同名文件，应使用新的独立文件路径。

对应服务运行：各 split 使用独立服务生命周期和唯一 `VLLM_R1_RUN_ID`；例如 `prefill_train_r4_001`、`prefill_calibration_r4_001`、`prefill_test_r4_001`。每轮使用独立日志、manifest 和 benchmark 输出目录。固定 FCFS、`max-num-seqs=32`、客户端 `max-concurrency=64`，记录请求率和运行配置；不在本轮启动服务。

## 采集后

使用现有 `request_collector.py merge --input-dir ... --output ...` 合并同一 run 的 shard，保留 footer、诊断和未关联记录。然后执行：

```bash
python benchmarks/r1_prefill_manifest.py validate \
  --approved "$NEW_APPROVED_JSON" --split train \
  --merged "$TRAIN_MERGED_JSON" --output "$NEW_TRAIN_CHECK_JSON"

python benchmarks/r1_prefill_manifest.py validate \
  --approved "$NEW_APPROVED_JSON" --split calibration \
  --merged "$CAL_MERGED_JSON" --other-validation "$NEW_TRAIN_CHECK_JSON" \
  --output "$NEW_CAL_CHECK_JSON"

python benchmarks/r1_prefill_manifest.py validate \
  --approved "$NEW_APPROVED_JSON" --split test \
  --merged "$TEST_MERGED_JSON" \
  --other-validation "$NEW_TRAIN_CHECK_JSON" "$NEW_CAL_CHECK_JSON" \
  --output "$NEW_TEST_CHECK_JSON"

python benchmarks/r1_prefill_manifest.py compare \
  --validations "$NEW_TRAIN_CHECK_JSON" "$NEW_CAL_CHECK_JSON" "$NEW_TEST_CHECK_JSON" \
  --output "$NEW_ALL_SPLITS_CHECK_JSON"
```

检查输出 `valid`、`cross_split_check`、`cross_split_errors`、`decision_batches`、正式和 probe 数。`cross_split_check` 为 `not_performed`、`partial` 或 `complete`；仅完整三划分且每项有效、run_id 和批次均两两无交集时才为 `complete`。`compare` 明确报告缺失、重复、无效 split 和清单哈希不一致；同一 split 的每个 merged 报告也必须使用不同的服务 run_id。单独执行一个 split 不能证明跨划分隔离。上述 `validate` 读取 merged 报告中的原生标签来检查 prefill 时间，必须在模型方案锁定后才对 test 执行；本工具目前没有独立的不读取 test 标签的身份核验入口。若方案未锁定，可只使用预先准备的无标签身份清单进行排除，不要运行 test 的 merged 校验。该校验要求 merged 报告完整且每次运行正式输入集合与批准 split 完全一致；不完整记录不会伪装为成功。

## 时序边界

原生 `prefill_time` 为首次 scheduled_ts 到首次输出批次 first_token_ts 的秒数，包括过程中可能发生的抢占等待；它不是精确 GPU 首 token 时刻或客户端 TTFT。prefix cache 的实际命中量在原生准入流程中才查询，不能未经验证当作先前候选时已知的特征；chunked prefill 可能跨多个调度轮。首次准入共享状态快照与潜在候选决策时点也必须单独核对。真实类型只用于离线覆盖，不进入预测器输入。
