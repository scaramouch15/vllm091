# 研究内容一：首轮在线 shadow 接入（待服务端验收）

本分支以兼容工具提交 `a48a4a34497519b190daa81628f0094325dd7d6c` 为父提交，运行源码原基线为 `09c35e97b0605bd437c725562ca05afaacdb82e1`。未合并研究分支。用户已报告完整服务副本的依赖、1440 条预测及真实 V1 GPU 请求通过；这是接入前环境证据，不是本分支 shadow 验收结果。不重新训练、不安装原环境依赖。

## 行为和模型口径

默认 `VLLM_R1_SHADOW=0`：不导入 sklearn/joblib、不加载模型、不创建 shadow writer、计数器或首次记录集合，也不做快照聚合/预测。新增 ECR 可选字段缺省 None；因此需前端/Core 同版本，不能滚动混用旧 schema。关闭时保留原生队列、发送、计时和统计路径；新增字段属性及条件判断仍有极少量 Python 开销，不宣称二进制协议或墙钟耗时完全不变。

开启：仅 CPU 预测，绝不排序 waiting/running 或候选列表。`np.sort` 只对一个请求的三个分位数做训练规定的后处理。本轮没有“预测+排序完整耗时”。候选预测只是观察，没有通过 KV 检查的含义。

B 模型固定 SHA256：`23aff057712e7bec05db91d470af39a1b34eabddbf5852914710e1b8e4b96e1f`。仅从用户自己的可信本地 joblib 加载，先核对哈希，再反序列化。固定 Python 3.10.22、NumPy 2.2.6、SciPy 1.15.3、sklearn 1.7.2、joblib 1.5.3、threadpoolctl 3.6.0；不匹配启动失败，不能悄悄换模型。前端/每个 Core 初始化各加载一次完整原 artifact，随后只保留对应角色的 encoder+length head 或 time head。当前未拆分 artifact；启动反序列化和瞬时内存成本不能称为零。

TF-IDF.transform 原文本→SVD.transform→float64 的 `[num_prompt_tokens,text_0,...,text_31]`→原 sklearn 三个模型 predict→非负裁剪、分位数排序、2048 截断。树 float32 输入转换/阈值比较由 sklearn 原生实现负责，不重新实现树。时长顺序固定 `[num_prompt_tokens,running_count,kv_usage,pred_output_q50,pred_output_q90,pred_output_q95]`，相同非负裁剪/分位数排序，单位秒，目标是原生 inference_time 而非 GPU 时间或客户端 E2E。模型所有字段顺序、分位数、scaler/policy 均检查。

## 源码接入位置

| 文件/函数 | 改动与生命周期 |
|---|---|
| `serving_completion.py:OpenAIServingCompletion.create_completion` | 原生预处理、to_sampling_params 和 `request_id_item` 生成之后，generate 之前调用 prepare。每次 API 请求计算一次；同文本的另一次请求重新计算，不做跨请求缓存。普通 curl 的原生 ID 同样可以使用。|
| `async_llm.py:AsyncLLM.__init__/shutdown` | Core 启动后创建前端模型与 writer，避免把 writer 线程继承给子进程；Processor 共享同一实例。原生 api_server context 的 shutdown finally 显式关闭。|
| `processor.py:Processor.process_inputs` | 对 cloned SamplingParams 消费并移除保留键 `_vllm_r1_shadow_static_v1`；校验最终 token 数/n/max_tokens；保留其他 extra_args。|
| `engine/__init__.py:EngineCoreRequest` | 在数组结构末尾追加可选三数字 tuple，不传原文本、task_type、参考答案或新 HTTP 字段。|
| `request.py:Request/from_engine_core_request` | 将长度数字保留至请求释放。有有效静态预测时才创建 admitted/predicted bool，恢复不清零，不用 num_computed_tokens==0 判断首次准入。|
| `scheduler.py:Scheduler.schedule` | running 调度完成、waiting 接纳之前，原生无本轮抢占、有 waiting/token_budget/运行名额时调用 sweep；仅 O(1) 读 len(running)、KV usage，不调用采集器 snapshot/make_stats、前缀查询或 allocate_slots。|
| 同函数的实际 WAITING→RUNNING 位置 | 原 KV 分配等检查成功后记录首次准入；不把候选日志当作准入日志。|
| `Scheduler._free_request/shutdown`、`EngineCore.shutdown` | 记录 terminal 覆盖、清理随 Request 生命周期的字段，Core 在模型 teardown 前先关闭 writer，Scheduler 幂等兜底。|
| `metrics/r1_shadow.py` | 标准库配置/门禁、惰性数值库加载、模型推理、限额候选、共享状态与异步记录。|

CompletionRequest 当前只把 kv_transfer_params 映射到 extra_args，不接受客户端任意 extra_args。保留键只由服务器构造本地 StaticEnvelope 对象；Processor 拒绝 dict/列表伪造值并移除键，IPC 中没有该对象。这个内部 Python 对象不是客户端协议或安全隔离边界，不允许不可信插件在进程内构造请求。

## 支持范围与跳过

启动模型/Tokenizer 名称必须是 Meta-Llama-3-8B-Instruct，核对 Llama 架构、4096 hidden、32 层、128256 vocabulary，禁止 encoder lower-case 改写、remote KV、speculative config，首轮 DP=PP=TP=1、使用原生 V1 FCFS scheduler 类。此检查不等于 LLM 权重/Tokenizer 文件完整哈希认证，实际部署路径和版本仍须在验收中记录。header 记录解释器、实际预测包版本/导入位置、模型/Tokenizer revision、dtype 和实际线程池。

请求仅参与 `/v1/completions`、单个字符串、n=1、非 beam、无 adapter、无截断、add_special_tokens=false、有效 max_tokens=2048、无 guided/remote KV。前端直接用规范化后的文本，必须与原字符串完全相等、token IDs 与送入引擎的相等。mixed 已应用一次 chat template；服务不再套模板，也不 strip/压平代码空白。不同模板/模型输入口径的验证仍由客户端固定验收请求保障，服务端没有 prompt_hash allowlist。

token-ID、embeds、多 prompt、n>1、beam、截断（包括 -1）、特殊 token/adapter/上限不符等仍走原生请求路径，输出 static_skip 原因。chat API、直接 engine API 等没有本阶段静态数据的请求由 Processor 记 no_static；不做 token 解码来伪造文本。原生自身不支持的请求依旧按原生错误处理。

启动模型/版本错误显式报错。单次静态/动态预测异常只计数、记录异常类型（不记录异常消息，防止带入 prompt），跳过该次预测；不填零值。动态失败仍能走原生准入，日志 `predicted_in_this_batch=false`。取消、错误、自然结束均保持原生状态机；terminal 不作为完成时长标签。静态后、入 Core 前发生取消/错误时可能只有前端日志，不能据此补造 Core 标签。

## 有界首轮测量设置（验收建议）

| 环境变量 | 默认/要求 |
|---|---|
| `VLLM_R1_SHADOW` | 0，显式 1 才开启 |
| `VLLM_R1_SHADOW_MODEL` | B joblib 的实际绝对路径，开启必填 |
| `VLLM_R1_SHADOW_RUN_ID` | 本次服务生命周期的安全名称，开启必填 |
| `VLLM_R1_SHADOW_DIR` | 仓库外独立输出根目录，开启必填 |
| `VLLM_R1_SHADOW_CANDIDATES` | 32；每轮最多 K 个有效候选 |
| `VLLM_R1_SHADOW_SCAN_LIMIT` | 128；最多检查队列前 S 个位置，K<=S<=4096 |
| `VLLM_R1_SHADOW_EVERY` | 8；第 1、9、17…次符合原生 waiting 机会条件的轮次预测 |

在扫描窗口内保持原队列顺序，选前 K 个有效且未首次准入的 WAITING 请求；不为填满 K 扫描无界队列。WAITING_FOR_*、PREEMPTED、已首次准入、静态缺失均跳过并计数。空候选不组装特征、不预测。状态标量每个符合条件轮次读取一次；有采样间隔的轮次仍可记录实际准入状态。同一请求等待期间可多次动态预测，静态计算不重复。覆盖率分母是记录的 completion/Processor 请求、waiting 机会、被有界扫描看到的机会，不能声称覆盖全部队列。

决策 ID 优先复用当前采集器 admission_snapshot.decision_batch_id；采集器关闭时用 shadow writer.instance_id:轮号。每轮共享纯数值 snapshot，无 running ID 集合、无动态对象引用。candidate_count 为本轮实际选取数量；candidate 记录与实际 first_admission 记录分开，同批是否预测明确标注。抢占恢复不覆盖首次记录。

## 计时和写出

perf_counter_ns 单调高精度计时。每个静态成功记录含 tfidf_ns、svd_ns、text_transform_ns（前两项合并区间）、length_head_ns（含 float64 组装与后处理）、static_compute_ns（prepare 开始至预测/临时值构造结束，含门禁，不含 emit）。summary 的 static_total_ns 包含门禁、失败/跳过、记录构造及入队；与 compute 是嵌套关系，不相加。

动态记录含 state_read_ns、extract_assemble_ns（从 sweep 开始至矩阵就绪，包含 state_read）、time_head_ns（批量三头及后处理/校验）、shadow_compute_ns（至候选记录构造结束，含上述区间）。summary 的 shadow_total_ns 再包含 emit 与返回开销，失败/空/间隔跳过也计入。不存在排序耗时项。候选数量会限制记录大小，但可变大小 JSON 的实际成本由后台度量。

日志复用 RecordWriter：每进程独立文件、1024 有界队列、put_nowait、约 1 秒或64条 flush、正常关闭最多2秒 drain。写入异常/队列满计数、首次报错后抑制重复刷屏；可能丢失 summary，自身 footer 的 dropped_records、serialization_errors、writer_errors 和 clean_shutdown/records_complete 必须一起检查。缺 footer 表示不完整或未确认完整，不代表全部丢失。两秒排空超时不是完整成功。

summary.counts 中 log_enqueue_ns/calls 是调用 emit 的整体前台时间，footer.metrics.enqueue_ns 是 writer 内部入队区间，两者嵌套；后台 background_serialize/write/flush 分开，不能相加解释端到端增量。后台线程与前台仍争用 CPU/GIL。shadow_static/candidates/first_admission/terminal/summary 均不保存完整 prompt。

本轮仅追加 ECR 数字字段，没有给 IPC 加计时 Hook。真实编解码功能由 check_ipc.py 验证；字段编码增量、ZMQ 传输开销尚未测量。跨进程送达时差不能直接称为序列化耗时，包含排队/进程调度。

不移动 scheduled_timestamp：它仍在 schedule 较早的位置生成；旁路动态计算位于它之后，可能进入原 scheduled_ts→last_token_ts 的 inference_time 区间。前端静态阶段位于 Processor 默认 arrival_time 之前，可能不进入原引擎统计，但会进入客户端 E2E。EngineCoreOutputs 批次时间、SCHEDULED/NEW_TOKEN 与完成公式均不改。仅保证不主动改变准入规则，CPU争用仍可能改变实际墙钟、到达交错和批次轨迹。

线程限制由启动环境在导入前设置并检查实际 threadpoolctl 状态，不在预测函数中修改全进程线程池。四个线程变量也会影响服务内其他 CPU 库；关闭/开启对照必须使用相同设置。不能把微基准的单 CPU taskset 套到整个服务进程。没有本次资源授权，不自行选 CPU 或申请计算节点。

## 与旧采集器的组合

shadow 和 `VLLM_R1_COLLECT` 完全独立，各自 run_id/输出根目录。shadow 不要求类型、预测值、ID 前缀或 matching-run 客户端。

| 服务 COLLECT | 服务 SHADOW | 行为 |
|---|---|---|
| 0 | 0 | 原生请求，两个扩展均不运行 |
| 0 | 1 | 推荐首轮普通 curl/shadow 检查；无需特定 ID |
| 1 | 0/1 | 旧采集器仍要求 matching-run 客户端的 r1.<run_id>.<uuid> ID、单 prompt/n=1、原生统计开启；普通 curl 可能400，这是旧采集器限制，不是 shadow 新增限制 |

下面客户端单独开启 COLLECT 是为保存样本→attempt/API/engine ID；服务器 COLLECT=0 时仍可接受这些普通 headers。其旧 merge 会因缺 core/frontend 采集记录报不完整，这是预期；不要用旧采集器 merge 判定 shadow 完整性。各类日志保存在不同根目录，不能混给旧合并器。

## 服务器待执行：CPU 先验与真实验收

以下仅准备命令，本轮不自行连接服务器或执行。使用已通过 GPU 验证的完整服务副本，不能换成仅 sklearn 环境。先实际确认解释器、迁移包和获准节点；不安装包。

```bash
set -euo pipefail
# 在实际服务源码仓库中执行；新 worktree 不改原运行 checkout。
git fetch origin review/r1-shadow-20261025
R1_REVIEW_COMMIT="$(git rev-parse origin/review/r1-shadow-20261025)"
R1_SOURCE_PARENT="$(mktemp -d "$HOME/whb/r1_shadow_source.XXXXXX")"
git worktree add --detach "$R1_SOURCE_PARENT/source" "$R1_REVIEW_COMMIT"
cd "$R1_SOURCE_PARENT/source"
read -r -p '已验证完整服务副本的绝对 Python 路径：' R1_SERVICEPY
read -r -p 'A800 迁移包实际解压根目录：' R1_MIGRATION
test -x "$R1_SERVICEPY"
R1_BMODEL="$R1_MIGRATION/config_B_new310/results/models/tfidf.joblib"
test -f "$R1_BMODEL"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R1_ACCEPT="$(mktemp -d "$HOME/whb/r1_shadow_accept.XXXXXX")"
printf 'R1_REPO=%q\nR1_SERVICEPY=%q\nR1_MIGRATION=%q\nR1_BMODEL=%q\nR1_ACCEPT=%q\n' \
  "$PWD" "$R1_SERVICEPY" "$R1_MIGRATION" "$R1_BMODEL" "$R1_ACCEPT" > "$R1_ACCEPT/paths.sh"
printf 'commit=%s\npython=%s\nmodel=%s\noutput=%s\n' \
  "$R1_REVIEW_COMMIT" "$R1_SERVICEPY" "$R1_BMODEL" "$R1_ACCEPT"
"$R1_SERVICEPY" -B -c 'import sys,vllm; print(sys.executable); print(vllm.__version__,vllm.__file__)'

# 以下 CPU 命令应逐条退出码为0后再继续；保存各自完整日志。
"$R1_SERVICEPY" -B tools/r1_shadow/validate_cpu.py \
  --model "$R1_BMODEL" --validation "$R1_MIGRATION/data/validation.jsonl" \
  --reference "$R1_MIGRATION/config_B_new310/results/tfidf_validation.jsonl" \
  --out "$R1_ACCEPT/model_validation" > "$R1_ACCEPT/model_validation.log" 2>&1
echo "exit_code=$?"
"$R1_SERVICEPY" -B tools/r1_shadow/check_ipc.py \
  --out "$R1_ACCEPT/ipc" > "$R1_ACCEPT/ipc.log" 2>&1
echo "exit_code=$?"
```

set -e 保证任何前置检查失败即停止；保留已打印输出目录和日志，不继续启动。新 worktree 不包含旧 checkout 的未跟踪编译产物；若现有安装依赖它们，CPU 导入检查可能报缺少原生扩展。此时先回传实际 import/source 错误，不安装或重建服务环境，也不将旧 Python 文件混回新分支绕过。

上面验证实际 shadow 函数，全部1440记录按身份/顺序对照 B 自己保存的长度/时长预测，rtol=atol=1e-10，并检验1/8/32/64/128批量路径。失败停止，不放宽容差。check_ipc 用真实 ECR/原生 MsgpackEncoder/Decoder，经 spawn 子进程→Request，比对缺省与数字字段；不是实际 ZMQ/LLM 请求验收。

在已获准 GPU 节点，CPU 两项先通过后，服务终端设置（先检查模型路径、端口可用；本轮只给命令）：

```bash
export VLLM_R1_COLLECT=0
export VLLM_R1_SHADOW=1
export VLLM_R1_SHADOW_RUN_ID=shadow_accept_001
export VLLM_R1_SHADOW_DIR="$R1_ACCEPT/shadow"
export VLLM_R1_SHADOW_MODEL="$R1_BMODEL"
export VLLM_R1_SHADOW_CANDIDATES=32 VLLM_R1_SHADOW_SCAN_LIMIT=128 VLLM_R1_SHADOW_EVERY=8
# 使用前一段已核实的副本绝对解释器、工作目录及线程设置。
VLLM_USE_V1=1 "$R1_SERVICEPY" -m vllm.entrypoints.openai.api_server \
  --model "$HOME/whb/models/Meta-Llama-3-8B-Instruct" \
  --host 127.0.0.1 --port 8011 \
  --disable-log-requests --enable-chunked-prefill \
  --enforce-eager --tensor-parallel-size 1 --max-num-seqs 8 \
  > "$R1_ACCEPT/server.log" 2>&1
```

同节点另一终端先 `source /本次实际输出目录/paths.sh`，再 `cd "$R1_REPO"`；清除 PYTHONPATH/PYTHONHOME 并设置与上面相同的四个线程变量及 PYTHONNOUSERSITE/PYTHONDONTWRITEBYTECODE。路径文件位于刚打印的 R1_ACCEPT，不是字面占位符。以下三个变量不要求 shadow，但客户端 COLLECT 用于关联样本；输出名称存在时必须换新：

```bash
VLLM_R1_COLLECT=1 VLLM_R1_RUN_ID=shadow_accept_001 \
VLLM_R1_COLLECT_DIR="$R1_ACCEPT/client" \
"$R1_SERVICEPY" benchmarks/benchmark_serving.py \
  --model "$HOME/whb/models/Meta-Llama-3-8B-Instruct" \
  --backend vllm --host 127.0.0.1 --port 8011 \
  --endpoint /v1/completions --dataset-name mixed \
  --num-prompts 12 --seed 0 \
  --mixed-chat-count 4 --mixed-code-count 4 --mixed-summary-count 4 \
  --mixed-chat-path "$HOME/whb/hf_datasets/lmsys-chat-1m/data/train-00000-of-00006-4feeb3f83346a0e9.parquet" \
  --mixed-code-path "$HOME/whb/hf_datasets/CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl" \
  --mixed-summary-path "$HOME/whb/hf_datasets/cnn_dailymail/3.0.0/validation-00000-of-00001.parquet" \
  --request-rate 4 --max-concurrency 16 \
  --mixed-output-len 2048 --temperature 0.9 \
  --mixed-export-path "$R1_ACCEPT/mixed_12_manifest.json" \
  --save-result --save-detailed --result-dir "$R1_ACCEPT/results" \
  --result-filename mixed_12_shadow_on.json > "$R1_ACCEPT/client.log" 2>&1
```

客户端控制固定请求/seed/文件；通过客户端 shard 的 engine_request_id 与 shadow_static/候选/准入关联，不用行顺序。静态预测对照同一实际 prompt/token/上限；时长需使用该候选日志里的实际 running_count/kv_usage 和长度预测重算，不能拿不同历史运行状态的时长预测直接比较。先用12条检查联通、ID/跳过/错误；若动态候选太少不能声称测得有效开销，可按已有120条（各40）配置另做有界验收，但不在本轮自动运行。

关闭/开启对照：先正常停止服务，确认 footer，再换全新 R1_ACCEPT/run_id；同一 checkout、解释器、模型、线程、服务/客户端参数，先 `VLLM_R1_SHADOW=0` 再重启 `=1`。固定seed/sources、检查manifest中的prompt顺序相同；ID每次实际发送唯一。比较实际吞吐/原生延迟及异常/覆盖率、候选数量，不把一次小验收写成调度收益或完整在线开销实验。服务仍运行时 footer 未闭合，不能判定完整。

正常停止服务和客户端后，在输出根目录检查 `shadow/<run_id>/shadow-frontend-*.jsonl`、`shadow-core-*.jsonl` 的 header/summary/footer；确认 dropped_records/serialization_errors/writer_errors/close_timeouts 为0、clean_shutdown/records_complete 为true；失败或缺 footer 如实保留。前端 static、Core candidate、first_admission、terminal 是否匹配/缺失须逐 ID检查；未准入取消可有静态但无准入，这不是零标签。原生 finished统计仅在旧采集器正确配置时存在，不自行造完成标签。

打包回传（不含模型、数据、环境；manifest含请求文本，请私下传输，不上传公开仓库）：

```bash
R1_DELIVERY="$(mktemp -d "$HOME/whb/r1_shadow_delivery.XXXXXX")"
tar -czf "$R1_DELIVERY/r1_shadow_accept.tar.gz" \
  -C "$(dirname "$R1_ACCEPT")" "$(basename "$R1_ACCEPT")"
(cd "$R1_DELIVERY" && sha256sum r1_shadow_accept.tar.gz > r1_shadow_accept.tar.gz.sha256)
printf '回传：%s\n' "$R1_DELIVERY"
```

## 本地检查与未验证项

本地现有 Python3.12，仅使用标准库及已有 NumPy；不安装包、不读取训练/测试数据、不加载真实 B 模型。轻量测试执行实际 Completion、Processor、Request、Scheduler 方法（AST加载，资源 stub），比较运行基线与 shadow开/关调度输出、KV调用、事件/状态和队列，覆盖静态次数/空白、字段门禁、伪造值、失败回退、首次/恢复/取消、共享数值快照和有界扫描。已有采集器及 mixed 回归继续执行。采集器原“整个IPC文件不变”断言仅调整为保护原有字段，显式允许新增尾部可选字段。

本轮本地结果：shadow 21项正常模式通过、21项 `-O` 通过；采集器34项通过（含单类 mixed 保护）；完整 mixed 测试因现有解释器缺 pyarrow，在导入阶段受阻，未安装依赖或声称通过。Python3.10 语法静态检查、五个 Bash 块语法检查与两个验证 CLI 的 --help 检查通过。完整日志保存在仓库外，不随源码推送。

```bash
python -B tests/benchmarks/test_r1_shadow.py -v
python -B tests/benchmarks/test_request_collector.py -v
```

真实 sklearn B artifact 的全部验证预测、真实 Msgpack+spawn、ZMQ/服务端路径、开关后真实GPU输出/吞吐/延迟/批次、取消时序以及长时间队列满与进程异常退出均待服务器验收。兼容工具的23文件哈希记录属于旧运行基线，不能用来声称新 Hook 文件仍与旧基线相同。本分支是待审查实现，不是已经完成的在线实验结论。
