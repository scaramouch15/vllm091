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

所有区间使用 perf_counter_ns 单调墙钟，包含被操作系统暂停/争用的等待，不是纯 CPU 指令时间。新增完整前台阶段直方图独立于成功预测日志：正常关闭时在 shadow_summary.foreground_distributions 导出，不靠累计总时间推算分位数，不因逐条日志队列满而停止内存聚合。

### 完整前台分布及有界方法

- `overall` 按 stage 合并所有 outcome；`groups` 按 stage/outcome/candidate_count 分组。每组有 n、total_ns、min_ns、max_ns、非零 bins，以及 `p50_bounds_ns/p95_bounds_ns/p99_bounds_ns` 和对应 `p50_upper_ns/p95_upper_ns/p99_upper_ns`。没有观察时分位数是 null，不能当0；未发生的组不创建。
- 这是**近似耗时分位数**：nearest-rank 落入的桶给出上下界，upper 字段是上界估计，不冒称原始样本精确分位数；与 NumPy 默认线性插值分位数不是同一口径。0–15ns 精确；其余每个2倍区间8个桶（保留4位有效二进制位），桶宽不超过下界的1/8，另保留 >=2**64 ns 的固定溢出桶（该尾桶不承诺上述相对精度）。最小/最大观测值收紧桶边界；保留 bins 可离线合并同口径分布。
- 每组固定497个 uint64 计数器（原始桶数组3976字节），没有逐样本数组、随机采样或随请求数增长的 ID 集合。组数受候选上限K和固定stage/outcome约束；默认K=32的桶存储低于0.5MiB，极限K=4096时约32MiB，另有有界 Python 元数据。按需创建分组；首次建桶开销同样需要计入测量机制成本，不作零成本假设。
- 更新为 O(1) 整数运算；只有关闭时计算分位数、合并 overall 和构造 summary。不新增“计时记录”的 emit，不递归计时或递归日志。`timing_bookkeeping_calls/ns` 单独汇总最后统计更新的成本，包括按需创建桶，不再为它建立嵌套计时分布。summary_build_ns 记录关闭时汇总构造成本。

| stage | 一次观测的测量边界 | 包含/排除 |
|---|---|---|
| `static_foreground` | prepare 入口→finally 中最后计时点 | 包含门禁、文本变换、长度头、临时数字构造、成功/失败/跳过处理、调用方日志字典构造及 emit 全程；按 success/failed/skipped 分组。每次实际调用prepare一条聚合观测，同文本不同请求仍各算一次。|
| `dynamic_foreground` | sweep 入口→finally 中最后计时点 | 包含状态读取、候选提取/特征组装、批量时长预测、候选ID返回值构造、日志构造及 emit（若有）；只覆盖符合原生waiting机会条件而实际调用sweep的决策批次，不包含其他schedule轮次。|
| `processor_consume` | consume_static 入口→finally 中最后计时点 | 包含内部键字典复制/移除、token数/n/max_tokens/三数字校验与转换、计数和 no_static/mismatch 日志的构造/入队。按 received/no_static/mismatch/failed 分组；不含此前原生SamplingParams.clone、token预处理、函数调用方门禁或之后ECR构造。|
| `first_admission` | admit 入口→finally 中最后计时点 | 包含无静态/已准入的短路检查；实际首次准入包含标志更新、关联字段/候选成员检查、日志构造和 emit。按 admitted/no_static/already_admitted/failed 区分；不包含原生KV分配、状态机和SCHEDULED事件。|
| `terminal` | finish 入口→finally 中最后计时点 | 包含覆盖计数、状态字段复制、结束日志构造和 emit；按 completed/failed 区分；不包含原生连接器/KV释放、完成统计计算。|
| `log_enqueue` | Shadow.emit 入口→最后计时点 | 包含 model_sha256 字段/记录复制、RecordWriter.emit 构造与非阻塞入队、结果计数及首次失败告警；按 enqueued/dropped/failed 区分。调用方在进入emit前构造的字典不在这个子区间，已包括在外围完整阶段。|

“完整前台”指入口到该方法末尾的业务/记录区间，**包含记录构造和入队**；所有 stage 的最后一次直方图更新、最终计时指令之后的统计/函数返回，以及调用方分支/分派不包含在自身区间。最后统计更新单独计数，其成本以及新增字段引起的原生clone、Request/ECR构造增量仍属于真实扩展开销，不能把分布宣称为所有新增字节码都已逐项计量。内层log_enqueue自身聚合开销会进入外层static/dynamic/Processor/准入/结束阶段的墙钟区间。

动态每组的 candidate_count 是本批**实际已收集**候选数，不是配置K；成功/失败可按其值分别对照，失败中途发生时它可能是已收集的部分候选数量。empty 的数值为0；interval_skip 未扫描，使用 null（未知），不能解释成队列没有有效候选。`dynamic_outcome_success/failed/empty/interval_skip` 显式计数且未发生时为0；static_outcome_success/failed/skipped 同理。success 表示预测/阶段逻辑成功，日志能否保存另看 log_dropped/log_errors 和 footer。overall 的动态分布混合规模与结果，比较规模开销应筛选 groups 的实际candidate_count及outcome，不能只比较overall。

原成功静态记录的 tfidf_ns、svd_ns、text_transform_ns（前两项合并）、length_head_ns（含float64组装/后处理）和 static_compute_ns 继续保留；它们不含 emit。原动态记录的 state_read_ns、extract_assemble_ns（包含state_read）、time_head_ns 和 shadow_compute_ns 继续保留；它们也不含 emit。新 static_foreground/dynamic_foreground **包含**对应emit/记录路径；static_total_ns/shadow_total_ns 与对应完整分布的 total_ns 对齐。不能将子区间、外围区间、分项P50/P95/P99或不同时机的附加成本相加，直接解释成客户端E2E增量。

日志复用 RecordWriter：每进程独立文件、1024有界队列、put_nowait、约1秒或64条flush、正常关闭最多2秒drain。写入异常/队列满计数、首次报错后抑制重复刷屏；summary 本身也可能丢失。必须检查 footer 的 dropped_records、serialization_errors、writer_errors、clean_shutdown/records_complete；缺summary或footer时分布不完整/未经确认，不能将已有样本冒充完整成功。summary 是关闭前的数值快照，排除它自身的emit及之后shutdown，writer footer仍计其入队/写出。关闭汇总不在逐请求/逐批次分布里。

log_enqueue 与外围完整阶段嵌套；footer.metrics.enqueue_ns 是 writer 内部子区间，后台 background_serialize/write/flush 单独记录。后台线程仍争用CPU/GIL；不能将前台、后台和嵌套区间简单相加。日志及直方图不保存完整prompt。

### IPC和端到端边界

`check_ipc.py` 只验证真实ECR/原生Msgpack编解码、spawn和Request身份/数字字段的**功能正确性**，不测IPC开销。processor_consume也不是IPC耗时。新增字段引起的SamplingParams.clone、ECR/Request构造、编码字节/编解码增量、ZMQ传输与接收等尚未测量；跨进程送达时差包含排队和进程调度，不能直接命名为序列化开销。后续如测增量，需同内容有/无字段的编码/解码单独测量，并将发送到接收命名为传递延迟，不在本轮补做IPC开销实验。

不移动 scheduled_timestamp：它仍在schedule较早的位置生成；动态计算、首次准入记录等位于其后，可能进入原scheduled_ts→last_token_ts的inference_time区间。前端静态位于Processor默认arrival_time之前，可能不进入原引擎统计，但进入客户端E2E；Processor字段消费位于该arrival_time之后。结束日志位于Core释放路径，也可能影响同轮输出可见时机。EngineCoreOutputs批次时间、SCHEDULED/NEW_TOKEN和完成公式均不改。局部区间不能覆盖完整服务内增量，真实端到端影响只能通过**完全同配置的shadow off/on**比较客户端原生延迟和吞吐评估。

线程限制在启动导入前设置/检查，不在预测函数中反复修改全进程线程池；四个变量也会影响服务内其他CPU库。off/on必须保持同线程和服务/客户端配置，不能把微基准单CPU taskset套到整个服务进程。旁路仍不排序、不主动改变准入规则，CPU争用可能改变墙钟交错和实际批次轨迹。

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

### IPC 超时诊断重跑（先回传，不启动服务）

用户提供的服务器证据：在 `542fc998b9a22b87b93c7820c1327baa7b5fe8b6` 上，Python3.10.22、vLLM0.9.1、实际源码路径及shadow runtime全部1440条B模型一致性检查已通过；IPC第一项在等待60秒时失败。旧 `cases=[]` 只说明没有一项完整通过，第一项是None，不能归因于新增三个预测数字。没有阶段/栈证据时，模块导入、spawn启动、编解码阻塞等仅是待验证假设；NCCL警告不是已确认原因。300秒通过也不能证明原来一定是导入缓慢。

**源码已确认的旧工具缺口**：case只在成功后追加；start在清理try之外；poll只表示有字节/EOF，后续recv仍可能等不完整帧；仅terminate/join5秒，没有kill兜底、最终存活核对或进程句柄关闭。它们是工具可确认的问题，**不是此次poll超时的已确认根因**，特别是不完整帧问题不能解释已观察到的poll返回超时。

本版仍强制spawn，顺序检查None及(1.,2.,3.)，逐项核对原request_id/token数/数字值、正常exitcode=0，以及子进程和父进程CUDA未初始化。不跳过子进程、不换fork、不放宽通过标准。child路径也核对实际导入vLLM源码位置。

`--timeout`是每项父进程开始等待**完整结果消息**后的功能验收上限，默认60秒；正有限数才合法。下面300秒只是诊断等待上限，包含新进程导入/调度/处理等多项因素，**不是在线IPC开销**。父进程另有10秒正常退出检查；超时栈刷新宽限0.25秒；清理terminate最多join5秒，仍存活则kill再join5秒；接收线程join最多1秒。延长结果等待上限不改变正常退出或回收要求。

为避免poll后的recv遇到部分帧而无界阻塞，recv放在单个daemon reader线程，主线程使用有截止时间的队列等待。失败时仍关闭Pipe两端、terminate/kill并回收所属子进程，所有显式join有上限。仅在极端OS状态下kill仍不能回收时，先写出失败JSON、标注未回收，再退出诊断器以避免multiprocessing解释器结束时的隐式无界join；这不是“回收成功”，需按报告中的PID另外核查。若start未返回且无PID/handle，状态为unknown，不声称进程已回收。OS级start、文件I/O或第三方模块导入本身不能由Python等待参数保证硬实时终止，watchdog可尽量留下栈，不额外启动调试器或引入新的无限等待。

阶段日志每行含monotonic_ns、该进程/该case的elapsed_seconds、PID、case_id、role和stage，写入专用JSONL后立即flush，并flush镜像到console。child覆盖入口、ECR/Request/Decoder/Torch实际导入前后（隐式依赖归在相应外层导入区间）、decode、Request构造、CUDA检查、发送和target结束；parent覆盖本地导入、编码、start、等待、收到消息、退出检查及清理。

- `setup.parent.stages.jsonl`：父进程前置导入/最终CUDA检查；`none.*`、`three_lengths.*`：各case的parent/child阶段与stacks文件。
- 子进程在target入口安装一次性faulthandler watchdog，使用同机monotonic诊断截止时间；成功发送后保留文件和退出watchdog至进程真正退出，用于观察target返回后的atexit阻塞。不发送可能未注册的信号，也不周期性无限dump。父进程仅在setup/启动阶段使用自动watchdog，进入有界结果等待前将其取消；结果/退出等待超时时由主线程同步dump自身全部线程（含reader）栈，避免自动/手动转储在异常处理期间并发。
- 辅助watchdog从setup/case开始计时，可能比结果等待上限更早输出栈；栈文件存在本身不是超时或根因结论。子进程若卡在target入口之前，无法靠target内的watchdog得到其栈；报告标注无child阶段/无栈证据，不能据此判断具体bootstrap阻塞点。
- `ipc.json.cases`在每项开始前保留记录，失败也不删除。包括passed、exception/traceback、wait_timeout_seconds、实际wait_elapsed_seconds、child_pid/exitcode、是否仍活着、cleanup动作/错误/reaped、父子栈可用性。
- `child_stage_at_timeout`是采取清理动作前读取的阶段快照；`child_last_known_stage`是清理后的最后记录。两者仅来自对应child JSONL的**最后一条完整JSON行**，校验case/role/PID，最多读取1MiB，忽略未完成尾行。缺文件/身份不符等返回verified=false/last_event=null。它们不依赖stdout行顺序或跨进程时钟猜测。`child.target_exit`只说明target走到结束，不代表进程已正常退出，必须看exitcode/cleanup。child响应中的原异常及异常时最后刷新的阶段另行保留，不被发送/清理的第二个错误覆盖。

先从GitHub取**新工具提交**并创建新worktree，不覆盖仍保留的旧源码或失败目录。先source上一轮实际paths.sh取得已核实的R1_SERVICEPY；不重新克隆环境、不安装依赖，也不需要因本次工具修正重训/重跑已通过的1440条验证：

```bash
set -euo pipefail
test -x "${R1_SERVICEPY:?请先source原验收目录的paths.sh，取得实际解释器路径}"
git -C "$HOME/whb/vllm091" fetch origin review/r1-shadow-20261025
R1_IPC_SHA="$(git -C "$HOME/whb/vllm091" rev-parse origin/review/r1-shadow-20261025)"
R1_IPC_SOURCE="$(mktemp -d "$HOME/whb/r1_ipc_source.XXXXXX")"
git -C "$HOME/whb/vllm091" worktree add --detach "$R1_IPC_SOURCE/source" "$R1_IPC_SHA"
cd "$R1_IPC_SOURCE/source"
R1_IPC_DIAG="$(mktemp -d "$HOME/whb/r1_ipc_diag.XXXXXX")"
printf 'source_commit=%s\npython=%s\nsource=%s\n' \
  "$R1_IPC_SHA" "$R1_SERVICEPY" "$PWD" > "$R1_IPC_DIAG/source.txt"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# result必须不存在；保留原ipc/和ipc.log，不覆盖。
if "$R1_SERVICEPY" -B tools/r1_shadow/check_ipc.py \
    --out "$R1_IPC_DIAG/result" --timeout 300 \
    > "$R1_IPC_DIAG/console.log" 2>&1; then
  R1_IPC_CODE=0
else
  R1_IPC_CODE=$?
fi
printf 'exit_code=%s\n' "$R1_IPC_CODE" | tee "$R1_IPC_DIAG/exit_code.txt"
printf '诊断目录：%s\n' "$R1_IPC_DIAG"
if test -f "$R1_IPC_DIAG/result/ipc.json"; then
  cat "$R1_IPC_DIAG/result/ipc.json"
fi
# 成功/失败均先回传；失败不继续启动服务。
```

新的`--out`存在会报错，原失败材料不被删除或覆盖。异常杀死父进程可能来不及写最终ipc.json，已flush的独立阶段/栈文件仍应保留，不能据缺JSON宣称检查完成。

打包新诊断及原失败JSON供私下回传（不打入模型、环境或验证数据）：

```bash
read -r -p '原失败ipc.json所在目录的实际绝对路径：' R1_IPC_PREVIOUS
test -f "$R1_IPC_PREVIOUS/ipc.json"
R1_IPC_DELIVERY="$(mktemp -d "$HOME/whb/r1_ipc_delivery.XXXXXX")"
mkdir "$R1_IPC_DELIVERY/original_failure"
cp -p -- "$R1_IPC_PREVIOUS/ipc.json" "$R1_IPC_DELIVERY/original_failure/ipc.json"
tar -czf "$R1_IPC_DELIVERY/r1_ipc_diagnostics.tar.gz" \
  -C "$R1_IPC_DELIVERY" original_failure \
  -C "$(dirname "$R1_IPC_DIAG")" "$(basename "$R1_IPC_DIAG")"
(cd "$R1_IPC_DELIVERY" && sha256sum r1_ipc_diagnostics.tar.gz > r1_ipc_diagnostics.tar.gz.sha256)
printf '回传压缩包及小写.sha256：%s\n' "$R1_IPC_DELIVERY"
```

本地优化模式曾观察到自动/手动栈转储交错并伴随原生访问异常，失败日志保留在仓库外；增加取消watchdog后再手动dump的顺序保护，父进程有界等待阶段只保留主线程超时转储，并补充回归测试。该本地事件不是服务器原超时根因的证据。

本地新增标准库测试使用真实spawn/Pipe和合成依赖/codec，覆盖正常两项、结果等待超时、异常退出、发送正确结果后异常退出/不退出、阻塞reader、start失败、kill兜底、原错与次要清理错误、阶段来源、CUDA拒绝及失败JSON。它们验证诊断/资源生命周期，**不是本地真实vLLM/torch/Msgpack共存验证**；服务器实际导入/编解码及此次超时根因仍待新材料定位。

### 后续服务验收（IPC通过后另行执行）

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
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  > "$R1_ACCEPT/server.log" 2>&1
```

**max-num-seqs=8 是首轮等待候选覆盖验收设置**，配合既有请求率/并发检查waiting机会，不是正式部署最优值，也不代表本轮已完成正式开销实验。off/on两组服务命令均显式保留 `--max-model-len 8192 --max-num-batched-tokens 8192`；所有服务与客户端的行为参数必须完全相同（模型、上下文/批次预算、max-num-seqs、chunked prefill、请求数/seed/来源/顺序、请求率、并发、生成上限/temperature、线程及采集开关等）。只改变shadow开关及为防覆盖而更换的run_id/输出名称。

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

## CPU writer 后端开销诊断（离线工具）

`diagnose_cpu_overhead.py` 在离线环境比较同一真实 Predictor 和 `Shadow.sweep()` 的两种记录后端。它不启动 API 服务、benchmark、GPU 或调度循环，也不改服务入口和运行时代码。需要使用已经验证的完整 Python 3.10.22 服务环境和 B 模型；工具会由 `Predictor` 核对模型 SHA、Python/包版本和数值线程池。它不需要读取验证集或 prompt。

输入是本轮 shadow 运行生成的 Core `shadow-*.jsonl` 文件。工具从一条真实 `shadow_candidates` 记录取同一时刻的 `running_count`、KV 使用率、token budget 和至少 8 个真实候选。候选规模 1 与 8 使用这条记录中的真实数值输入；32 与 128 循环复用前 8 个数值候选，输出会逐条标记 `synthetic_copy=true`。没有包含至少 8 个候选的原始记录时，工具明确失败，不拼接不同状态时刻的数据。

A 使用真实 `RecordWriter` 的有界队列、后台线程和 JSONL 写出，并检查正常关闭、footer、dropped/error 计数。B 使用诊断内存 sink；它仍执行 `Shadow.emit`、调用方字典构造、字段校验和 `Shadow` 的计时统计，只捕获有界的候选预测投影并直接返回成功，没有队列、后台线程或文件写出。B 不是“完全无日志模式”，也不能声称有完整逐请求日志或线上记录完整。

每个独立测量块只创建当前条件的资源。块内记录后端启动、预热、正式样本、关闭时间；预热结束后用同一 Predictor 和当前 writer 创建新的诊断 Shadow，清空内部直方图再测正式样本。块关闭并完成校验后才开始下一块，因此 A writer 不会在 B 测量期间运行。块顺序交替 AB/BA；A 的 close、线程退出或 footer 检查失败时立即停止该配对，不生成有效 A/B 对照。关闭成本不计入 sweep 样本。

本次命令显式使用 `every=1`，使每次 sweep 都执行预测，便于核对同一批调用的外部时间和内部直方图。每一个 `Shadow.sweep()` 调用都在内存中限量保留；每种规模最多 150 个正式成功预测事件、最多 100 个预热事件，且对 128 候选另设单条件预测捕获上限。正式 `dynamic_foreground` 分布来自重建后的 Shadow，仅统计正式块；外部 `sweep_summary` 也只统计同一块的正式调用。工具逐块核对外部调用数、内部直方图数、成功数和预测频率，多块汇总后再次核对计数。所有逐样本计时结果在测量结束后写入新建输出目录。

默认 30 个正式成功样本适合冒烟检查。nearest-rank 的 P99 使用 `ceil(0.99*n)`，n=30 时 rank=30，P99 就是最大值。正式诊断建议每个条件、每个候选规模至少使用 150 个成功样本；即便如此，这仍是小规模 CPU 诊断，不能作为稳定的在线 P99 结论。

正式 `sweep()` 墙钟时间由调用方用 `perf_counter_ns` 包围函数入口到返回，包含该次调用的候选处理、预测、`Shadow.emit`、调用方记录字典构造及函数内计时统计。JSON 同时保留实现自身 `dynamic_foreground` 分布；它是嵌套区间，且末尾不包含自己的最后一次直方图更新。分位数用 nearest-rank：rank=`ceil(p*n)`，从小到大的一基索引；报告 P50/P95/P99/max，候选规模按实际预测 `candidate_count` 分组，interval skip 使用 `candidate_count=null` 单独报告。A 关闭耗时和后台线程 footer 指标单独保留，不能与前台时间简单相加。

先从以前的 shadow 输出目录选取实际 Core JSONL 文件。下面示例中的占位符要替换为现场已验证的绝对路径；输出目录用 `mktemp` 新建，不覆盖旧结果。必须在导入数值库前设置四个单线程变量，不要对整个服务设置 `taskset`：

```bash
set -euo pipefail
R1_SERVICEPY="/absolute/path/to/verified-vllm091-python"
R1_BMODEL="/absolute/path/to/results/models/tfidf.joblib"
R1_CORE_JSONL="/absolute/path/to/prior-shadow-run/shadow-core-PID-INSTANCE.jsonl"
test -x "$R1_SERVICEPY"
test -f "$R1_BMODEL" -a -f "$R1_CORE_JSONL"
R1_CPU_DIAG="$(mktemp -d "$HOME/whb/r1_shadow_cpu_diag.XXXXXX")"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$R1_SERVICEPY" -B tools/r1_shadow/diagnose_cpu_overhead.py \
  --model "$R1_BMODEL" --shadow-jsonl "$R1_CORE_JSONL" \
  --out "$R1_CPU_DIAG/result" --run-id cpu_diag_001 \
  --every 1 --warmup-sweeps 3 --sweeps 30 --block-size 5 \
  > "$R1_CPU_DIAG/console.log" 2>&1
printf 'diagnostic_dir=%s\n' "$R1_CPU_DIAG"
cat "$R1_CPU_DIAG/result/diagnostic.json"
```

输出目录包含 `diagnostic.json` 和每个 A 块独立的真实 JSONL/footer。报告保存脚本/模型和输入 shadow 文件标识、输入行号与 SHA、运行环境、线程池、每块冷启动/预热/关闭成本、正式样本、A/B 预测逐项一致性、多块合并分布及每个 A footer/线程退出结果。正式推荐可将命令的 `--sweeps 30` 改为 `--sweeps 150`；输出目录必须保持全新。它不把候选 ID/数值伪装成原始 prompt，也不复制原始日志到仓库。

该诊断只回答“在此 CPU、此 Predictor、此候选与状态输入下，真实异步 writer 后端和内存 sink 的局部差异”。CPU结果不能直接解释客户端 E2E 增量，也不能证明在线尖峰来自锁、磁盘、GIL或操作系统调度。B 绕过 writer 后端仍保留调用方日志构造与 `Shadow.emit`，因此 A/B 的差异是 writer 后端在这条 `sweep()` 路径中的诊断，不是服务开关 off/on 结果。32/128 是合成规模，不表示在线已经观测到这些候选数。实际服务影响仍需同配置 shadow off/on 运行比较。

## 本地检查与未验证项

本地现有 Python3.12，仅使用标准库及已有 NumPy；不安装包、不读取训练/测试数据、不加载真实 B 模型。轻量测试执行实际 Completion、Processor、Request、Scheduler 方法（AST加载，资源 stub），比较运行基线与 shadow开/关调度输出、KV调用、事件/状态和队列，覆盖静态次数/空白、字段门禁、伪造值、失败回退、首次/恢复/取消、共享数值快照和有界扫描。已有采集器及 mixed 回归继续执行。采集器原“整个IPC文件不变”断言仅调整为保护原有字段，显式允许新增尾部可选字段。

前轮前台耗时修正的本地结果：shadow 28项正常模式通过、28项 `-O` 通过；采集器34项通过（含单类 mixed 保护）。新增测试用可控时钟验证完整区间包含emit、成功/失败/空候选/间隔跳过各自计数、按实际候选规模分组、丢弃日志仍聚合、无递归emit、桶边界/合并/固定存储与缺失分位数。真实原生调度输出与队列/KV调用不变的既有测试继续运行。Python3.10语法、五个Bash块语法及git diff检查通过；未执行真实服务。

前轮两个验证CLI的 --help 已通过；本轮仅完善check_ipc，validate_cpu未修改。IPC诊断工具另有16项标准库测试（正常模式及-O通过），并检查--help、默认60/诊断300参数、Python3.10语法、当前README七个Bash块及既有服务参数断言。本轮没有改动运行时shadow/调度器。完整mixed测试在前轮因现有解释器缺pyarrow而导入受阻，本轮未安装依赖或声称其通过。完整日志保存在仓库外，不随源码推送。

```bash
python -B tests/benchmarks/test_r1_shadow.py -v
python -B tests/benchmarks/test_request_collector.py -v
python -B tests/benchmarks/test_r1_shadow_ipc.py -v
```

用户已报告服务器shadow runtime全部1440条B预测验证通过；本地没有重复执行真实模型。完善诊断后的真实Msgpack+spawn、此次超时的根因、ZMQ/服务端路径、shadow开关后真实GPU输出/吞吐/延迟/批次、取消时序和长时间压力仍待服务器验收。兼容工具的23文件哈希记录属于旧运行基线，不能用来声称新 Hook 文件仍与旧基线相同。本分支是待审查实现，不是已经完成的在线实验结论。
