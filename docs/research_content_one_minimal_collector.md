# 研究内容一：第一版最小采集器

本实现基于 `research-v0.9.1` 的
`84fda164294519fae14ff6730e2a2d9a433a358f`。实施前已有用户未跟踪的
`AGENTS.md` 和实施前审查报告，均未修改。本文说明实际实现，原审查报告保留为历史审查依据。

## 范围与配置

沿用原生 `benchmarks/benchmark_serving.py`、当前 mixed 数据集、V1 和 FCFS。
采集模式仅支持 `--backend vllm --endpoint /v1/completions --dataset-name mixed`，
每个 HTTP 请求一个文本 prompt、`n=1`，不支持 beam search、`best_of>1`、
profile 控制请求或 LoRA 模块。采集开启时，不支持的组合明确报错；关闭时不限制原有路径。

没有预测器、输出长度预测、类型专家、SLO 打分或 waiting 排序。
没有新增 Benchmark 主程序或必须读取的实验配置文件。

| 环境变量 | 默认值与含义 |
| --- | --- |
| `VLLM_R1_COLLECT` | 缺省或 `0` 关闭，`1` 开启；其他值报错 |
| `VLLM_R1_RUN_ID` | 开启时必填；1～128 个字母、数字、点、下划线或短横线，首字符为字母或数字 |
| `VLLM_R1_COLLECT_DIR` | 开启时必填；该进程可写的采集根目录 |

**一次服务进程生命周期对应一个 run_id。每次重新启动服务必须使用新的 run_id。**
服务端和客户端分别在各自启动前设置这三个变量，并使用相同 run_id；目录可以不同。
服务端在创建 AsyncLLM/Core 时固定配置，客户端环境变量不会修改已经启动的服务端。
服务端开启采集时必须保留原生统计，不得设置 `--disable-log-stats`。

服务端会校验采集客户端使用的引擎 ID 命名空间和 `n=1`。
因此采集中的该服务应专用于本次采集客户端，不应混入其他 API 客户端。
这不是自动探测远端服务版本或配置的协议；验收仍须检查实际启动日志。

关闭采集时不创建 writer、后台线程或首次记录集合，不聚合快照，不生成采集用 UUID。
客户端新增输入字段默认为 `None`。mixed 数据构造、参考长度/固定上限模式、
聊天模板、特殊 token 设置、请求顺序、全局到达 RNG 和原生指标公式均保持原样。
`--mixed-dry-run` 仍在采集初始化之前返回，不创建采集文件、不发送请求。

## 请求身份与文件

客户端在 Completions 发送器中、原生请求计时起点之前，针对本次发送生成 UUID：

```text
attempt_id = r1.<run_id>.<uuid4_hex>
HTTP X-Request-Id = attempt_id
API ID = cmpl-<attempt_id>
engine_request_id = cmpl-<attempt_id>-0
```

ID 不依赖 prompt 哈希、seed 或 Python/NumPy 全局 RNG。UUID 在实际进入发送器后生成，
不会给仅在客户端等待并发名额、尚未进入发送器的任务提前标记已发送。
这里的 attempt 表示进入 HTTP 发送尝试，不保证网络已送达或服务端已经接纳。

API/engine ID 按当前 vLLM 0.9.1 单 prompt、`n=1` 规则推导，
服务端记录中的引擎 ID 用于核对关联。没有改写 SSE 响应解析以另行读取响应 ID。
`extra_body` 仍只控制请求体；请求 ID 经 HTTP header 传递。

probe 与 main 分别使用独立上下文和不同 UUID。probe 引用首个 mixed 样本，
后续正式发送该样本时产生新的 attempt；重复运行同一个 prompt 也产生新 ID。
样本 metadata、真实 `task_type`、来源和哈希只保存在客户端，不进入请求头或请求体。

每个进程独占一个文件：

```text
<VLLM_R1_COLLECT_DIR>/<run_id>/client-<pid>-<instance_uuid>.jsonl
<VLLM_R1_COLLECT_DIR>/<run_id>/core-<pid>-<instance_uuid>.jsonl
<VLLM_R1_COLLECT_DIR>/<run_id>/frontend-<pid>-<instance_uuid>.jsonl
```

文件独占创建，不覆盖或共同追加。只有负责记录的客户端、Core 和前端进程创建 writer，
没有给每个 GPU worker 创建文件。文件句柄和 writer 不经 Core IPC 传递。

记录类型：

- `header`：schema、run_id、角色、PID、进程实例标识及相应配置。客户端记录 seed、
  预期 main/probe 数量、原生请求率/并发/采样配置、mixed 文件及 tokenizer 信息等。
- `attempt`：probe/main、完整 mixed 样本 metadata、attempt/API/engine ID、实际发送参数。
- `attempt_result`：既有客户端成功/错误、输出 token 数、latency 和 TTFT 值的副本。
- `first_admission`：引擎 ID 与首次准入快照。
- `finished`：引擎 ID、原生完成统计、原生时间戳副本、有效性和结束来源。
- `diagnostic`：取消、通用 abort、校验错误、生成错误、引擎死亡、输出处理错误或关闭时未完成。
- `footer`：正常关闭标记、记录计数、丢失计数及汇总开销。

客户端原生结果文件和指标格式没有增加采集字段。采集信息独立保存。

客户端 header 的 `request_rate` 在正无穷时记录为合法 JSON 字符串 `"inf"`，
有限请求率仍记录为数值。只转换采集元数据，`args.request_rate`、传给原生 Benchmark 的
请求率、到达过程和默认值均不变。writer 继续使用 `allow_nan=False`。

## 快照的准确位置和含义

快照位于 `Scheduler.schedule()` 的 running 处理之后、waiting 接纳之前。
仅在以下条件同时成立时计算，每轮至多一次：

```text
采集开启 && 本轮没有 preempted_reqs && 存在 waiting
&& token_budget > 0 && running 数量 < max_num_running_reqs
```

一次遍历取得两个总和，其余为标量读取：

| 字段 | 来源和解释 |
| --- | --- |
| `running_count` | `len(self.running)`，不包含本轮尚未接纳的 waiting |
| `running_computed_tokens` | running 的 `num_computed_tokens` 总和，尚未加上本轮 schedule 尾部的递增 |
| `running_remaining_prompt_tokens` | running 的 `max(num_prompt_tokens-num_computed_tokens, 0)` 总和 |
| `token_budget` | 已扣除本轮 running 安排后的剩余预算 |
| `kv_usage` | 块池使用比例，已反映本轮 running 分配/释放，不是整卡显存或算力利用率 |
| `decision_batch_id` | 当前 Core 文件实例标识加本地快照序号，同轮首次准入者相同 |
| `scheduled_timestamp` | 原生本轮较早生成的时间值，不新增一套准入生命周期时钟 |

该快照混合了调度逻辑进度与本轮安排的资源状态，不能整体称为 GPU 已完成状态。
批次队列存在时，先前安排的 token 也可能仍在执行。

只有原生成功分配、真正进入 running 的首次准入者才导出记录。
失败分配和异步等待远端 KV 不导出准入关联。恢复不覆盖首次记录；
即使异步状态恢复为 WAITING，也由采集侧首次标志保护。终止时清理该标志。
不以 `num_computed_tokens==0` 判断首次准入。

快照为不可变数值对象，写出前复制为只含数值/字符串的字典，不保存 live Request、
running 对象、整个 running ID 集合或整个 waiting。没有调用 `make_stats()`，
没有额外前缀命中查询或第二次 KV 分配。

## 完成统计和标签

前端在原生 `_update_stats_from_finished()` 执行后复制已生成的 `FinishedRequestStats`，
并附上仍在当前局部变量中的 `req_state.request_id`。没有修改统计实现或重算公式。

- `scheduled_ts` 是首个 SCHEDULED 事件使用的本轮较早时间，不是快照读取瞬间；恢复事件不覆盖它。
- `first_token_ts` 和 `last_token_ts` 来自 Core 输出批次时间。
- `inference_time` 为原生首次 SCHEDULED 到最后 NEW_TOKEN 批次的区间，包含中途抢占及恢复等待。
- 它不是精确逐请求 GPU 执行时间，也不是客户端 E2EL。Core 单调时钟不能与客户端时钟相减。

`native_stats` 保存原生值；`inference_time_label` 仅在原生完成统计和时间戳有效、
结束原因是 STOP/LENGTH 时提供。原生未初始化时间戳导出为 `null`，并附缺失字段和有效性。
缺失/非法统计的标签为 `null`，不填零，也不回退到客户端 latency。
无效记录中的原生统计数值只供审查，不能绕过 `stats_status` 当作训练标签。
单 token 的原生 `decode_time=0` 等合法零值保留。

正常 STOP 和 LENGTH 都保留。LENGTH 额外标记 `length_capped=true` 和
`actual_end_under_requested_limit`，表示指定限制下的实际结束，不保证自然回答完成。
原生 LENGTH 也可能由服务端模型上下文长度上限触发，不能仅据结束原因区分具体哪一种上限。

前端 stop string 使用 `terminal_origin=frontend_stop_string`，仍保存原生 STOP 统计。
它通知 Core abort 时不会走前端客户端取消记录路径；完成后发生的迟到断开也不重新标成取消。
客户端取消和错误通常没有 `FinishedRequestStats`，保存诊断及缺失状态，不制造完成标签。

## 有界写出、故障和关闭链路

每个 writer 队列容量为 1024 条。热路径非阻塞入队；后台线程负责 JSON 序列化和写出，
每 64 条或约 1 秒刷新，正常关闭时排空并刷新。
这些是第一版内部缓冲参数，不是研究实验规则，也未据此宣称开销足够小。

队列满不等待磁盘，累加丢失计数；每类错误仅首次输出错误日志。
I/O 失败会显式报告未确认写出的记录数，后续入队失败继续计数。
footer 能写出时包含汇总；不能写出时依靠可见错误及离线完整性检查识别。
普通 flush 不等于每条 fsync，不承诺断电耐久性。

实际关闭链路：

```text
API server 的 build_async_engine_client_from_engine_args() 上下文 finally
  → AsyncLLM.shutdown()
    → EngineCoreClient.shutdown() → 资源 finalizer → Core 进程管理器 close
      → SIGTERM；原生 Core 信号处理抛出 SystemExit
      → run_engine_core() 的 finally → EngineCore.shutdown()
        → 优先关闭 Core collector，再执行原生模型资源关闭
        → Scheduler.shutdown() 再次幂等关闭 collector
    → 取消前端输出 handler
    → 记录前端尚未完成请求的诊断，排空并关闭前端 writer

客户端 main() 中 asyncio.run(原生 benchmark) 的 finally
  → ClientCollector.close()
```

新增 `core.py` 的提前关闭是审查表之外的必要小修改：原生父进程只给 Core 约 5 秒退出，
模型执行器清理原本位于 Scheduler 关闭之前，可能耗尽排空机会。
没有增加 IPC、信号处理器或重构引擎生命周期。

writer 的单次关闭等待最多 2 秒；超时可见并标记未确认完整，重复 close 安全。
Core 仍可能被原生退出期限、SIGKILL 或异常退出打断，不能保证所有异常下都有 footer。
缺少 footer 的文件报告“不完整或未确认完整”，仍保留可读记录，不能推断全部数据已丢失。
footer 确认 writer 的关闭/写出情况，不表示所有推理请求都已正常完成。
应等待正式请求结束并正常停止服务后再做最终合并。

## 离线合并

若客户端和服务端目录不同，先将同一 run_id 的全部三类 shard 文件汇集到一个目录，
保留包含角色、PID、实例 UUID 的文件名。该步骤不需要共享写文件，也不建立在线协调服务。

以下仅是离线数据合并命令，不加载模型、不启动服务、不发送请求：

```bash
python vllm/v1/metrics/request_collector.py merge \
  --input-dir "/ABS/PATH/TO/COMBINED_RUN_SHARDS" \
  --output "/ABS/PATH/TO/NEW_merged.json"
```

输出文件独占创建，不覆盖已有文件或输入 shard。
合并使用 `(run_id, engine_request_id)` 的完整并集；不依赖文件/完成顺序或跨进程时钟。
重复关联保留全部原记录；缺少 client、admission、finished、结果的请求逐项列出状态。
取消、错误、未匹配请求、坏行及无合法关联键的记录保留在请求列表、文件问题或
`unassociated_records` 中，不做 inner join 丢弃。

- `recording_status`：文件及角色层面是否确认写出完整；缺 footer 等情况为 `incomplete_or_unconfirmed`。
- `collection_complete`：还要求逐客户端批次数量、请求关联和标签均完整。因此已明确记录的取消也会令它为 false，
  这不等于已证实发生记录丢失。
- 每条请求的 `status` 列出 missing/duplicate/client error/diagnostic 等原因。
- 每条请求同时保留原记录及其来源文件和行号。
- 只有关联完整且原生统计有效时，合并层 `inference_time_label` 可用；异常记录保留而不补零。

每次 Benchmark 已有一个独立 client shard。合并输出 `client_batches`，以现有
`(run_id, instance_id)` 标识该客户端实例，同时保留 `source_file` 和 header 的 `source_line`。
同一服务生命周期内可以有多份 client shard；不新增线上 session_id/batch_id，
不把调度 `decision_batch_id` 作为 Benchmark 批次。

每批分别比较 header 的 `expected_main_requests`、`expected_probe_requests` 与
`actual_unique_attempts.main/probe`；唯一请求按该 client 文件的 attempt 中的引擎请求 ID 计数，
不按结果数或合并后的总行数反推发送数量。重复 attempt 不增加同 phase 的唯一计数，且明确报错。
即使两批总数相等，一批不足也不能用另一批超出抵消。

预期数量必须是非负整数（不接受布尔值、字符串或浮点数），声明总数必须大于零。
缺失/非法预期、缺失实例标识、同 run_id 重复实例标识、非法 phase、无法关联的 attempt 身份、
有歧义的 client header、数量不足/超出或重复 attempt 均列出批次 `status`，
逐记录问题另附原始行号。原有请求级 missing/duplicate/diagnostic 检查继续保留。
`client_batch_counts_complete` 只有全部批次数量可核验且匹配时才为 true，
并成为 `collection_complete` 的必要条件。writer 文件关闭完整与数量核验结果分别表达。
因此，预期 120 main + 1 probe 却没有任何请求，或只有 1 main，不能判为采集完整。

## 开销测量与行为保护

footer 汇总 `snapshot_*`、`record_build_*`、`enqueue_*`、`client_prepare_*`，以及
`background_serialize_*`、`background_write_*`、`background_flush_*` 和关闭排空耗时。
没有为测量额外写逐调度轮日志。

前台耗时覆盖快照聚合、记录构造和入队；后台耗时覆盖序列化、写出和刷新。
footer 中的后台统计不包含 footer 自身尚未完成的序列化/写出/刷新，
计数维护和初始化等固定成本也需通过关闭/开启对照覆盖。
前后台存在重叠，不能把两类 CPU 耗时简单相加解释成端到端延迟。
后台线程仍有 GIL、CPU 和 I/O 竞争，不能声称零开销。

服务器验收应同时记录原生 Benchmark 指标、进程 CPU、写出量、计数和关闭排空情况，
在相同数据、参数、缓存/热身条件下对照关闭与开启采集。
本批没有测量 A800 上的实际开销或预设可接受阈值。

## 测试与验证记录

在仓库根目录、已有 Python 环境中运行，不使用会加载 GPU conftest 的 pytest 全量入口：

```bash
VLLM_R1_COLLECT=0 python -B tests/benchmarks/test_request_collector.py -v
VLLM_R1_COLLECT=0 python -m unittest discover \
  -s tests/benchmarks -p test_mixed_workload_dataset.py -v
```

首版历史验证结果：

- 新增 **23 项标准库 CPU 测试通过**：原生发送器与 probe/main、原生 API ID 赋值、n=1 添加请求、
  首次准入/恢复/失败分配、同轮数值快照、原生完成统计与恢复时间边界、stop string、取消/缺失、
  队列满、序列化/写出失败、正常/超时关闭、离线 CLI、重复/缺失/未匹配/坏行、关闭采集保护。
- 与官方 `v0.9.1` 的 Scheduler 使用相同模拟资源执行对照；关闭采集的调度输出和资源调用相同。
  原生到达函数、指标函数、统计公式文件及 Core IPC 定义与标签对照未改变。
- 原有 **15 项 mixed 测试通过**，包括真实本地合成 Parquet、HF 合成 tokenizer、特殊 token、
  两种输出上限模式、去重/冲突、局部 RNG、原生非 mixed 和 probe/main 构造保护。
- Ruff 检查通过，检查时使用 `--ignore UP045`，保留仓库既有 `Optional[...]` 注解风格；
  新文件格式检查、语法检查和 `git diff --check` 通过。

当前 Windows 环境的旧测试 Python 可执行文件曾被应用控制阻止；改用已具备的内置 Python，
只在本次测试进程内加入已有测试包目录，没有安装或升级包。
初始运行也遇到默认沙箱临时目录权限和异步 socketpair 限制；
测试通过工作区临时目录和不创建 socket 的 CPU 事件循环隔离解决，未改生产异步循环。
复验原 mixed 测试时，只在测试进程内应用上述临时目录/事件循环设置，未修改原测试文件。

本机实际使用的复验命令（PowerShell，仓库根目录；现有包目录已存在）：

```powershell
$collectorPython = 'C:\Users\scara\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $collectorPython -B tests/benchmarks/test_request_collector.py -v
& $collectorPython -B -c 'import sys,runpy,tempfile,asyncio; from pathlib import Path; root=Path.cwd(); tempfile.tempdir=str(root); helper=runpy.run_path("tests/benchmarks/test_request_collector.py"); asyncio.run=helper["run_async"]; sys.path.insert(0,r"C:\Users\scara\AppData\Local\Temp\vllm-mixed-validation-20261020\Lib\site-packages"); sys.argv=["test_mixed_workload_dataset.py","-v"]; runpy.run_path("tests/benchmarks/test_mixed_workload_dataset.py",run_name="__main__")'
```

测试按实际源码 AST 隔离未安装的生产依赖，HTTP、KV 和模型资源使用 stub。
它们不等于完整 vLLM 导入、真实 HTTP、msgpack/ZMQ、Linux SIGTERM 链或 GPU 验证。
现有环境没有 aiohttp/PyTorch；本次没有为运行完整入口安装依赖。

### 2026-10-22 两处局部修复回归

沿用上面的 `python -B tests/benchmarks/test_request_collector.py -v` CPU 入口，
本轮 **34 项测试通过**（保留原 23 项，新增 11 项）。新增覆盖预期 121 却零请求、
预期 120 main 却只有 1 main、probe 数量不足/超出、同 run_id 的两次 Benchmark
分别完整或一少一多、重复 attempt、非法 phase、预期数量无法核验和 client header/实例歧义。
使用现有实例和来源信息，未增加线上身份字段。

请求率回归通过 AST 执行真实 `benchmark_serving.main()`，使用真实 ClientCollector/writer
构造 header；仅将数据集、tokenizer 和 Benchmark 协程替换为 stub，不运行实际 Benchmark。
同时核对正无穷/有限请求率的严格 JSON、传给原生调用的数值未变，以及关闭采集时不创建文件。
Ruff 检查（仍使用 `--ignore UP045`）、新文件格式、语法和 `git diff --check` 通过。

本轮未重新运行原 mixed 15 项独立测试、真实 HTTP/IPC/信号退出或 GPU 验收；
原 mixed 的历史结果与本轮 collector 内的 mixed 行为保护测试分别保留，不混称为新验收结果。

## 120 请求服务器验收命令（本轮仅修订，未执行）

以下沿用用户已经跑通的命令，只增加采集所需的三个环境变量，并将 manifest 和结果文件
示例名称由 `collector_check_001` 改为 `collector_check_002`。入口、端口、模型和数据路径、
数量、seed、并发、请求率和生成参数均按原命令保留，不增加模型别名、上下文或 tokenizer 参数。

服务端终端，在启动服务之前设置：

```bash
conda activate vllm091-whb
cd "$HOME/whb/vllm091"

export VLLM_R1_COLLECT=1
export VLLM_R1_RUN_ID="collector_check_20261022_002"
export VLLM_R1_COLLECT_DIR="$HOME/whb/mixed_collector_records"

VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/whb/models/Meta-Llama-3-8B-Instruct" \
  --host 127.0.0.1 --port 8011 \
  --disable-log-requests --enable-chunked-prefill \
  --enforce-eager --tensor-parallel-size 1 --max-num-seqs 8
```

客户端在同一计算节点的另一个终端，使用与服务端相同的 run_id：

```bash
conda activate vllm091-whb
cd "$HOME/whb/vllm091"

export VLLM_R1_COLLECT=1
export VLLM_R1_RUN_ID="collector_check_20261022_002"
export VLLM_R1_COLLECT_DIR="$HOME/whb/mixed_collector_records"

python benchmarks/benchmark_serving.py \
  --model "$HOME/whb/models/Meta-Llama-3-8B-Instruct" \
  --backend vllm --host 127.0.0.1 --port 8011 \
  --endpoint /v1/completions --dataset-name mixed \
  --num-prompts 120 --seed 0 \
  --mixed-chat-count 40 --mixed-code-count 40 --mixed-summary-count 40 \
  --mixed-chat-path "$HOME/whb/hf_datasets/lmsys-chat-1m/data/train-00000-of-00006-4feeb3f83346a0e9.parquet" \
  --mixed-code-path "$HOME/whb/hf_datasets/CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl" \
  --mixed-summary-path "$HOME/whb/hf_datasets/cnn_dailymail/3.0.0/validation-00000-of-00001.parquet" \
  --request-rate 4 --max-concurrency 16 \
  --mixed-output-len 2048 --temperature 0.9 \
  --mixed-export-path "$HOME/whb/mixed_120_requests_llama_s8_c16_r4_collector_check_002.json" \
  --save-result --save-detailed \
  --result-dir "$HOME/whb/mixed_smoke_results" \
  --result-filename mixed_120_llama_s8_c16_r4_collector_check_002.json
```

两端同节点使用同一采集根目录，各进程仍独占不同 shard 文件，不共同追加一个文件。
运行前确认示例 run_id 目录、manifest 和结果输出名称未被先前运行占用；
若已存在，为新服务生命周期选择新的 run_id，并为本次 manifest 和结果另换新文件名，不覆盖旧产物。
不要添加 `--disable-log-stats`；客户端设置环境变量不能修改已启动服务端的采集配置。

run_id 继续表示服务进程生命周期，不表示一次 Benchmark 批次。
同一个服务不重启而再次运行 Benchmark 时，客户端继续使用该服务的 run_id，
仅为第二次 manifest 和结果换新名称，例如 `collector_check_003`；不要因此给客户端另换 run_id。
合并器使用现有 client shard 实例和来源信息分别核验两批 main/probe 数量；
不新增 session_id、Benchmark batch_id 或其他协议，也不将调度决策批次标识解释为 Benchmark 批次。

客户端结束并不表示服务端 writer 已关闭。需要最终合并时，先等待请求结束并正常停止服务，
再检查全部 shard 的 footer、关闭状态、丢失计数和请求关联。
一次 Benchmark 正常无错误时预期有 1 个 probe 和 120 个 main；probe 不计入正式 120 条。
probe 会沿原生流程影响热身和可能的前缀缓存，不自动重置缓存。

这些命令本轮只写入文档，没有执行。若验收没有触发取消或抢占恢复，不能声称已验证这些路径；
单次 120 请求验收也不能作为预测精度或调度收益结论。

## 尚存限制

- 未启动真实服务、训练或 Benchmark；未验证目标 A800、真实 HTTP/IPC/信号退出及采集开销。
- 多机/Ray 等部署的环境传递和文件汇集未经验证；不能仅凭本地进程隔离测试宣称支持。
- run_id 唯一生命周期规则由运行者配置；重启服务不能复用旧 run_id。
- 队列满、磁盘失败或强制终止可能留下缺失/未确认完整记录；合并不丢弃现有数据，也不伪造标签。
- LENGTH 的后续建模纳入规则尚未决定，采集层只保留原生结束原因和上限语义。
- 采集模式是专用的单 prompt、n=1 Completions 路径，未扩展到其他 backend/endpoint/并行采样。
