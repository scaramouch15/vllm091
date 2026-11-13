# 论文 LTR 模型的 vLLM 0.9.1 V1 waiting 前缀适配

本实验默认关闭。它不复现作者原版对 waiting、running、swapped 的联合调度，也不使用真实类型、真实输出长度或未来时间作为输入。当前支持范围沿用 admission 的单卡、原生 FCFS、普通 decoder 配置边界；unsupported 配置按既有机制记录并回退。

## 启用与资源

使用已核验并与 vLLM 兼容的独立服务环境。显式设置：

```bash
export VLLM_R1_LTR_ENABLE=1
export VLLM_R1_LTR_MODEL_DIR=/path/to/verified/finetuned
export VLLM_R1_LTR_TOKENIZER_DIR=/path/to/verified/opt-tokenizer
export VLLM_R1_ADMISSION_MODE=apply   # observe 只记录计划
export VLLM_R1_ADMISSION_HEAD_POLICY=paper_ltr
export VLLM_R1_ADMISSION_EVERY=1      # 示例，不是推荐策略
export VLLM_R1_ADMISSION_PREFIX_LIMIT=32
export VLLM_R1_LTR_MAX_INFLIGHT=8
export VLLM_R1_LTR_TIMEOUT_SECONDS=5
export VLLM_R1_SHADOW=0
export VLLM_R1_TYPE_SLO=0
```

模型 SHA256 固定为 `e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab`，checkpoint revision 为 `39df2b41ffe88d5ed967c6035d3838b5b5960379`；tokenizer revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`。tokenizer 五个文件和权重在加载前校验；这些外部文件不会由 Git 包含。加载前路径需处于可信、不可并发替换的部署目录：现有加载代码先按路径校验，再由 Transformers 按路径读取，尚未消除校验与加载之间的替换窗口。

仓库 `requirements/common.txt` 要求 transformers >=4.51.1 和 tokenizers >=0.21.1。本机在独立 target 目录测试 transformers 4.52.4、tokenizers 0.21.1，与现有 CPU PyTorch 一起通过真实 checkpoint 文本探针；这不证明 g10 服务环境或其 GPU 栈兼容。不要向已有服务环境覆盖旧版本；请在单独的服务虚拟环境中安装与该服务器 vLLM 构建兼容的一组依赖，再核对实际 import 路径。先人工审查并做小规模验证，不能凭本机探针推断收益。

## 数据流与语义

Completion 的预处理返回标准化文本；仅单文本、单输出、无 beam/LoRA/远端 KV 等受支持请求尝试评分。前端启动时只加载一次 CPU OPT，评分使用单 worker 和有界在途数，队列满立即回退，超时后正在执行的工作仍占用名额直至完成；取消传播给调用方，已运行线程继续完成并丢弃结果。服务关闭拒绝新工作并取消排队任务，运行中的线程会完成。默认关闭时不创建 LTR 模型或评分器。

同步 tokenization/推理在 worker 中，每个请求评分一次，不在 Scheduler 每轮推理。评分结果通过仅可由进程内部构造的 `ScoreResult` 暂存于采样参数，Processor 消费并检查 request_id、有限性，再写入 `EngineCoreRequest.r1_ltr_score`。Core 解码后将其放入 Request，完成或取消后随请求释放；不跨请求或跨轮缓存。评分失败、队列满、超时、非有限值及不支持请求不会携带有效分数。日志包含请求 ID、tokenization/inference/前台等待纳秒及失败类别，不含 prompt。模型和 tokenizer revision、模型摘要、设备和 dtype 在启动时记录。

调度器以 `paper_ltr` 排他 head policy 使用缓存分数，按 `(-score, 原位置)` 稳定排序。共同资格包括 WAITING、从未准入、尚无计算/输出、无特殊状态等；LTR 分支不依赖 B 模型长度缓存或 shadow writer。不合格或缺失分数截断前缀，空前缀原生回退；身份、重复 ID 或前缀不一致在队列应用前整窗回退。保留原生容量、token、KV、抢占、skipped 以及现有窗口恢复。与 `SKIP_SINGLE_SLOT=1`、SLO shadow、fairness、type SLO 和 B shadow 同时开启将报错；SJF 与 LTR 通过唯一 head policy 互斥。

`R1_LTR r1_admission_v1` 记录候选、检查数、`prefix_stop_detail`、原顺序、分数、计划顺序、是否应用、尝试和实际准入、退出原因及计时。`trigger_prefix_ns` 包含机会/扫描，`ltr_cache_read_ns` 包含运行状态快照及分数读取，`decision_ns` 包含身份检查和排序，`apply_ns` 和 `restore_ns` 为队列修改和恢复，`admission_schedule_foreground_v1` 包含这些子阶段；不能相加分位数。输出走 Python 日志，依赖日志级别和运行环境；不具备异步 writer 的完整性 footer/丢失计数。前端 `foreground_ns` 包含等待与评分，包含其 tokenization/inference 子区间。

## 验证限制

真实 checkpoint CPU 检查、mock Scheduler 方法测试、AST 提取的实际 Processor 方法测试和真实 msgspec 字节编码／子进程 echo 测试分别报告；它们不证明完整 EngineCore IPC 或 GPU 服务运行。作者 AUXLLM GPU 路径尚未直接验证。由于配置和 tokenization、模型数值可能在目标环境不同，服务器复测前须先核对模型和 tokenizer 摘要、Python/torch/transformers/tokenizers/safetensors 版本、代码提交及导入路径，并验证日志是否完整。现有训练身份清单无法证实 LTR 与 seed=1/2 无重叠。无法由本适配宣称吞吐、Goodput、SLO 或公平性收益。

## 评分诊断字段（diagnostic_v1）

`R1_LTR diagnostic_v1` 以 request_id 记录 `submitted`、`worker_started` 和 `worker_finished` / `worker_failed`。`queue_wait_ns` 从提交前采样到 worker 开始，包含线程调度和提交记账，不是精确的纯队列时间；正常结束另记录 `tokenizer_ns`、`inference_ns`。调用方超时或取消时，`caller_exit` 包含稳定 reason、`foreground_ns` 和该瞬间的 `still_running`；worker 后续结束的 `late` 表示在结束记账时已观察到调用方退出。两条日志是并发事件，不是资源事务回滚。评分器维持原有单 worker、有界在途量，超时后的运行任务仍占容量直至结束，结果不会转交其他请求。

completion 失败原因使用稳定值 `queue_full`、`duplicate_request_id`、`scorer_closed`、`invalid_score_identity`、`timeout`、`cancelled` 和 `model_exception`；后者额外记录异常类型，不记录原始异常消息或 prompt。服务端仍用普通 Python 日志，日志等级、缓冲和丢失不受 shadow JSONL footer/丢失计数机制保证；按请求 ID 联合客户端 attempt/result 和 admission 记录统计缺失、重复及未关联，并明确日志完整性仍未知。

`cpu_queue_diagnostic.py` 是使用固定合成文本和真实 checkpoint 的本机 CPU 比较工具，需单独提供模型路径、tokenizer 路径与新输出路径；输出每次请求、worker 事件、线程数及机器信息。它使用一个 worker、有限在途数和超时，不代表 GPU 服务或不同节点的运行成本。CPU 的瞬时突发实验不能独立确定生产参数。
