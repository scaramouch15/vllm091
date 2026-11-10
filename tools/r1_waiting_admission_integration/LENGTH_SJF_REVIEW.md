# 预测输出长度 P50 SJF：实现与验证边界

本实验基于 `20a76449b5f02b136192c62955c5cc7977803e3b`。它使用前端 B 长度头缓存在请求上的输出 token 长度 P50，按 `(P50, 原始位置)` 升序；P50 不是均值，也不是耗时。它不读取真实输出长度、未来耗时或真实类型来排序。该模式不提供等待时间或无饥饿保证。

## 配置与路径

显式设置 `VLLM_R1_ADMISSION_MODE=apply`、`VLLM_R1_ADMISSION_HEAD_POLICY=length_sjf`，设置既有 `VLLM_R1_SHADOW=1`、`VLLM_R1_SHADOW_MODEL`、`VLLM_R1_SHADOW_RUN_ID` 和 `VLLM_R1_SHADOW_DIR`。`VLLM_R1_ADMISSION_EVERY` 和 `VLLM_R1_ADMISSION_PREFIX_LIMIT` 与其他 admission 策略共用；实验需显式统一它们。`observe` 只记录计划、不应用；默认 mode 为 off、head policy 为 protected。`SKIP_SINGLE_SLOT=1` 与本策略冲突。feasible protection、fairness apply/alternate 和 fairness compare 需要 type_slo，不能和本策略组合；旧 SLO shadow 与本策略不可组合，以免将长度当成时间。prefill/type-rank shadow 的既有限制要求 admission off 或 type_slo，不能与 SJF 开启。本策略保留原生配置不支持时回退，而不是声称已执行 SJF。

前端 `Shadow.prepare` 只为其现有支持范围生成并缓存三分位长度；例如要求 max_tokens=2048、单文本且不使用特殊配置。前端记录 `shadow_static`（包含 `length_head_ns`、`text_transform_ns`、`static_compute_ns` 等）、`shadow_static_skip` 和 `shadow_static_error`。原模型 B 的固定 SHA 检查未修改；模型部署路径、训练数据和实验输入的重叠需另行核验。核心侧加载现有 B 工件；SJF 本身不调用动态耗时头、prefill 头或组装动态预测矩阵。这里不宣称模型加载没有开销。若整个 admission 因不支持配置被禁用，既有 shadow 回退行为可能继续调用旧 sweep，其成本不能算作本 SJF 决策的动态推理。

与 type_slo admission 共享每轮 running 之后的机会判定、EVERY 相位、PREFIX_LIMIT、连续首入队合格前缀以及排序后的原生准入和恢复路径。合格性仍要求有效 B 长度三分位；因而即使两组使用相同窗口参数，类型 SLO 策略还需要类型、SLO、时间、动态耗时和 prefill 等信息，而 SJF 只需要长度缓存。未来 LTR 若复用 `eligible` 将无必要地继承 B 长度依赖，须单独设计而非默认为公平比较。

缺失、非法或不支持的对象截断前缀，不跨越搜索；已准入或 PREEMPTED 请求不当作首次准入候选。对选出的前缀再次核对缓存、实例、轮次、有序 ID、数量、唯一性及实际队首引用，失败则原生回退。后续依然由原生循环进行容量、token 和 KV 检查；KV 失败不跳过继续搜。未准入窗口对象依既有逻辑恢复原序，skipped 按原生路径回填。observe 不调整队列。

## 日志与成本

`r1_admission_v1` 中 `score_rule=predicted_output_length_p50_sjf_v1`、`length_p50_scores` 与 `candidate_ids` 一一对应、`length_valid_count`、`checked`、`candidate_count`、`prefix_stop`/`prefix_stop_detail`、`reason` 描述覆盖与截断；`sorted_ids` 和 `order_changed` 是计划，`applied_order_changed` 是队列计划是否应用，`window_attempts`、`window_admitted`、`actual_admission_order` 是实际尝试与准入。单个准入对象是否越过原队首，应结合其 ID 与原候选判断；不能仅凭计划顺序变化宣称实际变化。SJF 不写入时间预测 `scores_p90`，也不增加动态 `prediction_successes`；首次准入日志 `predicted_in_this_batch=false` 表示没有本轮动态预测，并保留入口快照。

`trigger_prefix_ns` 是机会检查与候选扫描；`cached_length_read_ns` 包括 running/KV 快照读取、缓存校验和长度决策构造；`decision_ns` 包括身份复核、稳定排序和决策校验；`apply_ns` 和 `restore_ns` 是队列应用和恢复。已有 `admission_log_foreground_v1` 是前台日志处理，`admission_schedule_foreground_v1` 是完整调度前台区间；后者包含子阶段，不应将分项分位数相加。前端静态预测按上述独立事件统计，不是 scheduler 阶段开销，writer 的后台写盘也不包含在前台入队计时中。

## 对照与限制

原生 FCFS（研究预测和日志关闭）描述生产基线；预测与观察打开但不应用排序的 FCFS 有额外预测/日志成本，可用于隔离顺序变化与观测成本，但并不等于无研究扩展的原生 FCFS。固定输入、负载、硬件和相同 opportunity、窗口、检查完整性与所有失败原因后，仍需分别报告可排序覆盖。benchmark 客户端 attempt 与服务端原生 QUEUED/arrival 信息来自不同采样点和可能不同的时钟；现有 attempt 记录不能单凭它直接精确重建逐次跨机发送到服务端接收延迟，需核验相应时间戳和时钟同步，不修改 benchmark 到达流程。

本次为 CPU 测试和源码实现，尚无真实模型、Linux vLLM 进程、GPU 资源管理或在线服务验证。运行前须审查代码，核验 B 的实际文件 SHA、训练与 seed 清单重叠、运行环境、日志完整性及候选覆盖，再由操作者进行小规模服务测试；不能由此宣称吞吐、延迟或 Goodput 收益。
