# RC1 waiting 准入实验（代码审查阶段）

该实验在原生 V1 waiting 循环前对有限连续前缀做一次预测；P90 仅为机制和开销对照，不是最终调度公式。不是 SLO、等待公平性或收益证明。原生 api_server 和 `benchmarks/benchmark_serving.py` 不变；不依赖类型标签。

## 配置及回退

| 研究环境变量 | 语义 |
|---|---|
| `VLLM_R1_ADMISSION_MODE` | `off`（默认）、`observe`、`apply`，其他值启动失败。off 保留原生及既有 shadow sweep。observe 计算决策但不改 waiting；apply 使用同一逻辑改变本轮窗口尝试次序。|
| `VLLM_R1_ADMISSION_EVERY` | 正整数，默认 8。策略机会 1、9、17… 更新，其余机会不提取候选，不预测。|
| `VLLM_R1_ADMISSION_PREFIX_LIMIT` | 1–4096，默认 32；最多检查该数位置，遇首个不合格对象截断，不读取 P+1。|
| `VLLM_R1_ADMISSION_HEAD_POLICY` | `protected`（默认）或 `ranked`，非法值启动失败。protected 固定候选队首，其余按 P90 稳定升序；ranked 对整个候选窗口按 P90 稳定升序，同分均保留原位置。|
| `VLLM_R1_ADMISSION_SKIP_SINGLE_SLOT` | `0`（默认）或 `1`，非法值启动报错。仅在 observe/apply 的抽样更新机会启用；`1` 时合格前缀非空且 waiting 准入前只有一个空闲 running 名额，跳过动态组装、推理、排序和窗口应用/恢复。|

上述参数均在 Scheduler 初始化时解析，不支持热更新；非法数值启动失败。observe/apply 需要原有 `VLLM_R1_SHADOW=1`、shadow 的模型路径、run ID、目录及其已有 Python/依赖/线程门禁，以复用前端静态长度、Core 动态模型和 writer。shadow 关闭时打印 `shadow_disabled` 并原生回退，不自动启用预测。off 与 shadow 开/关的旧组合继续按原路径运行，新增一次 Python 调用包装及配置判断，不能声称零开销。新模式启用时不执行旧 `shadow.sweep` 或旧 P90 排序，即使旧 `VLLM_R1_SHADOW_SORT=1`；不会再次批量预测。旧 shadow 的频率、128 扫描上限和跳过无效候选逻辑只在 off 模式保留。8/32 只是首轮值；旧 8/128/32 不是最终参数。

`observe`/`apply` 下 `HEAD_POLICY=ranked` 与 `SKIP_SINGLE_SLOT=1` 冲突并在启动时明确报错：ranked 的单名额也可能需要比较候选。`protected` 保留现有单名额跳过行为。admission `off` 不启用研究排序，可解析上述组合但不触发策略。后续 protected/ranked 对照拟统一采用 `EVERY=1`、`PREFIX_LIMIT=32`、`SKIP_SINGLE_SLOT=0`；这是实验配置，并非适用于所有负载的最终参数。

支持范围由 `r1_admission.unsupported_reason` 根据真实配置字段检查：原生 V1 FCFS Scheduler、普通文本 decoder、DP=PP=TP=1，无 KV connector、speculative、LoRA 配置；并沿用现有 B 工件验证所需的 Meta-Llama-3-8B-Instruct 模型/tokenizer、Llama 架构及不 lower-case 条件。不支持时前端和 Core 的 shadow 加载均回退，记录可辨识启动警告，继续原生。该检查位于内置 Scheduler 和现有 shadow 创建路径；如果配置了其他 Scheduler 且 shadow 关闭，那个外部 Scheduler 不运行本代码，因此本模块无法为其提供独立启动警告。已启用 shadow 的模型工件哈希、Python/包版本及线程检查仍执行，错误会按已有 shadow 行为报错，不作为配置回退。

## 调用与队列契约

Scheduler 每轮产生单调 `schedule_id`，每实例使用随机 UUID。running 阶段之后，无本轮抢占、有 waiting、token budget 为正、running 未满时递增独立 `opportunity_id`，队首不合格亦计入；空前缀及预测失败不撤销机会。计数与成功预测数及已有 shadow batch 是不同量。

候选须为 WAITING、静态长度有效、`r1_shadow_admitted is False`、零输出和零 computed tokens、无多模态/encoder/LoRA/structured output/KV transfer。缺少 admitted 标志会拒绝，不能放行未知对象。共享一次 running count 和 KV usage 快照，用已有 `Predictor.assemble` 与 `dynamic` 的六个原特征批量预测一次，P90 是准入到完成时间预测，不是等待时间或客户端 E2EL。结果携带实例 UUID、schedule ID、有序候选 ID 和分数；应用前验证身份、长度、唯一性、有限非负值，并仅遍历前缀再次比对对象身份。异常和无效结果只在预测/决策边界捕获，在修改队列前回退；不会捕获原生 KV、状态维护等异常。

apply 的 protected 模式固定候选队首；ranked 模式将队首纳入整个合格窗口的 `(P90, 原位置)` 稳定升序。只取出并重新放入有限窗口，不跨不合格对象；observe 计算所选模式的顺序，但不应用或恢复队列。原生循环负责 capacity、token、prefix cache、KV 分配、preemption 和状态推进。KV 返回 None 时仍 break。容量/token/KV 退出后、原生 skipped 回填前，按原序恢复窗口未准入对象；仅扫描窗口残留。若残留与 skipped 同时存在、或残留身份不匹配，抛出不变量错误，不丢弃对象或伪装资源回滚。窗口全部消费后继续原生处理尾部，不开启第二窗口。原生资源异常透传，队列恢复不是资源事务；改变尝试顺序可能改变资源分配结果。

单名额开关在 running 调度完成后，以 `max_num_running_reqs - len(running)` 判断，包含本轮未分配 token 的 running 请求；不使用本轮 scheduled-running 数量或事后 `window_M`。原有机会计数先递增，EVERY 再抽样、前缀资格检查随后执行；仅对原本会预测的非空合格前缀、空闲名额恰好为 1 的机会生效。故 EVERY=8 时第 1 次即使被跳过，第 2 次仍是 `interval_skip`，第 9 次才进入下一次更新。`empty_prefix` 和普通特殊请求的原生 skip 语义不变；合格队首 KV 分配失败依旧 break。此优化不读旧预测、不启动额外 shadow sweep，不改变前端静态预测。

单名额跳过仍读取当前 running 数和 KV usage，构造真实 waiting 准入前 snapshot，传递 `(snapshot, ())` 给原有 `shadow.admit`：空 ID 表示本次没有预测，`predicted_in_this_batch=false`，不会伪造分数或标记候选已预测。首次准入和终止记录继续走既有路径；如果读取状态快照失败，记录 `snapshot_failed` 并原生回退，不声称获得了完整快照。该步骤只验证当前保护队首规则下的链路和潜在开销变化，不证明 E2EL 收益、所有负载的开销或最终公式；若以后队首保护或失败规则改变，须重新评估此优化。服务器后续对照可以由用户手动沿用 admission EVERY=1、PREFIX_LIMIT=32 及既有负载；旧影子 8/128/32 不与当前 admission 参数混同。

请求新增/取消的串行假设及特殊状态分析见 `tools/r1_waiting_admission_design/DESIGN.md`；其他 Scheduler、并发重入和真实 KV/connector 环境未经验证。首次准入标志由已有 shadow 的 Request 生命周期跟踪；不应把 score 或决策跨轮复用。

## 记录与计时

每个启用且受支持的 schedule 完成时经现有异步 writer 生成 `r1_admission_v1`。记录 mode、规则、every/prefix_limit、实例、schedule 和机会 ID、waiting_size、checked、prefix_stop、candidate_count、候选和排序 ID、P90 分数、是否变序和实际应用、窗口 `window_A/window_M` 与尝试/准入 ID、尾部 `tail_A/tail_M` 和准入 ID、`actual_admission_order`、reason、exit_reason、预测成功总数、running/KV 快照和 token budget。尾部尝试只保留数量，避免在大量原生 skip 时为日志建立随 waiting 长度增长的 ID 列表。日志不包含 prompt 或 Request 引用，writer 丢失由既有 summary/footer 完整性字段报告。

每条记录含 `head_policy`，`score_rule` 分别为 `head_then_p90_ascending_stable_v1`（protected）和 `all_p90_ascending_stable_v1`（ranked）。`order_changed` 只说明计算顺序相对 `candidate_ids` 变化，`applied_order_changed` 说明顺序是否应用；两者均不等于实际准入变化。

日志完整时，可用 `available_running_slots == 1`、非空 `candidate_ids`、`window_admitted` 与 `candidate_ids[0]` 对照，识别该窗口实际准入了非原队首的案例；应结合 mode、reason 和 exit_reason，不能用 `sorted_ids` 推断已经准入。可在同一服务实例和 run 的后续记录中按请求 ID 查找该原队首的候选、首次准入及 terminal 记录，观察有记录的推进；间隔跳过、窗口外请求、取消、缺失或丢日志限制覆盖，无法仅凭未出现判断仍在等待或证明公平性，也不能由这些字段计算可靠等待时长。`window_admitted` 和 `actual_admission_order` 的 ID 次序可区分已记录的准入顺序与集合；跨运行比较两者时，需配对请求和配置，并考虑随机输出、资源状态和执行轨迹差异，不能据此构造同一时刻的反事实准入集合。检查 summary/footer 完整性，否则连这些观察都可能不完整。

新增 `skip_single_slot` 和 `available_running_slots` 字段。抽样且合格的单名额机会记录 `reason=single_slot_skip`，与 `interval_skip` 分开；`prediction_ns`、`decision_ns`、`apply_ns`、`restore_ns` 均为 0，预测成功计数不增加，`scores_p90` 和 `sorted_ids` 为空。开关判断、有限扫描及该路径的快照构造计入 `trigger_prefix_ns`；额外日志构造和入队计入 `admission_log_foreground_v1`，外围 `admission_schedule_foreground_v1` 包含前述前台开销。前台路径不包含后台写盘、flush、自身最终直方图记账和最终函数返回；不要把 schedule 总耗时当作纯预测或排序耗时，也不能与内部区间相加。

记录中的 `trigger_prefix_ns` 从 prepare 入口至机会判定/前缀提取结束；`prediction_ns` 包括状态读取、矩阵组装和动态预测；`decision_ns` 包括身份/前缀复核、分数检查、稳定排序和列表构造；`apply_ns` 只含窗口队列应用；`restore_ns` 只含窗口恢复。它们是顺序子区间，不包含方法调用间隙，也不包括整个原生 waiting 循环。shadow summary 中 `admission_log_foreground_v1` 从该次记录构造到 emit 结束，`admission_schedule_foreground_v1` 从 schedule 包装入口至日志及日志直方图记账之后；后者包含上述子区间、原生调度和日志，不能与子区间相加。两个前台直方图各自不含自身最后直方图记账；schedule 总墙钟还不含最终返回及自身记账。后台序列化、写盘和 flush 由 writer 单独统计，CPU/GIL 争用仍可影响前台墙钟。分位数为桶区间，不是精确样本。失败时不会发布成功的 schedule 记录；无统计并不代表零开销。

## CPU 测试

```bash
python -B tests/benchmarks/test_r1_admission.py -v
python -B tests/benchmarks/test_r1_shadow.py -v
python -B tools/r1_waiting_admission_design/test_model.py -v
python -B tools/r1_selection_microbench/test_selection.py -v
```

集成测试从实际 `scheduler.py` 抽取执行 Scheduler 类，使用可控 Request、KV、预测器和 writer 替身；它执行真实 `schedule` 方法，但不是完整 vLLM 导入或真实 GPU 服务。独立状态模型不证明生产资源和并发安全。未进行服务端 CPU 或 GPU 实验，不能据此声称 E2EL 收益、公平性保证或大负载开销可控。

这是队首策略消融，不是完整公平性策略；ranked 暂无新增等待时间上界或防饥饿保证，120 请求有限实验不能证明长期公平性。先完成只读审查，再由用户手动运行实验；目前不宣称 E2EL、吞吐或 SLO 收益。
