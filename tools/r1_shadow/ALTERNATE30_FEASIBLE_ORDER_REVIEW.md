# alternate30 可行组排序审查与交付

基准提交：`20a76449b5f02b136192c62955c5cc7977803e3b`。
输入实现：`2988610df1d641f691636cd613871748c73ace29`，来自 deadline-slack bundle。
本地独立仓库分支：`review/alternate30-feasible-slack`。这项改动针对 alternate30，LTR 仍是独立基线。

## 审查结论

输入提交的普通执行路径没有发现改变等待保护组、组间交替、原生准入、抢占或 skipped 的代码；相对基准仅修改 admission 的可行组 key、配置和日志。`scheduler.py`、`r1_shadow.py`、scheduler interface/output、预测器和 LTR 实现均未修改。

可行组：等待少于 30 秒且 score >= 0；等待 >=30 秒的请求无论 score 如何均进入保护组并维持原位置顺序；其余负 score 请求维持原位置顺序。score 为 `slack / slo`，slack 为 `slo_s - wait_s - prediction_p90_s`。Chat 使用 TTFT SLO 和 prefill P90，Code/Summary 使用 E2E SLO 和对应 P90；这不是对 Chat TPOT 的预测。全部输入是当前调度快照已有的信息，不读取未来完成或旧日志的事后间隔。

| 参数值 | 可行组内部 key |
|---|---|
| normalized（默认） | `(score, position)`，与原 alternate30 相同 |
| absolute | `(slack_seconds, position)` |
| critical | 先 `slack_seconds <= H` 的请求，按绝对 slack 秒数、position 排；非临界请求仍按 score、position 排 |

临界窗口 H 为显式的正有限秒数（等号计入临界），不能由事后准入间隔自动推导。只改变可行组占据的槽位中的请求顺序；保护与可行组的交替形状和负分组尾部不变。只有实际成功准入才更新交替游标，失败准入、preemption、skipped、prefix 外 tail 均沿用基准规则。

输入实现的不足是非法策略名可能在独立 helper 中静默作为 critical 处理，以及缺少窗口边界、默认等价和新策略的原生路径覆盖。本次在 helper 边界明确拒绝非法模式/窗口，补充这些测试；未调整合法策略的排序定义。默认等价指合法输入下组别、排序结果、准入与游标行为；新增日志字段与排序耗时不可能与旧运行逐字节相同，未证明 GPU 运行时序或最终达标数完全一致。

## 验证

Python 3.14.6，stdlib CPU 测试，不加载模型或真实 GPU/KV 资源。

- `python -m unittest discover -s tests/benchmarks -p test_r1_feasible_order_review.py -v`：27 项通过。含对 pinned 基准源码 AST 提取的原排序函数的 400 个确定随机场景、两种起始游标与默认/显式 normalized 差分；零 slack、H 边界、稳定 ties、空组、组槽位、无副作用、显式配置/非法输入；absolute/critical 重跑已有原生 Scheduler 方法的 KV/容量/token budget/抢占/skipped/异常恢复和游标测试。
- `python -m unittest discover -s tests/benchmarks -p 'test_r1_fairness*.py' -v`：26 项通过。
- `python -m unittest discover -s tests/benchmarks -p test_r1_type_slo_admission.py -v`：38 项通过。
- `git diff --check`：通过。CPU harness 调用真实 Scheduler 方法，但预测器、KV 和模型资源是替身；没有 Slurm、节点或 GPU 实验。

测试对归一化／秒数排序安排了不同的可行组实际准入结果，同时检查保护请求仍在同一交替位置，剩余队列恢复及游标不变。默认差分测试依赖仓库中能读取上述基准 Git 对象，交付完整 bundle 包含它。

## 在线参数

保持现有 alternate30 的预测器、SLO、候选前缀、频率、原生调度和负载参数；其已有开关包括 `VLLM_R1_ADMISSION_MODE=apply`、`VLLM_R1_ADMISSION_HEAD_POLICY=type_slo`、`VLLM_R1_ADMISSION_TIME_OBSERVE=1`、`VLLM_R1_FAIRNESS_APPLY=1`、`VLLM_R1_FAIRNESS_ALTERNATE=1`、`VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS=30`。不要开启冲突的 feasible_protection/observe-comparison；其余模型、prefill 和 type-SLO 初始化要求沿用原实验。

只在启动服务进程前切换下列参数；当前进程读取配置后不会热更新：

```bash
# 原 alternate30 顺序（默认，亦可 unset 此变量）
export VLLM_R1_FEASIBLE_ORDER=normalized
unset VLLM_R1_CRITICAL_HORIZON_SECONDS

# 或绝对松弛秒数
export VLLM_R1_FEASIBLE_ORDER=absolute
unset VLLM_R1_CRITICAL_HORIZON_SECONDS

# 或临界优先；0.5 仅示例，必须在实验前预先确定并记录 H
export VLLM_R1_FEASIBLE_ORDER=critical
export VLLM_R1_CRITICAL_HORIZON_SECONDS=0.5
```

`r1_admission_v1` 中的 feasible_order、critical_horizon_seconds、sorted_ids、actual_admission_order、fairness_alternate_turn_before/after 用来核对实际策略和准入。不能将 planned order 当作已准入，也不能根据 GPU 未验证的代码推测在线 SLO 达标。
