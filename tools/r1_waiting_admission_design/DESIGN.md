# waiting 准入接入设计与独立状态验证

## 事实基线和范围

本文源码定位固定于 [`970cd83491feb2e80c4d150a96ae4b84cbd76558`](https://github.com/scaramouch15/vllm091/tree/970cd83491feb2e80c4d150a96ae4b84cbd76558)，它包含影子排序提交 `b6a80050b5c22402b023fd1e06882f8f61285cd0`。这里只提出设计并运行独立 CPU 状态模型；未接入生产，也没有使用生产 Scheduler 做运行正确性验证。P90 仅可能是未来分数来源之一，不是最终公式。实际读取的 `C:\Users\scara\Desktop\reasearch\vllm091\AGENTS.md` 是本地未跟踪文件；本文件没有依据声称 GitHub 包含它。

## 建议的单轮契约

1. 原生 running 阶段结束后，仅当无本轮抢占、waiting 非空、token 预算大于 0 且 running 未满，并且配置支持时，计为一次**真实策略机会**。独立于已有 shadow 机会计数。无更新机会则不改变队列，让原生循环继续。
2. 设新配置 `prefix_limit=P`（名字仅为设计占位），从队首最多检查 P 个位置，遇首个不合格位置立即停止；最多预测 P 个，不为了旧 128 扫描额外位置。若不合格对象在前 P 个位置内，它本身也计入实际检查数；达到 P 后不额外窥视 P+1。P=0 不启用。记录实际检查数、候选数及截断原因；P、预测频率和窗口大小尚未批准为产品参数。
3. 候选限于本轮普通文本、首次准入、静态长度有效的 `WAITING`。保守的请求级检查候选：`r1_shadow_admitted is False` 且静态三个分位数有效，`num_output_tokens == 0`、`num_computed_tokens == 0`、无 encoder/multimodal 输入、无 LoRA、无 structured output、无 KV transfer 参数；这些条件仍需生产实现逐项核验，不能仅凭 `num_computed_tokens` 判定首次准入。遇不符合条件的请求截断，队首不符合时直接由原生处理整条队列。
4. 同一时刻捕获一次状态快照，为该前缀批量计算**本轮**分数。独立模型在应用前检查 `round_id`、有序候选 ID 元组与实际前缀完全相等、ID 无重复、ID 和分数数量均与前缀一致，以及分数非负有限；不按集合匹配或重新拼接分数。预测跳过、异常、非法输出、身份不一致或过期时直接原生顺序；不读取旧日志、旧分数或跨轮缓存。分数由 CPU 模型调用者人工提供；未来若复用预测头，需要新接口提供本批数字，不重做模型推断。
5. 应用时，队首固定首个尝试；其余按 `(score, 原始位置)` 稳定升序。例：原 `[A,B,C,D,尾]`，分数 `[9,3,1,1]` → 本轮尝试 `[A,C,D,B]`；若只准入 A、C，恢复 `[B,D,尾]`。相同分数 C、D 按原次序。应用应和原生 waiting 检查循环共享，不复制或重写准入实现。
6. 循环因容量、token 或 KV 失败退出时，在**原生 skipped 最终回填之前**，将窗口内仍 waiting 的请求按原顺序恢复到原尾部之前；成功准入的对象不恢复。当窗口全部成功消费且仍有资源，以原生逻辑继续处理窗口外队首，不建立第二个排序窗口、不再次预测；特殊对象可按原生 skip/回填。不能从失败的候选跳过继续寻找其他对象。
7. 这是队列一致性设计，不是资源事务：仅预测/排序/决策校验在首次队列修改前失败可退回原生；原生 `allocate_slots`、connector 更新或状态变更之后抛异常，不能只恢复 deque 后声称 KV、running 或其他资源已回滚。此阶段未定义可恢复事务；未来实现应维护原生失败/终止语义，不吞异常并伪装成功，资源异常处理需单独设计和测试。

建议首版配置级回退：仅原生 V1 FCFS Scheduler、单文本 decoder、DP=PP=TP=1、无 KV connector、无 speculative decoding、无 LoRA 配置；其他配置整轮走原生。请求级对 FSM/structured、remote KV、PREEMPTED、LoRA、encoder/多模态、无有效静态信息等在前缀处截断；一旦到尾部继续运行，全部仍由原生分支处理。上述收窄是本设计安全边界，不是说这些功能不存在。是否要求此配置级范围、如何暴露开关及 exact predicate 均须在正式实现前批准。

## 固定提交源码映射

下表中“恢复位置”是建议设计，不是现有代码已实现的功能。所有行号是上述固定提交，不以本地未来 HEAD 自动更新。

| 分支 / 退出原因 | 建议窗口处理与恢复位置 | 原生行为及源码 |
|---|---|---|
| running KV 不足导致抢占 | 不构建策略窗口；原顺序保留 | 从 running 尾弹出、free、设 `PREEMPTED`、`waiting.appendleft`；本轮 waiting 循环被禁用。[scheduler 268](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L268)、[364](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L364) |
| 已存在 PREEMPTED 在前缀 | 截断；队首则完全原生，不能把其作为首次准入预测 | 原生在其成功分配后放入 `scheduled_resumed_reqs`；抢占对象从队首恢复。[scheduler 487](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L487)、[511](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L511) |
| `WAITING_FOR_REMOTE_KVS` / `WAITING_FOR_FSM` | 前缀截断；之后由原生判 ready 或跳过；窗口幸存者在 skipped 回填前恢复 | 未就绪 `popleft`，`skipped_waiting_requests.appendleft`；已就绪转 `WAITING` 并继续。[scheduler 369](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L369) |
| LoRA 数量约束 | 首版配置回退或请求截断，禁止策略绕过 | LoRA 过限时移至 skipped 并继续；最终回填。[scheduler 395](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L395) |
| encoder / 多模态不能安排 | 请求截断；首版不改变此分支 | 原生检查 encoder budget/cache；若 `num_new_tokens==0` 则 break。[scheduler 453](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L453)、[668](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L668) |
| KV connector / 异步加载 | 配置级回退；不能假设可用普通窗口模型 | 原生查外部匹配、更新 connector、分配后可转 remote waiting 并进 skipped。[scheduler 406](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L406)、[476](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L476) |
| token 预算、running 容量耗尽 | 停止；未准入窗口按原序放在原尾部前，再做 skipped 回填 | `while` 的 budget 条件和容量 break；token 数受预算截断。[scheduler 364](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L364)、[445](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L445) |
| KV 分配失败 | 停止，不跳过；失败对象和未尝试候选一起按原序恢复 | `allocate_slots` 返回 None 即 break，失败对象当时仍在队首。[scheduler 464](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L464) |
| 成功准入 | 成功对象不恢复；仍须原生维护所有状态和资源 | 原生 `popleft`、`running.append`、事件/new/resume、块、预算、状态和 encoder 分配；后续 schedule 内推进 computed tokens，输出处理可释放资源。[scheduler 487](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L487)、[628](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L628)、[751](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L751) |
| 窗口全部成功消费，仍有资源 | 不预测第二次，继续原生处理尾部；所有 skipped 最终在原循环末回填 | 原生可一轮循环多个请求；skipped 用 `appendleft` 收集，再 `waiting.extendleft(skipped)` 双重反转，保持被跳过对象原相对顺序并置于剩余 waiting 前。[scheduler 341](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L341)、[537](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L537) |
| 新增 / 取消 | 仅在当前同步 schedule 完成、窗口恢复后处理；不能让被摘出的请求接受独立取消操作 | 常规 EngineCoreProc 的 IO 线程只把 ADD/ABORT 放 input_queue；busy loop 在 step 之前处理，schedule 由同一 core 调用线程同步执行。新请求 append 尾，取消从 running 或 waiting 移除并释放。[core 536](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/engine/core.py#L536)、[629](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/engine/core.py#L629)、[scheduler 922](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L922) |

上述串行结论仅针对所读常规 CoreProc 调用流程；输入 socket 可以在 schedule 期间接收并入 input_queue，但不会由该 IO 线程直接更改 waiting。Inproc client 直接调用 Core，源码没有在此建立跨线程互斥的通用保证；batch queue、DP、外部重入、特殊实现均不因此获得证明。正式实现须限制受支持调用方式并验证无重入；若可能并发，不可用本模型假定安全。

## 已有 shadow 的差异和机会计数

当前 [scheduler 355](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/core/sched/scheduler.py#L355) 在上述原生资源条件成立后调用 `sweep`。现有 [shadow 435](https://github.com/scaramouch15/vllm091/blob/970cd83491feb2e80c4d150a96ae4b84cbd76558/vllm/v1/metrics/r1_shadow.py#L435) 入口递增 batch、读取状态后才检查 `(batch-1) % every`；EVERY=8 时第 1、9、17 次机会才尝试预测，不是每 8 次所有 schedule、固定时间或每次成功预测。无效 KV 状态在间隔判定前可导致失败。它扫描最多 128，跳过无效项继续找最多 32 个候选，并非连续前缀。

现有 `sweep` 成功返回 `(snapshot, 原顺序 ID tuple)` 给 `admit` 用于判断本批是否预测，排序 ID 是 `shadow_sort` 的异步日志字段，不在返回值中、可丢日志，且把队首也纳入排序。因此既不能依赖日志进行实时决策，也不能直接把现有结果当本设计的有序窗口。未来最小改造应单独定义同步的、当轮身份绑定的前缀预测结果/失败状态及机会计数；是否与 shadow 共用一次推断取决于让候选集合完全一致的明确设计，不能偷偷第二次预测、把旧 shadow 子集当连续前缀，或恢复旧值。无预测/无机会时按原生顺序处理，不是沿用上轮优先级。

### 首版机会计数验证约定

以下是拟验证的约定，不是已接入生产的行为，也不证明 EVERY=8 适用于所有负载：

- 策略开关开启、配置受支持，且本轮无抢占、waiting 非空、`token_budget > 0`、running 尚有容量，计为一次策略机会。running 阶段结束后，在前缀提取和预测之前递增机会计数；即使队首不合格也计入。
- 机会计数从零初始化；递增后以一为首个机会。EVERY=8 时，第 1、9、17……次机会进入更新流程；其他机会间隔跳过，不提取预测候选，直接走原生顺序。
- 进入更新流程后没有有效前缀则不预测；无效分数、预测失败或空前缀均不回退机会计数。成功预测次数另行记录，不作为机会计数。
- 新 Scheduler 实例启动时初始化自己的机会计数；首版不支持运行中热修改策略配置。每次 `schedule()` 的决策身份与机会计数分别定义：同一实例每次 schedule 即使没有策略机会也有独立决策身份；只有符合上述条件才增加机会计数。两者均不等于成功预测次数，也不与现有 shadow 的 batch 计数混用。

8/128/32 是旧影子验证参数，不证明适用于此策略。候选数、P 和频率不可简单按 request-rate、客户端 max-concurrency 或 max-num-seqs 比例设置，需基于服务端真实队列及状态、成本和公平性评估。独立模型的 `opportunity` 是调用方给出的单轮输入，并不实现或验证上述跨轮机会计数。

## 公平性和反例

在有限队列、没有新请求插队、队首每次可获资源且每轮至少有一个成功准入、没有特殊 skip 干扰等条件下，队首保护使旧队首不会被后续低分者反复超越；成功出队后，窗口之外对象可逐步向前，直到自身成为受保护队首。没有这些条件就没有等待时间上界或无饥饿证明。

- 每轮仅一个空位：如果队首可准入，每轮只准入队首；其余排序不改变当轮准入，后继逐步前移，但无性能收益证明。
- 队首长期 KV 失败：每轮首个尝试即 break，后继无法前移；保护队首不解决永久资源阻塞。
- `[A,B,特殊,C,D]`：候选仅 `[A,B]`，不会跨特殊请求预测 C/D。若 A/B 消费完且特殊请求未就绪，原生可能暂时跳过它并继续 C/D，最后将特殊请求回填；这属于原生语义，不等于策略跨障碍构建候选。特殊在队首时则整轮原生，后续请求仍可能因原生 skip 获处理。

## 独立模型、测试和生产未覆盖项

`model.py` 不导入 vLLM，采用不可变模拟请求、人工分数、抽象 token/capacity、一次性 round_id 和同步轮次。模型实际只校验本轮 `round_id` 与有序候选 ID；它没有 Scheduler 实例或生命周期身份。未来生产预测接口还须绑定唯一 Scheduler 实例/生命周期身份及当次 schedule 决策身份，防止重启或不同实例复用相同轮次号；身份的生成与传递方式待生产设计。模型的队首保护、连续前缀、失败停止、未成功原序恢复与双反转 skipped 回填是设计及源码关系的抽象；`eligible`、固定 skip、KV 成败、每请求 token 数、资源和回退是简化假设。模型可以复制完整模拟列表做清晰的状态验证，不能拿它的成本代表实现成本。取消/新增仅在轮次之间进行；不模拟真实缓存、状态转换、资源计数、prefix cache、异步 connector、并发和异常事务，也不证明生产并发安全。

从仓库根目录运行：`python -B tools/r1_waiting_admission_design/test_model.py -v`。测试覆盖空、单、双和大窗口；同分稳定、队首保护、全部/部分/零成功；KV 首次与中途失败、token 与容量耗尽、特殊截断及队首回退；机会跳过、预测失败/非法/过期回退；多轮恢复、尾部新增与身份守恒；以及原生式 skip 回填的简化边界。它不证明生产 skipped、KV、connector 或并发正确。

## 未来生产计时方案

每项使用单调墙钟、记录实际候选与检查数和 outcome：①触发判定及前缀提取；②状态快照/特征组装与动态批量预测；③分数校验、稳定排序和决策构造；④队列应用；⑤原生准入期间的窗口恢复；⑥日志字典、入队及计时记账；⑦完整 `schedule()` 入口到返回的墙钟。①–⑥可作为明确不重叠的子阶段，若记录一个端到端策略路径，它包含相关子阶段和控制流间隙；`schedule()` 包含整个策略路径与原生工作，故不能再相加。恢复与原生循环若交错，应定义多个累加的互斥子区间，避免把整个 waiting 循环标成恢复开销。日志后台序列化/写入/flush 与前台入队分开，后台 CPU 争用仍可影响墙钟；计时机制自身成本也应单列。现有 shadow 字段是嵌套口径，不静默复用为新字段；直方图结果是桶边界而非精确样本分位数。此前独立微基准的微秒数不代表生产真实开销。

正式实现前还须确定实例/生命周期及唯一决策 ID 的生成与传递、快照有效期、配置与请求回退、与现有 shadow 的单次预测共用方式、队列恢复与所有原生退出的测试、特殊对象及 skipped 的集成、时钟/日志版本、外部调用并发约束、窗口外公平性和跨轮失效。无自适应、老化、SLO、跨轮缓存或真实调度收益结论。
