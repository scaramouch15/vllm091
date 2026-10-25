# 研究内容一：服务依赖兼容审查与在线旁路接入准备

整理日期：2026-10-25。本文是可审查方案，尚未实施预测 Hook 或队列重排。服务器操作由用户执行；本轮没有连接服务器、启动服务或 GPU 作业。

## 1. 当前基线与完成状态

**已核实（源码基线）**：原始审查依据为 `research-v0.9.1` 分支的 `09c35e97b0605bd437c725562ca05afaacdb82e1`。原始审查时没有 tracked diff。本次公开源码审查分支从该提交创建，只新增 `tools/r1_service_compat_20261024/` 中的工具及文档；运行源码保持基线内容。原工作区已有未跟踪文件不纳入本次提交。

本报告与工具由原仓库外目录迁入独立工具目录，补齐了相对路径适配和按已提交 Git blobs 构建的 `build_package.py`。下载包的 `SOURCE_COMMIT.json` 标识工具源码 commit，`source_baseline.json` 标识服务运行基线，两者不应混淆。具体生成和使用方式见 [README](README.md)。

**已核实（前轮回传）**：B 环境及模型在独立预测器环境中已完成训练、质量检查和 A800 CPU 推理验证；AB 与 BA 四次模型验证均复现 1440 条保存预测。该证据尚未覆盖真实 vLLM 服务环境。不能把 B 的独立 sklearn 环境称为完整共存验证。

**待服务器执行**：历史服务解释器 `$HOME/minitest/envs/vllm091-whb/bin/python` 的实际包库存、pip 安装计划、`vllm.__file__`/版本/源码、独立完整服务副本及 CPU 共存检查。目前不能宣称服务依赖兼容已经通过。

## 2. 依赖安装计划与停止条件

| 对象 | 固定目标 | 原服务当前状态 | 预期最小动作 |
|---|---|---|---|
| Python | 3.10.22 | 待实际解释器核对 | 必须已匹配；不原地升级 |
| NumPy | 2.2.6 | 历史数值库版本，仍需本轮核实 | 若已匹配则保持；否则停止 |
| SciPy | 1.15.3 | 历史数值库版本，仍需本轮核实 | 若已匹配则保持；否则停止 |
| scikit-learn | 1.7.2 | 是否存在、版本与导入位置待查 | 缺失时拟新增；已有不同版本则停止替换 |
| joblib | 1.5.3 | 同上 | 缺失时拟新增；已有不同版本则停止替换 |
| threadpoolctl | 3.6.0 | 同上 | 缺失时拟新增；已有不同版本则停止替换 |
| torch、vLLM、CUDA 包和其他包 | 保持原版本 | 待库存及 resolver 检查 | 计划包含任何新增/替换这些包即停止 |

`predictor_requirements.txt` 固定五个预测依赖，其中 NumPy/SciPy 用于验证约束，不表示强制重装。若三个辅助包均缺失，预计只新增 sklearn、joblib、threadpoolctl；实际清单必须以 `assessment.json` 中每项 before/after/action 为准，不能用此预计替代现场结果。若目标版本均已存在，计划可以为空。

源码中的 `requirements/common.txt` 对 NumPy、SciPy 没有本任务所要求的精确版本锁定；`requirements/test.txt` 的测试环境锁定值也不能替代当前服务环境库存。依赖兼容判断以实际已安装元数据、pip check、解析计划及 CPU 共存检查为准。

### 只读检查工具

`inspect_service.py` 使用绝对服务解释器，在服务仓库作为 cwd 的条件下分别检查：

1. 原始环境（保留原 PYTHONPATH/user site 行为）与隔离环境（禁用 user site，清除 PYTHONPATH/PYTHONHOME）。记录解释器、prefix、sys.path、包版本/安装器/direct_url/Requires-Dist、Conda 包的 channel/url/build/hash、模块 spec 位置、环境变量、pip freeze/config/debug/check。普通 pip 索引安装若未保留 direct_url，实际历史下载索引可能未知，不能猜测来源。
2. 当前 pip 的 install help。支持时执行 `pip install --dry-run --report`，不升级 pip，不实际安装；可能访问现有索引并下载元数据/缓存。URL 凭据及 query 在交付记录中脱敏。
3. 两种环境下独立 CPU 导入身份检查，保存实际 `vllm.__file__`、`torch.__file__` 与可用预测库的导入位置。缺失 sklearn 等属于本轮拟新增项，不伪装成已安装。
4. 保存服务器 Git 状态/HEAD，并把实际导入的 vLLM 源码与本地审查文件的 LF 归一化 SHA256 比较。代码版本、导入来源或相关文件不同则暂停副本安装，先审查差异。

原始 pip check 失败属于**已有问题**；隔离后消失则优先标记用户 site/PYTHONPATH 污染；固定依赖计划要求替换既有包、引入无关依赖或解析失败属于**本轮计划问题**。两者分别保存，不自动修复。当前脚本对已有冲突采取保守 HOLD，不绕过它继续安装。

若 pip 不支持 dry-run/report，保存库存及 Requires-Dist 作为只读元数据证据；可额外读取已有 B 环境的对应元数据。此时不把“元数据看起来可行”当作安装批准，副本安装门禁保持关闭。不使用 `--no-deps` 安装。

HOLD 后的最小处理建议：若仅用户 site/PYTHONPATH 影响来源，先核查原服务实际启动环境，可研究进程级隔离而不是卸载用户包；若既有 sklearn/joblib/threadpoolctl 版本不同，先评估只在新完整副本替换这几个包及其影响，本轮脚本不擅自替换。若 Python/NumPy/SciPy 或 torch/CUDA 依赖必须变更，则保持原环境，考虑经单独审查的完整兼容副本，或将预测放在现成 B 环境的独立 CPU 工作进程；后者会新增 IPC/排队，不能当作当前进程内接入已完成。若无法取得可信解析计划，则先补齐可核验元数据/固定 wheel 证据，再审查安装，不用忽略依赖绕过。

## 3. 独立完整服务副本验证

**待执行命令**见同目录 [服务器执行说明](README.md)。只有检查结果 `clone_install_eligible=true` 才能执行 `clone_validate.py`，且仍不向原服务环境安装包。

- 克隆来源是现场实际服务 `sys.prefix`，不是独立 sklearn 环境。新 prefix 必须位于用户 HOME 内、尚不存在，并与报告目录分离。
- 先记录 `du --apparent-size` 和空闲空间，要求不超过用户输入的副本空间上限，并保留 20%+2 GiB 余量。使用 `conda create --clone ... --copy --offline`；缺少缓存、空间或复制失败时保留日志停止，不改用一个只有 sklearn 的环境冒充共存。
- 保存克隆前后包库存。仅在副本内下载固定五个包的 wheel，核对 Name/Version/SHA256，使用本地 wheelhouse 做完整依赖解析。副本计划必须与已审查的纯新增计划相同，才允许副本 `pip install --report`；不忽略依赖。
- 副本安装命令使用副本的绝对 Python、显式 prefix、`--no-user`，并在该子进程禁用隐含的 pip 配置目标，避免全局 `--target/--user` 配置写到其他位置。下载/解析失败不切换版本或修改原环境。
- 安装后 pip check、包库存比较，确认所有原有包版本均未改变。两个新进程分别测试 torch/vLLM→sklearn 与 sklearn→torch/vLLM 导入顺序；第三个进程在 vLLM/torch 已导入的情况下加载 B 模型，复现 1440 条长度/时长 raw/processed 预测，rtol=atol=1e-10。
- 复用迁移包现有 `common.py`、`train_cpu.stages/comparison` 和原辅助函数，不运行训练或原计时 main。核对模型、数据、源码、预测身份和顺序；失败保留具体误差，不放宽容差。
- 记录实际包路径、BLAS/OpenMP 和 torch CPU 线程设置。除可能为 editable checkout 的 vLLM 外，实际导入包必须位于副本 prefix；vLLM 的实际源路径和文件哈希另行记录，不能把原源码共享误称为源码也已独立复制。
- 不实例化 LLM/AsyncLLM/EngineCore，不加载 LLM 权重。一次性 CPU 检查子进程启用 NVML availability 检查，并阻止 torch 的 lazy CUDA 初始化及 CUDA C 后端初始化/device-count fallback，检查 CUDA 未初始化；NVML 等平台探测可发生。若导入需要被阻止的调用，则保存错误并停止，不能解除保护绕过。保护仅存在于检查子进程；检查通过仍不能证明 GPU 模型执行兼容。

此检查只能证明“完整服务依赖副本中的 CPU 导入和预测链路共存”。ABI、GPU 执行与真实服务并发仍留待以后授权验证。脚本运行结束后还会检查原服务包版本库存未变；源码文件只读。

## 4. 真实调用链与前端文本缺口

以下均为当前本地源码判断，服务器源码须先通过哈希核对：

| 阶段 | 当前函数/位置 | 对本方案的意义 |
|---|---|---|
| API Completion | `serving_completion.py:create_completion`，69 行；预处理约 117 行，采样参数约 158–167 行，generate 约 193 行 | 同时拥有原请求类型、规范化文本、最终前端 token IDs 和解析后的生成上限 |
| 文本处理 | `serving_engine.py:_normalize_prompt_text_to_input`，466 行；`_preprocess_completion`，707 行 | 可保留完整文本；engine_prompts_text 约 734 行只保留 token IDs |
| AsyncLLM | `async_llm.py:generate`，299 行→`add_request`，232 行→`_add_request`，279 行 | Processor 后 n=1 直接送 Core；n>1 fan-out 复制 EngineCoreRequest |
| Processor | `processor.py:process_inputs`，204 行；270 行补有效 max_tokens；327 行构造 EngineCoreRequest | 本轮数字预测字段应显式放进构造器，并核对最终 token 数与前端一致 |
| IPC | `core_client.py:AsyncMPClient._send_input`，793 行；`add_request_async`，845 行 | MsgpackEncoder.encode 之后经 ZMQ 发送 |
| Core 接收 | `core.py:EngineCoreProc.process_input_sockets`，626 行；约 676 行 decode；`EngineCore.add_request`，175 行 | 解码后还要经过 input_queue，再转换 Request |
| Request | `request.py:from_engine_core_request`，87 行；构造器约 54 行 | 复制最少数字字段；原生 num_prompt_tokens=len(prompt_token_ids) 已存在 |
| 调度 | `scheduler.py:add_request`，903 行→`schedule`，179 行 | 沿用原 deque、KV 分配和调度输出 |

现有 Completion 预处理明确把 `request_prompts` 与 `engine_prompts` 分开：后者文本分支只携带 token IDs。因此，不能假定 AsyncLLM 的 `prompt_str` 有原文本，也不能在 Core 对 token IDs 解码来替代训练文本。

身份链在首轮范围内为 `X-Request-Id: attempt`→`cmpl-{attempt}`→单 prompt 的 `cmpl-{attempt}-0`→同名 EngineCoreRequest/Request/Scheduler ID。多个 prompt 在 API 生成 `-i`；n>1 在 `parallel_sampling.py:get_child_info`（77 行）生成 `{child_index}_{parent_id}`。旁路沿用这些身份，不产生新 HTTP 协议或类型标签。

## 5. A：静态特征与最少字段（待实施建议）

### 范围与输入一致性

首轮仅参与：V1、已验证的 Meta-Llama-3-8B-Instruct 模型/Tokenizer 配置、`/v1/completions`、单个字符串 prompt、add_special_tokens=False、n=1、非 beam、有效 max_tokens=2048。LoRA/adapter、多模态、prompt embeds、token-ID 输入、多个 prompt、结构化输出、remote KV 或其他未经验证模式保留原生处理，明确计为未参与预测，不新增请求拒绝。

模型的文本输入是已应用聊天模板后的最终 prompt，包含模板标记及生成提示。当前 mixed 在 `mixed_workload_dataset.py` 约 354 行 apply_chat_template 一次，378 行按 add_special_tokens=False 计数；`benchmark_serving.py` 约 813 行为 mixed 设置 HTTP body 的 add_special_tokens=False。Completion 服务不再次套聊天模板。

建议在 `create_completion` 原生预处理和 `to_sampling_params` 成功后、创建 generate 之前，直接使用 `request_prompts[i]['prompt']` 与 `engine_prompt['prompt_token_ids']`。TF-IDF 直接调用训练 vectorizer.transform：不额外 strip、重排空白、解码、裁剪或重新实现分词。原生 truncate_prompt_tokens 会截 token IDs，却保留原文本，首轮凡开启该参数（含负数）均跳过；模型配置 do_lower_case 或其他输入改写未经核对也跳过。Processor 最终生成上限和 prompt token 数不符时丢弃数字预测、计 mismatch。

只看“文本请求”不足以证明模板/Tokenizer 来源匹配。建议首轮复用已知 mixed 导出清单中的 **prompt_hash allowlist**、对应 prompt_len/max_tokens 和配置/模型记录，前端初始化时仅读这些核验字段，不读 task_type 作为特征；未命中或 token 数不符明确记 unverified_prompt_provenance/mismatch。它只是旁路范围校验，不替代原始数据源，不要求新的 YAML/JSON 实验协议，也不向 Core 发送文本。若历史 Tokenizer 文件哈希没有记录，须明示该溯源缺口，通过已验证文本/长度逐条匹配限定首轮范围，不能声称核对了不存在的历史哈希。该设置仍待审查，不能用几个模板关键词假装可靠匹配。

### 字段与桥接

1. 前端的 encoder+三个长度头只计算一次。使用原 `num_prompt_tokens,text_0..31`、原后处理，得到有限、非负、已排序并截在 2048 的三个数字。
2. 为减少公共 EngineClient/V0 接口改动，建议以 **仅本进程使用** 的 `SamplingParams.extra_args` 保留一个内部数字 context：q50/q90/q95、前端 token 数/有效上限及状态码。原公开 Completion.to_sampling_params 仅在 extra_args 放 kv_transfer_params，服务自己写入保留键，拒绝把 HTTP 用户值视为可信预测。不能覆盖现有 kv_transfer_params。
3. Processor 在 `params.clone()` 后读出并移除该内部键，核对最终 token 数和 max_tokens，再显式构造新增 `EngineCoreRequest.r1_length_quantiles: Optional[tuple[float,float,float]]=None`、`r1_shadow_status: int=0`。前端 token 数/上限只用于本地一致性核对，Core 已有原生字段，不重复传输；sampling_params 中不残留另一份 context。
4. `Request.__init__/from_engine_core_request` 显式复制两个字段；继续复用 `num_prompt_tokens` 与 `max_tokens`。状态 0=未参与/未提供，1=有效，其他数值用于 unsupported/static_error/mismatch 等受控类别；缺失保持 None，不能补零伪装预测。
5. 不新增完整文本、TF-IDF、32 维 SVD、task_type 或模型对象到 IPC。n>1 在 **API fan-out 之前** 标记整个请求未参与；不能因子请求 n 已变为 1 而重新判为适用。现有 copy 可安全复制不可变 tuple，但首轮所有这些子请求只保留缺省预测和未参与状态。
6. 增加 Core 内的 `r1_first_admitted=False`，仅旁路启用时维护：在实际 native allocate_slots 成功且真正进入 running 的位置置 True；抢占后保持 True，恢复不重算为首次。不要用 num_computed_tokens==0。未来特性可考虑消除额外 bool 的替代，但首轮应优先显式正确性。

msgspec 的 EngineCoreRequest 为 array_like、omit_defaults。新增字段追加在现有缺省字段末尾；不能未经实测便声称关闭时编码字节完全不变或旧/新进程可以混搭。两端使用同一代码版本，必须验证真实编码、默认值和完整跨进程往返。

数字字段随原生请求对象存活，不另建长久 ID→文本缓存。取消时沿原 abort/finish 路径，Request 在 `_free_request` 清理附属状态；请求结束后释放，无跨请求共享可变数组。前端临时 context 在 Processor 提取后释放，验证异常时记录状态而不更改采样参数。

## 6. B：状态读取及原生准入语义（源码判断）

当前 `scheduler.py` 约 338–348 行，位于 running 调度处理之后、waiting 接纳之前。现有采集器就在这里生成共享首次准入快照，条件为无本轮抢占、waiting 非空、token_budget>0、running 有名额。

`len(self.running)` 是 O(1)；`kv_cache_manager.usage`（113 行）→`block_pool.get_usage`（331 行）→free_block_queue.num_free_blocks，也是直接标量读取，不清空状态，不需要 prefix 命中查询或 make_stats。KV usage 以全部 block 数为分母，包含原生保留/null block 等语义，不是逐请求独占显存比例。

running 阶段可能已对本轮工作分配 KV 或发生释放，但其模型工作尚未执行；num_computed_tokens 的本轮递增在 schedule 尾部约 613 行。因此此处状态包含本轮 running 阶段已做的资源决策，不等于 GPU 工作已完成。只需 running_count、kv_usage，不调用现有 `CoreCollector.snapshot` 的 running 聚合循环，不要求开启完整采集器。

此时尚不保证任一 waiting 候选能通过 KV 分配。原生 remote KV 更新、FSM readiness、LoRA 名额、缓存命中/remote KV 查询、token budget、encoder 输入预算及 allocate_slots（约 446 行）仍在之后发生。旁路不提前调用它们，也不以预测替代它们。原生不允许本轮尝试 waiting（如已抢占）时跳过旁路。

旁路取得的是“在当前准入机会下”的条件性时长估计；候选没有真实准入时不能把预测当作已获得准入的完成标签。首次准入模型不直接用于 PREEMPTED/恢复请求。实际 successful admission 后即使再次等待，也由显式 first_admitted 标记排除。

## 7. C：首轮有界旁路测量设置（待审查，不是调度策略）

| 设置 | 建议首轮值/行为 | 有界性和解释 |
|---|---|---|
| 开关 | 新旁路开关默认关闭；独立于 VLLM_R1_COLLECT | 关闭时不导入 sklearn、不加载模型、不建立预测统计容器、不计算/排序 |
| 请求范围 | 已知 mixed prompt_hash allowlist，最多 128 个符合范围的 engine 请求 | 可涵盖 120 main+probe；超过范围计 measurement_limit，不扩大实验 |
| 触发 | 上述 native waiting 尝试条件成立，且每 10 个这样的机会采样一次 | 计数而不是全局 RNG；最多 100 次旁路 sweep/服务生命周期 |
| 候选 | 按原 waiting 顺序，用 islice 只看前 128 个，选前 32 个支持且未首次准入者 | 扫描与矩阵均有上限；不遍历整个 waiting，也不扫描 running |
| 状态条件 | 首轮只选 WAITING、有效数字字段、first_admitted=False；FSM/remote/preempted 等跳过 | 不调用会推进这些状态的 native 方法 |
| 无候选 | 只更新触发/跳过计数，不做状态聚合、特征矩阵、predict 或 sort | 防止用无效计算稀释测量 |
| 时长预测 | 原 E 字段顺序及后处理，独立 float 数组，稳定 P90 升序排序原候选索引 | 临时列表及排名只用于检查，不对 waiting/running 调用 sort/popleft/append |
| 记录 | 默认只汇总，至多保存少量有界诊断样本 | 无每轮 print、无同步逐候选磁盘写入 |

模型初始化：建议 AsyncLLM 初始化阶段按开关惰性导入并加载一次 B bundle，保留 vectorizer/SVD/三个长度头供前端使用；Scheduler 初始化阶段加载一次 bundle，保留三个时长头供 Core 使用。可先在各进程加载同一小 bundle 后丢弃不用的部分，不为首版另造模型拆分格式。加载失败应在启动期显式报配置错误，而不是运行时默默全跳过。

### 线程方案

GBRT 没有本任务需要开启的并行预测选项，首先保留串行调用、每个进程不并发执行预测。TF-IDF/SVD 的内部线程应记录实际池配置，不能仅用“串行 Python 调用”推断 BLAS 一定单线程。

`threadpool_limits` 和 OPENBLAS/MKL/OMP 配置作用于相应进程内的库，不是单个预测函数的局部 CPU 配额。不能在每次预测前后切换全进程设置，也不能把 CPU12 的 taskset 套到整个服务。独立副本 CPU 验证中用 1 线程仅是该一次性进程的检查条件。

建议首个旁路原型不擅自改变服务进程已有线程策略，记录其实际配置及前端同步 CPU 调用的事件循环占用。若后续要求强制“仅预测计算”单线程，则需单独的预测工作进程，并重测其 IPC/排队成本；或经审查在进程初始化时固定数值库线程并明确影响同进程 torch/其他计算。此研究取舍尚未实施，不把独立微基准的线程结论直接冒充服务内保证。

## 8. 预计最小修改文件及生命周期（本轮未修改）

| 文件/函数 | 必要修改 |
|---|---|
| `serving_completion.py:create_completion` | 范围判定、准确文本与 token 数、一次长度计算，内部纯数字 context |
| `async_llm.py:__init__/shutdown` | 前端模型一次加载、汇总资源的明确关闭；保留原 add_request/generate/fan-out 逻辑 |
| `processor.py:process_inputs` | 提取/移除数字 context，核对最终输入，显式构造字段 |
| `v1/engine/__init__.py:EngineCoreRequest` | 追加两个有缺省值的数字字段 |
| `v1/request.py:__init__/from_engine_core_request` | 复制字段、首次准入标记；随请求释放 |
| `core/sched/scheduler.py:__init__/schedule/_free_request/shutdown` | Core 模型一次加载、有界独立候选、实际首次准入标记与释放、旁路汇总关闭 |
| 建议新 `v1/.../shadow_predictor.py`（确切目录实施前确定） | 惰性 sklearn 加载、复用训练推理、范围/数字验证、计时与有界汇总；不新增服务入口 |
| 必要的轻量测试 | IPC、范围、复制/取消、只读队列、关闭分支和失败计数 |

开销测量阶段还应列入 `core_client.py:AsyncMPClient._send_input` 和 `core.py:EngineCoreProc.process_input_sockets` 两处诊断计时：分别包围已有 encode/decode，记录耗时和字节数，关闭时不计时。它们不改写发送/接收流程。`serving_engine.py`、`stats.py` 和既有 request_collector 先作为语义审查/复用对象，不改写原生公式。

关闭链：AsyncLLM.shutdown（214 行）明确释放前端旁路汇总；Core.shutdown（286 行）最终调用 Scheduler.shutdown（1004 行）关闭 Core 旁路资源。不能只依赖 __del__。如需要后台写出，可复用既有标准库 RecordWriter 的有界非阻塞机制，使用独立角色/文件及旁路配置，不实例化完整 CoreCollector、不运行其 snapshot。正常关闭有 footer；丢失、错误、队列满与未确认关闭可见。

配置只在进程初始化时读取，不支持对存量请求热切换开关。后续原生时延对照保持 log_stats 开启；这不等于要求开启完整 request_collector。原生统计未取得时应标记 unavailable，不能补零或改用客户端 latency 充当 inference_time。

## 9. 计时边界及原生统计的影响

| 名称 | 建议边界 | 必须说明 |
|---|---|---|
| frontend_static | 范围/哈希检查、TF-IDF、SVD、长度头和后处理分别计时，另测整体 | 整体单独计时，不相加分项分位数；同步调用会占用前端执行时间 |
| field_prepare | 校验 context、构造 tuple、复制到 ECR | 与文本编码分别记录 |
| encode/decode | 在原 MsgpackEncoder.encode、decoder.decode 调用内同进程计时；记录字节数 | 新字段的增量可在相同输入的离线配对编码测试估计 |
| transport_or_enqueue | await send、Core 输入队列停留等分别命名 | 发送到接收的总间隔还含 ZMQ、排队与进程调度；不能称为纯序列化成本，也不未经验证跨进程时钟直接相减 |
| core_front | 有界候选收集、状态标量读取、矩阵组装 | 只读；自身整体成本另测 |
| time_predict/sort | 时长头、后处理、稳定排序分别计时，再单独测组合整体 | 不将排序纳入 native waiting 决策 |
| reporting | 汇总更新/构造/入队，与后台序列化/写出分别记录 | 不把后台时间简单加到前台时间当端到端时延 |

`scheduled_timestamp` 在 `schedule` 214 行、running 处理之前取得；约 484 行的 SCHEDULED 事件使用这个较早时间。新增旁路在该时间之后、native waiting 准入之前，因此对本轮实际首次准入者，旁路时间可能直接进入原生 inference_time/prefill 区间。对已有 running 请求，还可能延后本轮执行和后续 token 批次时间。

此版本 EngineCoreEventType 只有 QUEUED/SCHEDULED/PREEMPTED，没有单独的 NEW_TOKEN 枚举。NEW_TOKEN 是原统计代码对批次新 token 时间的称呼：Scheduler.update_from_output 约 875 行构造 EngineCoreOutputs，`__post_init__`（158 行）在 timestamp 缺省时记录 monotonic；前端 `IterationStats.update_from_output`（101 行）把该批次时间赋给 first_token_ts/last_token_ts。`update_from_finished_request` 的 inference_time=last_token_ts-scheduled_ts（168 行），不是精确 GPU 时间。

前端静态预测若放在 create_completion、Processor 默认 arrival_time 赋值（约 235 行）之前，其成本可能**不进入**以该 arrival_time 为起点的引擎统计，但会进入客户端观察的请求总耗时。queued_time 从 Core QUEUED 开始，也不覆盖所有前端工作。必须同时报告自有前端计时和原生客户端/引擎指标，不移动原时间戳来隐藏开销。

旁路开关只保证不主动改变准入规则或队列内容。CPU 占用、事件循环阻塞和调度发生的墙钟时刻可能变化，因而不能承诺真实批次轨迹或端到端延迟完全相同。

## 10. 覆盖率、异常与验证清单

建议按进程保存计数：API/engine 请求总数、范围内请求、static_attempt/success/error、数字字段接收/校验失败、native waiting 机会/采样/限流次数、扫描数、支持的未首次准入候选机会数、time_attempt/success/error、empty/unsupported/preempted/missing/limit 跳过数、唯一请求是否至少参与一次预测，以及写出丢失数。

分母必须明确：静态成功率针对已尝试范围内请求；候选覆盖针对被有界扫描看到的合格机会，不能冒称整个 waiting 队列覆盖率。唯一请求参与可由活动 Request 的旁路 bool 跟踪，在释放时汇总，不保留无界全局 ID 集合。初始化失败、运行时异常和 cap 用尽分别计数；不能在失败后继续报“开销很低”而隐藏低覆盖率。

只保留常数大小计数、固定桶直方图或有界 reservoir，并报告采样总量/丢弃量。默认不每轮输出。若使用采样，应使用独立局部 RNG；本方案频率计数本身不动原请求到达 RNG。

后续必要轻量测试（本轮是计划，尚未实施 Hook 测试）：

1. ECR 的 None/三数字/非法数值/default 字段经真实 Msgpack 和独立进程往返，request_id 与 num_prompt_tokens 对应；检查缺省编码字节及同版本 schema。
2. 同一固定调度状态、同样 KV/FSM/LoRA/connector stub 和固定测试时钟：旁路关闭/开启后 waiting/running 顺序、原生对象字段/状态、KV 调用次数、token budget、SchedulerOutput 完全一致；新增旁路字段和统计另行核验。
3. 首次分配失败不置 admitted；实际首次准入置 True；抢占、remote 状态转换和恢复不清零；已结束 Request 释放，无引用泄漏。
4. 单 prompt/n=1 支持；多 prompt、n>1、token IDs、embeds、截断、max_tokens 不符、模型/模板 allowlist 不符均走原生并计跳过；n>1 不因子 params.n=1 而误判。
5. 取消发生在静态前、静态后未送 Core、Core waiting、首次准入后等位置，以及模型推理异常/字段校验失败；原 abort 与错误传播保留，旁路数据不误作完成标签。
6. 关闭时通过 import spy、文件打开 spy、模型工厂 spy 检查不导入 sklearn、不加载权重、不创建预测容器/计算；执行原 mixed/collector 回归。
7. 有界性：全不支持的长 waiting 也最多扫描 128 个；最多 32 候选；无候选不调用 predictor；达到窗口/频率上限时覆盖率和 skip_limit 可见。
8. 未来另行授权真实服务后，以同节点同服务配置比较预测关/开吞吐、原生 TTFT/TPOT/ITL/E2E 等，保留两边请求映射、失败/取消/预测覆盖与日志丢失情况；先做既有 120 请求验收，不据此推出调度收益或预测精度泛化。

## 11. 尚未解决的问题与后续顺序

- 真实服务库存和安装计划尚未取得；当前只有明确的拟安装目标，不能断言不影响现有依赖。先运行只读检查，返回 assessment 和日志。
- 完整服务副本的复制成本、空间、pip/Conda 缓存和服务器源码一致性待现场检查。门禁 HOLD 时保留证据，由下一步审查决定最小处理，不自动替换包。
- 进程内预测的线程限制对 torch 等同进程库的影响，需要根据服务实际池配置选择初始化策略或专用预测进程；本轮不擅自改变服务线程配置。
- allowlist、128 请求窗口、32 候选/128 扫描、每 10 机会、最多 100 sweep 是首轮可测设置建议，尚未实施，不是最终调度公式。
- B 在 AB/BA CPU 微基准中能复现质量，但静态路径相对性能出现顺序/时段相关变化；不能将某次快值当成线上预算保证。
- 真实 GPU/服务并发、IPC 实际增量和墙钟干扰仍待未来授权；本轮不实施 Hook、Scheduler 预测、waiting 重排、训练或负载测试。

交付顺序：本地报告与工具已准备 → 用户执行服务只读检查 → 核查返回的计划与差异 → 条件允许时执行独立完整副本 CPU 共存验证 → 再审查最小接入方案。任何一步都不要求向原服务环境安装包。

## 12. 本地准备验证记录

本次工具的 9 项标准库测试覆盖纯新增/已满足计划、禁止重装既有包、禁止 torch/CUDA/无关包变更、已有冲突及缺报告、进程级隔离与 URL 脱敏、源码来源比对、Python 3.10 语法和假模块 CUDA 初始化保护。迁入后验证 6 个 CLI help、README 四段及 TESTING 一段 Bash 命令语法；完整检查方法见 TESTING.md。测试输出保存在仓库外，不纳入源码提交。

测试使用现有解释器，不安装新依赖；没有在本地导入真实 vLLM/torch/sklearn 或执行服务克隆。迁入后增加仓库根目录自动定位及 `R1_COMPAT_TEST_REPO`，使同一测试可在仓库内和解压工具包中核对真实基线。生成的包须逐文件与对应 commit 的 Git blobs 比较；报告与服务器说明不将未执行项目标记为成功。
