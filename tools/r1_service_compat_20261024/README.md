# 服务依赖兼容检查：服务器逐步命令

准备日期：2026-10-25。先执行第 1 步只读检查，再返回结果审查；第 2 步是条件允许时的完整服务副本验证，不向原服务环境安装包。本工具不连接服务器，不启动 LLM 服务、训练、benchmark 或 GPU 作业。

源码旁路方案见 [审查报告](review_report_20261025.md)。包内没有预测 Hook 或 waiting 重排实现。

## 0. 从审查源码生成工具包

本目录是可提交的源码。环境、模型、验证数据和检查输出不纳入版本控制。服务运行基线仍为 `09c35e97b0605bd437c725562ca05afaacdb82e1`；工具审查 commit 与该运行基线是两个不同标识，不要为了运行工具切换服务源码分支。

在包含本目录的已提交仓库中，用现有 Python 3.10+ 生成包：

```bash
python -B tools/r1_service_compat_20261024/build_package.py \
  --revision HEAD \
  --out-dir /ABS/NEW/DIRECTORY/OUTSIDE/THE/REPOSITORY
```

将输出路径替换为仓库外的新目录，父目录必须存在。Windows 也可直接传 Windows 绝对路径。构建程序按显式文件清单读取指定 commit 的 Git blobs，忽略未提交工作文件；只生成工具源码及文档、`files_sha256.json` 和 `SOURCE_COMMIT.json`。后者记录完整源码 commit、仓库地址与运行基线。输出目录存在则报错，不覆盖旧包。

输出为 `r1_service_compat_20261025_tools.tar.gz` 和 `.sha256`。包中根目录保持 `r1_service_compat_20261024/`，因此下列服务器解压命令与已提供的调用参数一致。完整验证方法见 [TESTING.md](TESTING.md)。不要把私下回传的检查包作为源码包上传 GitHub。

## 1. 只读检查原服务环境

把本次 `r1_service_compat_20261025_tools.tar.gz` 与 `.sha256` 私下上传至服务器 `$HOME/whb/`。在可使用的已授权节点终端执行以下整段。此阶段**故意不提前全局 unset PYTHONPATH/PYTHONHOME 或禁用 user site**，以便记录原始状态；工具分别启动原始/隔离子进程作比较。CPU 导入进程有自己的保护和线程设置。

历史服务路径和仓库路径先实际 test；若不存在会停止，不能换成 B 的独立 sklearn 环境来完成检查。

```bash
(
set -euo pipefail
hostname
id
printf 'HOME=%s\n' "$HOME"
taskset -pc $$
cd "$HOME/whb"
tr -d '\r' < r1_service_compat_20261025_tools.tar.gz.sha256 | sha256sum -c -

R1_TOOLPY="$HOME/whb/envs/r1-predictor/bin/python"
R1_SERVICEPY="$HOME/minitest/envs/vllm091-whb/bin/python"
R1_SERVICEREPO="$HOME/whb/vllm091"
test -x "$R1_TOOLPY"
test -x "$R1_SERVICEPY"
git -C "$R1_SERVICEREPO" rev-parse --show-toplevel
ls -l "$R1_SERVICEPY"

R1_WORK="$(mktemp -d "$HOME/whb/r1_service_compat.XXXXXX")"
tar -xzf "$HOME/whb/r1_service_compat_20261025_tools.tar.gz" -C "$R1_WORK"
R1_KIT="$R1_WORK/r1_service_compat_20261024"

PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B - "$R1_KIT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
for name,digest in json.loads((root/'files_sha256.json').read_text()).items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise RuntimeError('Invalid package path: '+name)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Package checksum mismatch: '+name)
print('Tool package verified')
print('Source commit:',json.loads((root/'SOURCE_COMMIT.json').read_text())['source_commit'])
PY

printf 'R1_WORK=%q\nR1_KIT=%q\nR1_TOOLPY=%q\nR1_SERVICEPY=%q\nR1_SERVICEREPO=%q\n' \
  "$R1_WORK" "$R1_KIT" "$R1_TOOLPY" "$R1_SERVICEPY" "$R1_SERVICEREPO" > "$R1_WORK/paths.sh"
printf '保留检查目录：%s\n后续使用：source %q\n' "$R1_WORK" "$R1_WORK/paths.sh"

PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B "$R1_KIT/inspect_service.py" \
  --service-python "$R1_SERVICEPY" \
  --service-repo "$R1_SERVICEREPO" \
  --predictor-python "$HOME/whb/envs/r1-sklearn172/bin/python" \
  --out "$R1_WORK/inspection_001"

PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B - "$R1_WORK/inspection_001/assessment.json" <<'PY'
import json,sys
assessment=json.load(open(sys.argv[1]))
print(json.dumps(assessment,ensure_ascii=False,indent=2))
PY
)
```

结果包含 `native_environment.json`、`isolated_environment.json`、各自 pip check/freeze/config、dry-run/report、实际 import 身份与源码核对。`imported_sources` 使用 schema_version=2：`vllm/*` 核对实际导入的包，`benchmarks/*` 核对显式 service_repo，各自报告 checked/skipped/mismatched；缺根目录、缺文件或不一致不能通过。旧版缺 benchmark 覆盖的 assessment 必须重新执行检查，不能直接用于本版克隆。

安装来源的 URL 凭据/query 脱敏。检查目录是新的，原环境只被读取，没有 pip install 正式执行。`cpu_checks.py` 的关键身份、版本、CUDA、哈希、样本和预测检查使用显式异常，`-O`/PYTHONOPTIMIZE 不会移除这些门禁；失败输出 passed=false，预测失败不输出成功信息。

若 `clone_install_eligible=false`，先打包第 3 步回传，**不执行第 2 步**。若 Python/NumPy/SciPy 不符、要替换既有包、出现 torch/CUDA/其他额外依赖、原环境已有冲突、user site 导入差异、pip 功能不足或源码不同，报告会列出 HOLD 原因。不升级 pip、不自动修复冲突。

## 2. 可选：完整服务副本的 CPU 共存检查

以下命令只供第 1 步评估通过且空间/资源允许时使用；它会创建完整原服务环境的独立副本。可以先把第 1 步结果发回核查，后续再运行，不需要现在连着运行。

先执行第 1 步实际打印的 `source .../paths.sh` 命令，使当前 shell 取得真实 R1_WORK 等变量；然后执行下段。克隆 prefix 与报告分开放，避免回传整个大型服务环境。环境副本保留，不自动删除。

```bash
(
set -euo pipefail
test -n "${R1_WORK:-}"
test -x "$R1_SERVICEPY"
PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B - "$R1_WORK/inspection_001/assessment.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
if not a['clone_install_eligible']:
    raise SystemExit('HOLD: '+repr(a['hold_reasons']))
PY

R1_CONDA="$(conda info --base)/bin/conda"
test -x "$R1_CONDA"
R1_SOURCEPREFIX="$(PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["prefix"])' \
  "$R1_WORK/inspection_001/isolated_environment.json")"
du -sh "$R1_SOURCEPREFIX"
du -s --apparent-size -B1 "$R1_SOURCEPREFIX"
df -h "$HOME/whb/envs"
read -r -p '请输入本次允许副本占用的最大 GiB（正数）：' R1_MAX_GIB
test -n "$R1_MAX_GIB"

# 填写已核实的 A800 迁移包解压目录；不要填写 V100 路径或公开下载地址。
read -r -p '请输入本次 A800 迁移材料的实际绝对目录：' R1_MIGRATION
test -f "$R1_MIGRATION/files_sha256.json"

R1_CLONEPARENT="$(mktemp -d "$HOME/whb/envs/r1_vllm_b_clone.XXXXXX")"
R1_CLONEPREFIX="$R1_CLONEPARENT/env"
printf '原服务：%s\n独立副本：%s\n迁移材料：%s\n' \
  "$R1_SOURCEPREFIX" "$R1_CLONEPREFIX" "$R1_MIGRATION"

PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B "$R1_KIT/clone_validate.py" \
  --inspection "$R1_WORK/inspection_001" \
  --conda "$R1_CONDA" \
  --prefix "$R1_CLONEPREFIX" \
  --out "$R1_WORK/clone_check_001" \
  --migration-dir "$R1_MIGRATION" \
  --max-copy-gib "$R1_MAX_GIB"
)
```

此流程先核对源库存未变化，再以 `--copy --offline --clone` 创建服务环境副本。副本只允许安装已解析的缺失固定依赖；先下载固定 wheel，再从本地 wheelhouse 完整解析，不用 `--no-deps`。所有既有包（包括 torch、vLLM、CUDA 包）必须保持版本不变，超出计划立即停止。副本 pip 配置仅在子进程内禁用隐含目标，并显式指向副本 prefix；不会修改原环境或全局 pip 配置。

克隆完成后、任何 pip 下载/安装之前，`clone_identity.json` 核对 sys.prefix、sys.executable（含解析后的路径）、实际导入 pip 的文件、pip 元数据位置/版本，以及 purelib/platlib/scripts/data/headers 安装 scheme 都指向副本。身份或目标不符即停止后续 pip 操作；不仅比较包版本。安装后的实际模块路径检查继续保留。

两种导入顺序分别在新进程验证，第三个 CPU 进程加载 B 模型对照 1440 条验证预测。没有单 CPU taskset 套用到服务，也没有启动服务进程。CPU 验证子进程固定数值库线程、记录实际线程池，阻止 lazy CUDA 初始化。该结论仍不能替代 GPU/真实服务并发验证。

若独立副本失败，保留 `failure.json`、命令与日志；不要反复在原环境试装包。若 Conda 离线缓存不足或空间超出上限，停止报告，不自动换安装方式或依赖版本。

结果目录建立后，成功或可捕获失败都尝试记录 `source_after.json` 和 `source_after_status.json`。结束核查比较前后包库存、prefix、解释器和所选模块来源，状态为 unchanged/changed/unknown；unknown 时 unchanged 和 original_package_versions_unchanged 为 null，不写 true。原始错误保存在 failure.json，结束核查的第二个错误不会覆盖它；结束核查不成功时 completion.complete=false。unchanged 只表示记录的快照字段一致，不表示整个环境文件系统逐字节未变。

## 旁路验收设计状态

本轮只审查工具与接入方案。未来首轮采用客户端固定验收请求，客户端保存样本与 request_id 关联，离线按 request_id 对照静态长度预测；时长预测还须带对应 sweep/state 数值，在相同状态下重算后比较。服务端只做输入形式、实际 token 数、有效 max_tokens、模型配置等必要检查；不按请求文本哈希做成员筛选，也不为该筛选增加跳过原因或计时。

SamplingParams.extra_args 仍是待实施的前端临时数字传递方案；候选上限与采样频率仅是验收建议。未实现预测 Hook、队列重排或正式在线开销实验，现有 CPU 微基准不替代服务内验收。

## 3. 打包回传（成功/失败均可）

使用第 1 步实际路径文件 source 后执行下段。报告目录不含大型环境 prefix，包会包含工具源码、所有已有检查日志及文件清单；可能包含包下载来源与私有目录信息，请私下传输。

```bash
(
set -euo pipefail
test -n "${R1_WORK:-}"
test -x "$R1_TOOLPY"
R1_DELIVERY="$(mktemp -d "$HOME/whb/r1_service_compat_delivery.XXXXXX")"
PYTHONDONTWRITEBYTECODE=1 "$R1_TOOLPY" -B "$R1_KIT/pack_checks.py" \
  --work "$R1_WORK" \
  --out "$R1_DELIVERY/r1_service_compat_return.tar.gz"
)
```

程序打印压缩包和 `.sha256` 的绝对路径。将这两个文件下载到新的本地目录后发送，保留旧结果，不上传公开 GitHub。即使只完成第 1 步，也应先回传检查包；本轮不要求直接安装到原服务环境。

## 当前交付状态

原始交付完成了本地标准库检查；迁入仓库后按 TESTING.md 重新验证相对路径、CLI 和命令语法。检查输出保存在仓库外，不随源码提交。Bash 只做 bash -n，不执行正文。CUDA 保护测试使用假模块，不是本地实际 torch/vLLM 导入测试。

实际服务依赖、resolver 结果、完整副本与 1440 条共存预测验证均待服务器执行。本地没有安装服务依赖，没有模型训练或下载。不能将本地合成测试或之前只有 sklearn 的 B 环境视为当前共存验证结果。

| 文件 | 职责 |
|---|---|
| inspect_service.py | 只读检查服务原始/隔离环境与 pip 计划，生成 assessment，不安装 |
| environment_probe.py | 标准库记录包/Conda 来源、环境、Git 和模块位置 |
| clone_validate.py | 门禁通过后复制完整服务环境，只在副本中解析/安装/检查 |
| cpu_checks.py | 一次性 CPU 导入顺序与 B 模型验证，无 LLM 引擎 |
| pack_checks.py | 打包工作目录日志/工具，排除外部大型环境 |
| test_checks.py | 本地标准库门禁、语法、来源与 CUDA 保护合成检查 |
| test_regressions.py | 优化模式失败、克隆身份门禁、结束检查及 benchmark 来源回归 |
| source_baseline.json | 23 个实际审查源码文件的哈希、函数行号与 Git 状态 |
| predictor_requirements.txt | 五个固定预测依赖约束，不包含 vLLM 或 torch 安装指令 |
| build_package.py | 从指定已提交源码文件清单生成工具包及源码 commit 标识 |
