# 轻量验证与发布包核对

所有命令使用现有 Python 3.10+，不安装依赖、不启动服务或 GPU 工作。标准库测试中的 CUDA 模块是合成对象，不能替代服务器真实导入顺序验证。

## 仓库内测试

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover \
  -s tools/r1_service_compat_20261024 -p 'test_*.py' -v
```

现有 9 项测试覆盖：新增三个包、目标均已满足、禁止替换/重装既有包、禁止无关包/torch/CUDA 依赖变更、已有冲突或缺少报告、隔离/脱敏、Python 3.10 语法、实际源码哈希比对和假模块 CUDA 初始化保护。源码检查自动向上定位仓库根目录，不导入真实 vLLM。

`test_regressions.py` 新增 11 项针对性测试（合计 20 项方法），其中优化模式回归在子进程分别以 `python -O` 与 `PYTHONOPTIMIZE=1` 执行 10 种失败场景：prefix、executable、CUDA、包位置/版本、Python 版本、哈希、越界清单路径、请求身份和预测不一致。失败必须退出非零、报告 passed=false、不输出成功信息；同时静态检查 cpu_checks.py 不含可移除 assert。

克隆测试全部使用合成快照和命令 mock：5 种身份/安装目标错误均不能到达任何后续 pip 操作；中途失败仍尝试 source_after，结束检查再失败时保留原错并标 unknown/null；已变化不能标 unchanged，前序成功但结束检查失败也不能标整体完成。源码测试把导入的 vLLM 与 service_repo 的 benchmark 分放两个根目录，覆盖未提供 benchmark 根目录及 benchmark 内容不一致。

对 `inspect_service.py`、`environment_probe.py`、`cpu_checks.py`、`clone_validate.py`、`pack_checks.py` 和 `build_package.py` 分别执行 `--help`；这些入口解析参数时不加载模型或执行安装。

README 的 Bash 命令块可保存到仓库外临时 `.sh` 后运行 `bash -n`。这只检查语法，不执行其中的服务器检查、下载或副本创建。另运行 `git diff --check`。

## 从已提交源码构建

按 README 的构建命令生成仓库外的新目录。`build_package.py` 使用显式源码文件清单；不扫描整个工作区，不打入未跟踪文件、日志、环境、模型或验证数据。构建读取指定 commit，而不是工作树；先提交本目录再构建。

核对：

1. 压缩包整体 SHA256 与旁边 `.sha256` 一致。
2. 解压到仓库外新目录，包内 `files_sha256.json` 中每项哈希匹配。
3. `SOURCE_COMMIT.json.source_commit` 等于拟审查的完整 commit。
4. 包内源码/文档字节与 `git show COMMIT:tools/r1_service_compat_20261024/FILE` 一致。只有 `SOURCE_COMMIT.json`、`files_sha256.json` 是构建时生成内容。
5. 设置 `R1_COMPAT_TEST_REPO` 为实际仓库根目录，在解压包里用 unittest discovery 再次运行两个 test_*.py；20 项应全部运行。未设置且包外没有仓库时，实际源码比对测试会明确 skip，不冒充通过。

如果需要保存测试完整日志或审查材料，应放在仓库外。GitHub 上只发布本目录的源码和说明；服务器的环境检查包仍是私下传输材料。

## 结果边界

本地轻量检查不证明实际 vLLM 服务环境依赖兼容。服务器原始/隔离环境、resolver 计划、完整服务副本、两种真实导入顺序及 1440 条验证预测仍须分别执行和审查。源码提交、工具包生成和 GitHub 推送也不代表这些服务器步骤已经完成。
