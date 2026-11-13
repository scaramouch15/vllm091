# JITServe 自然生成单请求验证

本目录的 `jitserve-natural.patch` 基于 UIUC-MLSys/JITServe `ef3ab0c3f3cf42baa6dc3f928bee7675c2efa0c6`，SHA-256 `7d8c1ddaf45f4e4586ff589c06f5d315fc64d716f9506a60b7569383de9cf85e`。这是明确标记的适配版，不是未经修改的作者原版；它没有移植 vLLM 0.9.1 V1 调度器。

## 已验证与未验证

- 本地替身 HTTP 服务运行实际 `benchmarks/benchmark_serving.py`（`--temperature 0.9`）：正常和断流各一轮，每轮 1 probe + 1 正式 Code，ID 可区分；正常轮分母 1、达标 1，断流轮分母 1、成功及达标均为 0。请求为 THROUGHPUT、`[0,0,30]`、`max_tokens=2048`、probe 和正式请求的 temperature 均为 0.9，没有真实目标长度；正常轮最终 token 数采用替身提供的 11。空记录后延迟发非空输出的测试确认首个空记录不触发 TTFT。
- HTTP backend 替身单测 3 项、probe/正式请求 benchmark 主循环单测 1 项和服务端 stop-checker 替身单测 3 项通过。服务端测试覆盖提前 EOS、上限、旧定长模式。这些不验证真实服务或吞吐性能。
- 真实 A800 单请求：**未执行**。正式 120 条对照：**未执行**。首次发送前取消/进程中断不保证登记为正式 attempt；不完整运行应判无效。
- 本地仅测试了非目标测试 tokenizer，两个 `add_special_tokens=False` 输入路径的完整 ID 一致；Meta-Llama-3-8B-Instruct tokenizer、两端版本、完整 token ID、BOS 和模板 token 仍需服务器预检。
- 计时语义存在差异：原 OpenAI completion 客户端在第一个带 choices 的记录计 TTFT（即使文本为空），本适配客户端等待第一个非空可见文本；隐藏 token 不能由此可靠观测。两者都从 POST 前的客户端 `perf_counter` 起算。TPOT 使用现有类型评分公式和服务端最终 token 数。正式对照前须确认此差异可接受。

`real_output_len` 在 stop checker 控制旧定长回放，旧 `_reconstruct_optimal_schedule` 中也被读取；静态搜索未发现普通 JITServe `_schedule` 调用该旧 baseline 分支。普通策略使用在线预测的 `request_info.output_len`，不能据此保证其他分支绝不会读取真实长度。自然模式不接收真实目标长度，用 EOS 或 `max_tokens` 停止；默认行为仍是作者定长模式。

## 应用补丁及依赖

用干净的固定上游提交执行（不要在含工作修改的上游目录应用）：

```bash
export CLIENT=/path/to/vllm091
export JIT=/path/to/JITServe-natural
export MODEL=/path/to/Meta-Llama-3-8B-Instruct
git clone https://github.com/UIUC-MLSys/JITServe.git "$JIT"
cd "$JIT"
git checkout --detach ef3ab0c3f3cf42baa6dc3f928bee7675c2efa0c6
git apply --check "$CLIENT/tools/jitserve_baseline/jitserve-natural.patch"
git apply "$CLIENT/tools/jitserve_baseline/jitserve-natural.patch"
```

已在固定提交干净副本执行 `git apply --check` 和应用。作者 setup 使用 `uv`、Python 3.10 和 commit `32176fee733b76b295346870d717d44cb7102944` 的预构建 wheel，并接入 `third_party/vllm` Python 源；QRF model 和 vectorizer 需由上游资产脚本单独下载。目标机器需验证模型、GPU/CUDA 与此环境兼容，不能将该 wheel 当作 0.9.1 V1。

```bash
cd "$JIT"
bash scripts/setup_vllm.sh
.venv/bin/python assets/download_qrf.py
.venv/bin/python -m jitserve.server --model "$MODEL" \
  --scheduling-policy jitserve --enable-chunked-prefill True \
  --max-num-seqs 8 --port 8000 2>&1 | tee jitserve-single-server.log
```

在另一个终端，`$CLIENT_PY` 应指向独立且具备该 benchmark 依赖的客户端 Python，`$CODE_JSONL` 指向本机已有的合法 CodeFeedback JSONL 样本（保留在本机；不要上传请求内容）。选用 1 条是 smoke 数据，不代替已批准的正式 manifest。服务会先收到 probe，随后收到 1 条正式请求：

```bash
export CLIENT_PY=/path/to/client-env/bin/python
export CODE_JSONL=/path/to/private-code-sample.jsonl
cd "$CLIENT"
"$CLIENT_PY" benchmarks/benchmark_serving.py --backend jitserve \
  --endpoint /generate --host 127.0.0.1 --port 8000 --model "$MODEL" \
  --dataset-name mixed --mixed-code-path "$CODE_JSONL" \
  --mixed-code-count 1 --mixed-chat-count 0 --mixed-summary-count 0 \
  --mixed-output-len 2048 --temperature 0.9 --num-prompts 1 --request-rate 8 --max-concurrency 32 \
  --goodput-by-type chat.ttft:2000 chat.tpot:30 code.e2el:30000 summary.e2el:20000 \
  --goodput-by-type-output /tmp/jitserve-code-score.json 2>&1 | tee /tmp/jitserve-code-client.log
```

使用新输出路径避免覆盖。检查结果中正式 attempt、客户端 ID、引擎 ID、发送/首次可见内容/完成时刻、结束原因、引擎输出数和 Code E2E 达标；用引擎 ID 对照服务端 `natural_request`、`natural_prediction` 日志，核对类型和 30 秒期限。QRF 为异步预测，单条可能尚未更新；在线细化默认 50 token 一次，单条不一定触发细化或抢占。无可见文本的完成会被客户端记为失败。日志和输出可能包含请求信息，保留在私有机器。

## 目标 tokenizer 预检

从实际客户端保存相同的**已渲染** prompt 到私有 UTF-8 文件 `$PROMPT_FILE`，不要二次套模板；分别以基线 vLLM 环境 `$BASELINE_PY` 和补丁 JITServe 环境的 tokenizer 编码，并比较完整 ID 列表及特殊 token 信息：

```bash
cat > /tmp/jit-token-preflight.py <<'PY'
import json, sys
from vllm.transformers_utils.tokenizer import get_tokenizer
model, prompt_file, mode, output = sys.argv[1:]
tok = get_tokenizer(model)
with open(prompt_file, encoding='utf-8', newline='') as f:
    prompt = f.read()
ids = tok(prompt, add_special_tokens=False).input_ids if mode == 'baseline' else tok.encode(prompt, add_special_tokens=False)
with open(output, 'w', encoding='utf-8') as f:
    json.dump({'ids': ids, 'bos': tok.bos_token_id, 'special_ids': tok.all_special_ids}, f)
PY
"$BASELINE_PY" /tmp/jit-token-preflight.py "$MODEL" "$PROMPT_FILE" baseline /tmp/baseline-tokens.json
"$JIT/.venv/bin/python" /tmp/jit-token-preflight.py "$MODEL" "$PROMPT_FILE" jit /tmp/jit-tokens.json
"$CLIENT_PY" -c 'import json; a=json.load(open("/tmp/baseline-tokens.json")); b=json.load(open("/tmp/jit-tokens.json")); print("identical:",a==b,"length:",len(a["ids"])); assert a==b'
```

还需保存两端 tokenizer 文件哈希和库版本；上述本地 toy tokenizer 成功不证明目标 tokenizer 成功。

## 本地验证环境与前置改动

Windows CPU 替身测试使用 Python 3.12.14、pandas 2.2.3、numpy 2.3.5、aiohttp 3.11.18、transformers 4.51.3、datasets 3.5.1、pyarrow 19.0.1；完整依赖见 `tools/jitserve_client_validation/requirements-win-cpu.txt`。测试夹具只替代未使用的 vLLM LoRA/多模态导入，实际 tokenizer、CLI、HTTP 和原类型评分代码均执行。此环境不是推理服务。用已准备的本地测试 tokenizer 运行：

```powershell
python -m venv .venv-jitserve-test
.venv-jitserve-test/Scripts/python -m pip install -r tools/jitserve_client_validation/requirements-win-cpu.txt
.venv-jitserve-test/Scripts/python tools/jitserve_client_validation/run_smoke.py --tokenizer <local-test-tokenizer> --output-dir <new-output-directory>
```

服务端 stop-checker 测试需要放到 JITServe 副本的 `tests/test_natural_stop_standalone.py`，从其根目录 `python -m unittest discover -s tests -p test_natural_stop_standalone.py -v`。客户端还包括 approved mixed manifest 重放前置改动和 `benchmarks/r1_prefill_manifest.py`；`r1_goodput_by_type.py` 来自客户端基线提交。不要上传真实请求、tokenizer 私有资产、凭据、模型权重或运行日志。
