# LTR 微批 CPU 评估工具（非生产接入）

本分支仅包含离线评估原型和测试，不修改 API、Scheduler、预测器或现有单请求线上路径。微批策略未达到可采纳的证据门槛；本目录中的队列实现不是生产实现，不应作为线上开关使用。

`microbatch_checkpoint_compare.py` 从本地 collector client JSONL 中读取固定分位位置的 8 条正式请求，并加入一条重复输入及一条合成超长输入。真实 prompt 不输出到结果文件。它在同一 CPU 进程对顺序处理和固定批次大小 2、4 做数值与饱和处理时间比较；输出 SHA、环境、长度、分数、排序、批次耗时及前后 RSS。该测试不是到达过程或服务吞吐模拟。

`microbatch_queue_prototype.py` 对相同输入做独立于 vLLM 的一个 worker 原型对照：逐条和最多 2 条一批、配置 10 ms 聚合等待、8 个在途、5 秒超时，合成固定间隔 rate=8 的到达。输入按长度选取并排序，不是服务器原始到达过程。组装等待由事件循环定时，操作系统和调度延迟可使实际值超过 10 ms，不能作为严格上界。prototype 不接入真实请求队列，输出每个请求的状态、分数差、队列和聚合等待、前台耗时、批次推理时间、超时后结果、进程 CPU 时间及前后 RSS。RSS 是检查点，不是内存峰值；每批推理耗时不能简单归因到单请求。

验证脚本需要独立、兼容的 Python/torch/transformers/tokenizers/safetensors/psutil 环境和本地 SHA 校验通过的官方模型、tokenizer。请勿以此前本机 target 目录覆盖服务器已有依赖，也不要将模型或输入数据提交 Git。两个原型脚本均要求新输出路径，不覆盖旧结果。例如：

```bash
python tools/r1_ltr_v1/microbatch_checkpoint_compare.py \
  --model /verified/model --tokenizer /verified/tokenizer \
  --client-jsonl /private/client.jsonl --output /new/path/reference.json --threads 2
python tools/r1_ltr_v1/microbatch_queue_prototype.py \
  --model /verified/model --tokenizer /verified/tokenizer \
  --client-jsonl /private/client.jsonl --reference /new/path/reference.json \
  --output /new/path/queue-results.json
```

两者固定使用同一输入文件的 SHA256 验证身份。输入文件及各输出只在授权环境内保存。微批模型输出接受 `atol=rtol=1e-5` 的浮点差；实际调度中的近同分排序不能仅凭有限样例保证一致。诊断数据只支持讨论本机和本批样本，不证明 g06 或 A800 服务性能。测试替身覆盖有限队列、超时、晚到结果、模型异常和关闭清理，不证明生产并发、真实 EngineCore IPC 或 GPU 行为。作者旧版 AUXLLM GPU 端到端仍未直接核验。
