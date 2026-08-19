# 精简测量数据

这里保存从原始实验产物摘出的机器可读摘要，方便复核文档中的数字。

| 文件 | 内容 |
|---|---|
| `qwen36-summary.csv` | Qwen3.6 27B TP1/TP2 与 35B-A3B 的 llama-benchy 代表点 |
| `deepseek-v4-summary.csv` | DeepSeek V4 GuideLLM、target/draft 与官方控制代表点 |
| `minimax-h3-summary.csv` | MiniMax H3 单/双机固定视频请求代表点 |

这些 CSV 不是全部原始日志，也不重新定义原工具字段。每行保留 `source_record`，指向
产生它的历史结果文件名或发布记录。完整解读与不可比边界见
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)。

数据发布规则：

- 不包含 prompt 文本之外的用户内容；
- 不包含主机名、用户名、私有 IP、token、权重或生成媒体；
- 不把 cold/JIT 结果混入 hot steady-state；
- 缺少 runtime provenance 的记录明确标注；
- 百分比和 speed ratio 在文档中计算，CSV 保留原始代表值。
