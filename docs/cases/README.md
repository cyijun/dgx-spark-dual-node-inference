# 模型案例索引

| 案例 | 重点 |
|---|---|
| [MiniMax H3](MINIMAX-H3.md) | 单机在线 FP8 兼容、双机 Ray/Ulysses、视频性能与质量 |
| [Qwen3.6](QWEN36.md) | 27B TP1/TP2 与 35B-A3B MoE 并发曲线 |
| [BigBang-v1](BIGBANG-V1.md) | BF16 15-shard cache 完整性与双机 TP |
| [DeepSeek V4 Flash](DEEPSEEK-V4-FLASH.md) | DSpark、DS-MLA KV、MoE backend、graph 与 hot control |
| [Qwen3.8 NVFP4 + MTP](../QWEN38-NVFP4-MTP.md) | 当前可运行模板、FP8 KV、实际后端与 MTP 证据 |
| [GLM-5.3 / Qwen3.8 Flash](GLM53-QWEN38-FLASH.md) | SM121 sparse MLA/QSA、MoE TP/EP、MTP、统一内存和完整 benchmark |

先看 [部署历史](../HISTORY.md) 理解演进，再看 [验证矩阵](../VALIDATION.md) 判断每个案例
到底能支持什么结论。
