# BigBang-v1：BF16 15-shard 双机 TP

这个案例验证了比 NVFP4 checkpoint 更大的 BF16 权重集，重点是缓存完整性、跨机 TP 和
parser 配置，而不是性能调优。

## 部署身份

| 项目 | 值 |
|---|---|
| Model | `endless-frontier/BigBang-v1` |
| Revision | `fe313c9057ca2c51a07c7cf141915dc8cc3f620e` |
| Weights | BF16，15 shards，71,903,645,408 bytes |
| Runtime | `vllm/vllm-openai:nightly`，记录版本 `0.26.1rc1.dev306` |
| Parallelism | TP=2，PP=1，vLLM native `mp` |
| Context | 131,072 |
| GPU memory utilization | 0.75 |
| Max seq / batch tokens | 4 / 4096 |
| Parsers | Qwen3 reasoning；Qwen3 Coder tool calls |
| API | Head loopback，历史与当前默认均为 8888 |

## 为什么 15-shard 检查值得单独保留

Hugging Face snapshot 大量使用指向 `blobs/` 的相对 symlink。只复制 snapshot 目录时，
`config.json` 可能存在，但实际 shard 断链。这个部署在两个节点分别检查：

- `refs/main` 等于固定 revision；
- `model-*.safetensors` 恰好 15 个；
- snapshot 顶层没有 broken symlink；
- 两节点完整 image ID 相同。

这比“目录大小差不多”或“两个路径同名”更可靠。当前通用模板进一步提供 rsync checksum
dry-run，用于逐文件验证。

## 分布式与服务配置

- 一节点一 GB10、TP rank 0/1；
- Worker 先以 `--headless` 等待 rendezvous；
- `VLLM_HOST_IP`、Gloo NIC、NCCL socket NIC 和 IB HCA 显式绑定 RoCE；
- GID index 在 Head/Worker 分别动态探测；
- 关闭 custom all-reduce，使用 PyNCCL/NCCL；
- API 只绑定 loopback；容器无自动 restart policy。

## 验证边界

保留材料证明部署路径与 OpenAI-compatible API 配置存在，且启动脚本对模型、镜像和网络
做 fail-closed 检查；但没有保存固定输入/输出、并发、warmup、错误数和结果文件完整的
benchmark。

因此本文不会提供：

- tok/s；
- TTFT/TPOT；
- 与 Qwen 或 DeepSeek 的性能排名；
- BF16 相对 NVFP4 的速度/质量百分比。

未来补测应同时保存 image ID、模型 tokenization、fixed prompt/output、C1/C2/C4、
cold/hot 和每节点统一内存。
