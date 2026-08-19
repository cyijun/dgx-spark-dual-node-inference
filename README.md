# DGX Spark 推理部署实战档案

这是我在一对 NVIDIA DGX Spark 上持续部署和验证大模型的实践记录。仓库不再只描述
Qwen3.8，而是覆盖 2026 年 8 月这轮实验中留下可核验材料的 MiniMax H3、Qwen3.6、
BigBang-v1、DeepSeek V4 Flash 和 Qwen3.8。

这里同时保留两类内容：

- `scripts/` 是当前可直接复用的双机 vLLM TP=2 模板，默认 profile 为
  Qwen3.8-27B-NVFP4 + MTP。
- `docs/`、`benchmarks/` 和 `profiles/` 是多模型部署历史、实测结果与复现边界。

> [!IMPORTANT]
> 仓库不包含模型权重、Hugging Face 凭据、容器镜像、私有主机信息或生成内容。
> 模型、镜像和生成结果各自受其许可证约束；MiniMax H3 尤其需要先阅读其模型许可证。

## 实践全景

以下“验证”只描述留存证据，不把“有启动脚本”自动等同于“完成正式 benchmark”。
详细证据等级见 [验证矩阵](docs/VALIDATION.md)。

| 时间 | 模型 / 工作负载 | 拓扑与运行时 | 留存验证 |
|---|---|---|---|
| 08-03～08-04 | MiniMax H3 FL2VA | 单 Spark，vLLM-Omni，在线 FP8 | 5 项补丁测试、完整媒体解码、性能/质量/内存记录 |
| 08-03～08-04 | MiniMax H3 FL2VA | 双 Spark，Ray + Ulysses SP=2 + NCCL/RoCE | 两 rank、双 GPU、媒体验收；约 2.3× 单机速度 |
| 08-04 | Qwen3.6-27B-NVFP4 | 单 Spark TP=1，vLLM | llama-benchy prefill/decode 与 C1/C2/C4/C8 |
| 08-04 | Qwen3.6-27B-NVFP4 | 双 Spark TP=2，vLLM 原生 `mp` | 与 TP=1 同 workload 对照 |
| 08-05 | Qwen3.6-35B-A3B-NVFP4 | 双 Spark TP=2，vLLM 原生 `mp` | dense/MoE 服务基准，C1 decode 约 98 tok/s |
| 08-05 | DeepSeek V4 Flash Abliterated NVFP4 | 双 Spark TP=2，Anemll vLLM 0.25 系 | DSpark-7、FP8 KV 的首个可运行配方；无独立正式结果表 |
| 08-09 | BigBang-v1 BF16 | 双 Spark TP=2，vLLM 原生 `mp` | 15 shard/revision/RoCE/API 部署记录；无留存性能基准 |
| 08-17～08-18 | DeepSeek V4 Flash | 双 Spark TP=2，自建 vLLM 0.27.1 | 量化、KV、MoE、CUDA Graph、MTP 的多轮对照与 GuideLLM |
| 08-19 | Qwen3.8-27B-NVFP4 | 双 Spark TP=2，vLLM 0.27.2 + MTP | 后端、内存、FP8 KV、MTP 接受率和真实请求验收 |

按日期串联的演进过程见 [部署历史](docs/HISTORY.md)，按模型展开见：

- [MiniMax H3：单机兼容到双机扩散并行](docs/cases/MINIMAX-H3.md)
- [Qwen3.6：27B TP1/TP2 与 35B-A3B MoE](docs/cases/QWEN36.md)
- [BigBang-v1：71.9 GB BF16 双机 TP](docs/cases/BIGBANG-V1.md)
- [DeepSeek V4 Flash：从 Anemll 到 vLLM 0.27 自建路径](docs/cases/DEEPSEEK-V4-FLASH.md)
- [Qwen3.8 NVFP4 + MTP 验收](docs/QWEN38-NVFP4-MTP.md)

## 最值得复用的结论

1. **先固定证据，再谈性能。** 可移动的 nightly tag 不是版本；至少同时记录 image ID、
   模型 revision、运行参数和原始结果。
2. **GB10 要按统一内存评估。** 权重、KV cache、activation、CUDA Graph、JIT 和宿主机
   都竞争同一内存池，不能只看传统 `nvidia-smi memory.total`。
3. **两机路径必须显式绑定 RoCE。** NIC、HCA 和每节点动态 GID 都要分别检查；日志里应
   看到 NCCL `NET/IB`。
4. **不同模型不能套同一量化参数。** Qwen3.6 的 NVIDIA checkpoint 使用 `modelopt`，
   Qwen3.8 Unsloth checkpoint 则必须让 `compressed-tensors` 自动解析。
5. **speculative decoding 需要测接受率和迭代成本。** “加载了 draft 权重”不等于
   端到端更快；DeepSeek 的 MTP-5 和 Qwen 的单层 MTP 也不是同一种 profile。
6. **冷启动、首请求、热态必须分开。** JIT、autotune、compile 和 CUDA Graph capture
   会严重污染第一次观测。
7. **证据不完整就明确降级表述。** BigBang 和早期 DeepSeek 配方保留了成功部署路径，
   但没有可发布的正式 benchmark，因此不补写吞吐数字。

## 基准摘要

这些数字只在同模型、同 workload、同口径内比较。不要横向比较文本 tok/s 和视频秒数。
完整表格与限制见 [基准与对照](docs/BENCHMARKS.md)。

| 对照 | 代表结果 |
|---|---|
| Qwen3.6-27B TP1 → TP2，TG128 C1 | 12.21 → 21.66 tok/s，约 1.77× |
| Qwen3.6-27B TP1 → TP2，PP2048 C1 | 1208.62 → 1847.89 tok/s，约 1.53× |
| Qwen3.6-35B-A3B TP2，TG128 | C1 97.97 tok/s；C8 aggregate 243.73 tok/s |
| DeepSeek V4 Abliterated，最终 hybrid | C1/C2/C4/C6 为 28.95/41.58/51.22/61.74 output tok/s |
| DeepSeek V4 官方 checkpoint 热态控制 | C6 97.20 output tok/s；Anemll 记录为 108.18 |
| MiniMax H3 双机 full-compute warm | 46.574 s；相近单机基线约 154.956 s |
| MiniMax H3 双机 balanced Cache-DiT warm | 30.578 s；该模式为近似缓存，不是无损 |

机器可读摘要保存在 [`benchmarks/`](benchmarks/README.md)。

## 当前可复用模板：Qwen3.8 双机 TP=2

模板使用两节点各一张 GB10、vLLM 原生多节点 `mp`、单条 RoCEv2 fabric、FP8 KV cache
和 Qwen MTP。API 默认只监听 Head 的 `127.0.0.1:8888`。

~~~bash
git clone https://github.com/<owner>/dgx-spark-dual-node-inference.git
cd dgx-spark-dual-node-inference
cp .env.example .env
# 编辑所有 CHANGE_ME，并确认 image ID、模型 revision 和网络设备。

make audit
make preflight
make up
make wait
make smoke
make memory
make mtp-metrics
~~~

如果模型只在 Head 缓存：

~~~bash
make sync-model
make verify-model
~~~

停止时只删除本模板创建的两个服务容器：

~~~bash
make down
~~~

历史记录中的 `8000` 是当时的实验端口，不是集群约束。这个仓库的新部署默认使用
`8888`，启动前仍会检查占用。

## 仓库结构

| 路径 | 用途 |
|---|---|
| [`docs/HISTORY.md`](docs/HISTORY.md) | 从单机兼容、双机并行到 MTP 的部署时间线 |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | 每个案例到底验证到了哪一层 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 可比结果、计算口径和不可比边界 |
| [`docs/cases/`](docs/cases/) | 各模型的配置、故障链、结论与证据缺口 |
| [`profiles/deployments.yaml`](profiles/deployments.yaml) | 模型、revision、镜像、并行和关键参数清单 |
| [`benchmarks/`](benchmarks/) | 精简、可审计的 CSV 测量摘要 |
| [`scripts/`](scripts/) | 当前 Qwen3.8 双机 vLLM 模板及诊断工具 |
| [`.env.example`](.env.example) | 当前模板的安全占位配置 |

## 安全与复现边界

- API 默认 loopback；非 loopback 必须显式启用并设置至少 24 字符的 API key。
- 不提交 `.env`、token、私有 DNS、模型权重、JIT cache、运行日志或生成媒体。
- 旧实验只记录可核验事实；缺失的历史 image ID 不用当前 nightly ID 代替。
- MiniMax H3 的代码仓库许可证与模型/输出许可证不同，生成前必须单独确认授权。
- 这里是实验室实测档案，不是厂商 benchmark、生产 SLA 或任意硬件的支持保证。

更多内容见 [安全边界](docs/SECURITY.md)、[复现方法](docs/REPRODUCIBILITY.md)、
[网络实战](docs/NETWORKING.md) 和 [统一内存](docs/MEMORY.md)。

## License

本仓库脚本与文档采用 MIT License。模型、权重、镜像及其输出遵循各自许可证。
