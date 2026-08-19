# Qwen3.6：27B TP1/TP2 与 35B-A3B MoE

Qwen3.6 这组实验建立了最早的“同模型 TP1/TP2”对照，也形成后续 Qwen、BigBang 和
DeepSeek 双机脚本的基本骨架：两节点本地 cache、vLLM 原生 `mp`、RoCE 动态 GID、
Worker 先启动、Head 提供 loopback API。

## 固定配置

| 项目 | 27B TP1 | 27B TP2 | 35B-A3B TP2 |
|---|---|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` | 同左 | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Revision | `0893e160...f404` | 同左 | `491c2f1e...fbce` |
| Runtime | vLLM nightly | vLLM `0.26.1rc1.dev306` | vLLM `0.26.1rc1.dev306` |
| Quantization | `modelopt` | `modelopt` | `modelopt` |
| Context | 131,072 | 131,072 | 131,072 |
| GPU memory utilization | 0.75 | 0.75 | 0.75 |
| Max seq / batch tokens | 8 / 8192 | 8 / 8192 | 8 / 8192 |
| Executor | local | native `mp` | native `mp` |

完整 revision 在 [`profiles/deployments.yaml`](../../profiles/deployments.yaml) 中保存。
旧 nightly 的完整 image ID 没有留存，这是复现缺口。

## 27B：TP2 的收益随 workload 改变

| Workload | TP1 | TP2 | Speed ratio |
|---|---:|---:|---:|
| PP512 C1 | 1285.53 | 2286.21 | 1.78× |
| PP2048 C1 | 1208.62 | 1847.89 | 1.53× |
| PP8192 C1 | 1161.35 | 1731.98 | 1.49× |
| TG128 C1 | 12.21 | 21.66 | 1.77× |
| TG128 C8 aggregate | 50.76 | 64.57 | 1.27× |

单请求 decode 的跨机 TP 收益明显；并发升高后，TP1 自身 batching 提高 aggregate
throughput，TP2 的相对比率缩小。选择 TP=1 或 TP=2 应同时看 latency、aggregate
throughput、并发和第二台机器的机会成本。

## 35B-A3B：MoE 改变吞吐量级

`nvidia/Qwen3.6-35B-A3B-NVFP4` 是 35B 总参数、A3B active 的 MoE。保存结果中：

| Concurrency | TG128 aggregate | TG128 per request |
|---:|---:|---:|
| 1 | 97.97 tok/s | 97.97 tok/s |
| 2 | 134.27 tok/s | 73.05 tok/s |
| 4 | 189.97 tok/s | 61.31 tok/s |
| 8 | 243.73 tok/s | 43.86 tok/s |

aggregate 随并发上升，但 individual request throughput 下降。这比只报“243.73 tok/s”
更接近真实服务取舍。

第二套结果文件名带 `b12x`：

| Concurrency | 第一轮 | `b12x` 文件 |
|---:|---:|---:|
| 1 | 97.97 | 98.18 |
| 2 | 134.27 | 136.86 |
| 4 | 189.97 | 181.39 |
| 8 | 243.73 | 228.63 |

由于没有把第二轮对应的 image ID、resolved kernel 和参数变更一起保存，不能仅凭文件名
断言 B12X 导致这些差异。这也是后来 DeepSeek 实验坚持保存 runtime payload hash、
backend 解析和 cold/hot 原始 JSON 的直接原因。

## 验证方式

双机启动前检查：

1. Head/Worker cache 的 `refs/main` 与固定 revision 相同；
2. 两节点 image ID 相同；
3. 两节点 RDMA link 为 ACTIVE；
4. 每个节点独立查找 IPv4-mapped RoCEv2 GID；
5. 目标容器不存在；
6. Worker rank 1 先启动，Head rank 0 后启动。

llama-benchy 保存：

- baseline：PP512/2048/8192，TG32/TG128；
- concurrency：PP2048/TG128，C1/C2/C4/C8；
- mean、方差、peak、TTFR/TTFT 等字段。

## 从 Qwen3.6 继承到后续项目的经验

- 模型路径相同不代表两节点 revision 相同；
- tag 相同只应作为“镜像相等”的运行前检查，发布时还要记录完整 ID；
- TP1/TP2 比较必须用相同模型 revision 和相同 benchmark；
- aggregate 和 per-request throughput 必须同时读；
- 历史端口 `8000` 只是实验选择，后续新部署统一默认 `8888`。
