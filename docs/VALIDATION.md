# 测试与验证矩阵

## 五层证据

本仓库用五层证据区分“写过脚本”和“验证过系统”：

| 层级 | 问题 | 最低证据 |
|---|---|---|
| S0 配置 | 输入是否可追溯？ | 模型 revision、镜像身份、参数 |
| S1 启动 | 两侧进程是否真的活着？ | 容器状态、rank/world size、无 OOM |
| S2 数据路径 | 是否真的跨机走预期链路？ | NCCL `NET/IB`、两 rank/两 GPU 活跃 |
| S3 功能 | 服务是否返回正确类型结果？ | health/model identity + 真实请求；媒体需完整 decode |
| S4 测量 | 性能或质量结论是否可复核？ | 固定 workload、原始结果、冷/热标记、错误数与边界 |

“通过 S4”也不代表不同模型之间可横向比较，只代表该案例自己的结论有完整测量链。

## 案例覆盖

| 案例 | S0 | S1 | S2 | S3 | S4 | 主要缺口 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| MiniMax H3 单机 | ✅ | ✅ | 不适用 | ✅ | ✅ | 仅一台机器；在线 FP8，不是 BF16 质量基线 |
| MiniMax H3 双机 | ✅ | ✅ | ✅ | ✅ | ✅ | 仅 batch=1 同步 FL2VA；自定义内部 executor |
| Qwen3.6-27B TP1 | ⚠️ | ✅ | 不适用 | ✅ | ✅ | historical nightly 完整 image ID 未保留 |
| Qwen3.6-27B TP2 | ⚠️ | ✅ | ✅ | ✅ | ✅ | 同上；历史端口为 8000 |
| Qwen3.6-35B-A3B TP2 | ⚠️ | ✅ | ✅ | ✅ | ✅ | `b12x` 复测缺少独立后端 provenance |
| BigBang-v1 TP2 | ⚠️ | ✅ | ✅ | ✅ | ❌ | 无固定 workload 的留存 benchmark |
| DeepSeek V4 Anemll 首通 | ⚠️ | ✅ | ✅ | ✅ | ❌ | 只有部署配方，无独立正式结果表 |
| DeepSeek V4 vLLM 0.27 | ✅ | ✅ | ✅ | ✅ | ✅ | 与 Anemll 仍是跨 runtime generation 对照 |
| Qwen3.8 NVFP4 + MTP | ✅ | ✅ | ✅ | ✅ | ⚠️ | 有功能/内存/MTP 指标，无标准化吞吐 sweep |

图例：✅ 有明确留存证据；⚠️ 部分证据或有重要限制；❌ 没有可发布证据。

## 各案例具体执行了什么

### MiniMax H3 单机

- 五项 loader/FP8/AdaLN 回归测试；
- 13/13 shard 加载与 exact model identity；
- 固定 T2VA 请求、HTTP 200、H.264/AAC、完整 FFmpeg decode；
- cold、first、两次 warm、engine/client latency；
- 每秒内存/swap、OOM/restart；
- same-seed SSIM、video/audio PSNR 与抽帧检查。

### MiniMax H3 双机

- Ray 两个 alive nodes、两 GPU allocation；
- NCCL 两 rank、两 node、`NET/IB` 与 RoCE HCA；
- 请求期间两 GPU 同时约 94%～96% utilization；
- 三个容器零 restart、`OOMKilled=false`；
- 固定 T2VA 完整媒体与非静音音频；
- full-compute 与 Cache-DiT 的 warm latency、same-seed quality 对照。

### Qwen3.6 系列

- llama-benchy PP512/2048/8192 与 TG32/TG128；
- TP1/TP2 相同模型 revision 和 workload；
- concurrency C1/C2/C4/C8，总吞吐与 per-request 指标；
- 双机脚本检查两节点模型 revision、镜像 ID 相等、RDMA ACTIVE 和动态 RoCEv2 GID。

不足是 benchmark Markdown 没有同时嵌入 historical image ID、命令行和 cold/hot 标志，
所以不能把它提升为跨机器可重复的正式报告。

### BigBang-v1

- 两节点 revision、15 shard、broken symlink 与镜像一致性检查；
- TP=2、NCCL/RoCE、API 与 parser 配置留存。

没有固定 prompt/长度/并发的结果文件。README 中的请求示例不是 benchmark。

### DeepSeek V4

- 多轮 GuideLLM 256-to-256，10 秒 warmup、60 秒 measured；
- C1/C2/C4/C6、零请求错误、256-token completed-output median；
- cold JIT 污染与第二次 hot run 分离；
- accepted tokens/target iteration 与 estimated iteration time 分解；
- container cgroup、host available、swap-growth guard 和 shutdown 回收；
- checkpoint layer microbenchmark 与数值 all-close；
- true FP4 KV、物理 FP8 DS-MLA KV、target/draft MoE 路径对照。

### Qwen3.8

- 两节点 image ID、revision、RDMA/GID 与端口 preflight；
- 日志确认实际 attention/GEMM/KV/GDN/collective backend；
- 真实 96-token Chat Completions 请求；
- MTP drafted/accepted Prometheus counters；
- 开关 MTP 前后的每节点模型内存、KV allocation 和 token capacity。

尚未保存固定 prompt、输入长度、输出长度、并发、warmup 和测量窗口齐全的吞吐 sweep，
因此 75.9% 只描述这次功能样本的接受率。

## 发布新数字前的检查单

- 使用同一 tokenizer 和 chat template；
- 固定输入/输出 token，记录 observed token；
- 分开 cold start、first request 和 hot steady-state；
- 同时保存 aggregate 与 per-request throughput；
- 记录错误、完成长度、超时和中止请求；
- speculative 模式同时保存接受率和 iteration rate；
- 记录 host available、swap、OOM、container restart；
- 把原始 JSON/CSV 与摘要一起保留；
- 对每个百分比写清分母和可比性。
