# 基准、对照与解读

## 先说口径

- `PP<n>`：prefill 约 n 个 token，表中为 prompt processing throughput。
- `TG<n>`：生成约 n 个 token，表中为 generation throughput。
- `C1/C2/C4/C8`：并发请求数；并发表报告 aggregate 和 per-request 是两个不同问题。
- GuideLLM 表使用 `metrics.output_tokens_per_second.total.mean`。
- MiniMax H3 报告一个固定视频请求的 client elapsed，不是 tokens per second。

本页只做同模型、同 workload 内的对照。原始精简数据见 [`benchmarks/`](../benchmarks/)。

## Qwen3.6-27B：TP1 与跨机 TP2

相同 checkpoint revision、131k context、`gpu-memory-utilization=0.75`。代表值取自保存的
llama-benchy 表。

| Workload | TP1 | TP2 | TP2 / TP1 |
|---|---:|---:|---:|
| PP512 C1 | 1285.53 tok/s | 2286.21 tok/s | 1.78× |
| PP2048 C1 | 1208.62 tok/s | 1847.89 tok/s | 1.53× |
| PP8192 C1 | 1161.35 tok/s | 1731.98 tok/s | 1.49× |
| TG128 C1 | 12.21 tok/s | 21.66 tok/s | 1.77× |
| TG128 C2 aggregate | 21.67 tok/s | 34.06 tok/s | 1.57× |
| TG128 C4 aggregate | 38.03 tok/s | 49.05 tok/s | 1.29× |
| TG128 C8 aggregate | 50.76 tok/s | 64.57 tok/s | 1.27× |

结论不是“TP2 恒定快 1.8×”。单请求 decode 接近 1.77×，但到 C8 aggregate 只剩约
1.27×；跨机 collective 和单请求/批处理形状共同决定收益。

## Qwen3.6-35B-A3B：MoE 并发曲线

| Workload | 第一轮 | 标记 `b12x` 的复测 |
|---|---:|---:|
| PP8192 C1 | 8500.83 tok/s | 8722.82 tok/s |
| TG128 C1 | 97.97 tok/s | 98.18 tok/s |
| TG128 C2 aggregate | 134.27 tok/s | 136.86 tok/s |
| TG128 C4 aggregate | 189.97 tok/s | 181.39 tok/s |
| TG128 C8 aggregate | 243.73 tok/s | 228.63 tok/s |

两轮低并发相近，高并发并没有单向改善。由于复测未把后端解析日志和完整 image identity
与结果绑定，`b12x` 只是历史文件标签，不能作为某 kernel 优劣的因果证据。

## 2026-08-27 Flash 模型：完整 PP/TG sweep

三套服务统一使用 llama-benchy `0.4.0`、exact TG128、关闭 prefix cache、每点 3 次。
GLM 是 vLLM；两个 Qwen 是 SGLang，因此表格用于记录各 profile，不构成框架或量化 A/B。

### 单并发 baseline

| 模型 | PP512 | PP512 TTFT | PP2048 | PP2048 TTFT | PP512 后 TG128 | PP2048 后 TG128 |
|---|---:|---:|---:|---:|---:|---:|
| GLM-5.3 Flash NVFP4 | 741.92 | 694.56 ms | 1380.07 | 1487.71 ms | 25.75 | 24.51 |
| Qwen3.8 Flash-Next NVFP4 | 1056.13 | 491.11 ms | 2151.06 | 955.13 ms | 26.58 | 24.98 |
| Qwen3.8 Flash-Next FP8 | 898.38 | 582.31 ms | 1905.23 | 1082.69 ms | 25.93 | 19.67 |

除 TTFT 外单位均为 tok/s。相同模型在 PP512 和 PP2048 后的 decode 数字不同，说明短窗口
波动与 prompt shape 都不可忽略。

### PP512 / TG128 并发总吞吐

| 模型 | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| GLM-5.3 Flash NVFP4 | 24.65 | 36.94 | 57.75 | 50.21 |
| Qwen3.8 Flash-Next NVFP4 | 23.30 | 41.07 | 68.16 | 93.66 |
| Qwen3.8 Flash-Next FP8 | 28.07 | 39.69 | 52.29 | 80.64 |

GLM 的 C8 超过 `max_num_seqs=4`，会排队；其 TTFT 从 C4 的 1445.62 ms 增至 C8 的
7659.85 ms。因此 C8 下降是 profile 容量边界，不应解读成硬件的最佳吞吐。

Qwen NVFP4/FP8 的末批 MTP 接受率分别为 52.83%/53.59%，平均接受长度为 2.58/2.61；
GLM 整轮累计接受 4074/4822 个 draft token（84.49%）。Prometheus gauge 与累计 counter
口径不同。机器可读数据见
[`benchmarks/flash-models-20260827.csv`](../benchmarks/flash-models-20260827.csv)。

## DeepSeek V4：把接受率与 runtime 分开

### Abliterated NVFP4，8 GiB KV hybrid

固定 256-to-256、10 秒 warmup、60 秒测量，全部零请求错误。

| Concurrency | Output tok/s | Full eager | Earlier graph | Anemll 方向性参考 |
|---:|---:|---:|---:|---:|
| 1 | 28.95 | 26.58 | 32.06 | 41.27 |
| 2 | 41.58 | 38.22 | 27.95 | 64.30 |
| 4 | 51.22 | 46.01 | 50.46 | 90.87 |
| 6 | 61.74 | 56.23 | 63.96 | 108.18 |

C2 hybrid 比 earlier graph 高 48.7%，主要解决 exact 12-token batch 被 graph padding
拖慢的问题。C1/C6 的短窗口差异同时受 speculative acceptance 影响，不能只归因于 graph。

### Target/draft MoE 路径

| Target path | Draft path | C6 output tok/s | Output tok/iter | Estimated iter |
|---|---|---:|---:|---:|
| NVFP4 weight，B12X W4A16 | Marlin | 80.27 | 2.493 | 186.4 ms |
| NVFP4 weight，B12X W4A16 | B12X | 69.43 | 2.442 | 211.0 ms |
| NVFP4 weight，CUTLASS W4A4 | B12X | 68.40 | 2.498 | 219.1 ms |

这个 GB10 C6 decode shape 下，保持 NVFP4 packed checkpoint、target expert 用 W4A16
更快。W4A4 的动态 activation reduction/scaling/packing 开销没有被很小的 routed M
摊薄。该结论限定于当前 shape 和实现，不推广为所有 FP4 workload。

### 官方 checkpoint 热态控制

| C6 256-to-256 | Output tok/s | Output tok/target iter | Estimated iter | Mean TPOT | Mean TTFT |
|---|---:|---:|---:|---:|---:|
| 当前 vLLM 0.27.1 hot | 97.20 | 2.706 | 167.0 ms | 61.98 ms | 743.58 ms |
| Anemll 记录 | 108.18 | 2.873 | 159.3 ms | 55.67 ms | 704.29 ms |

10.15% aggregate gap由两部分组成：当前接受 token/iteration 低 5.82%，estimated
iteration time 慢 4.82%。只看 tok/s 无法判断是 draft 质量还是 runtime kernel。

## MiniMax H3：速度必须和质量一起报

### 固定 768×448、20-step、2-second T2VA

| 拓扑 / profile | Warm client |
|---|---:|
| 单 Spark，SDPA/eager baseline | 152.911 s |
| 单 Spark，cuDNN/compile full-compute | 111.373 s |
| 单 Spark，cuDNN/compile + Cache-DiT 0.10 | 80.579 s |
| 双 Spark，cuDNN/compile full-compute | 46.574 s |
| 双 Spark，cuDNN/compile balanced Cache-DiT | 30.578 s |

单机和双机的最终 public-release 数字来自不同验收批次，适合说明工程演进；严格的早期
同输入对照是双机 68.783/64.888 秒与单机 154.956 秒，约 2.3×。

### 1344×768、50-step 质量 workload

| 双机 profile | Client elapsed | 相对 full-compute | Same-seed quality |
|---|---:|---:|---|
| Full compute | 1353.506 s | baseline | 对早期 SDPA：SSIM 0.9134 |
| Cache-DiT 0.15 | 608.991 s | 55.0% lower，2.22× rate | 对 matched full compute：SSIM 0.8879，PSNR 27.04 dB |

Cache-DiT 结果通过完整 decode 和视觉检查，但像素发生变化，因此它是速度/质量 profile，
不是无损优化。

## 当前还不能回答的问题

- BigBang-v1 没有固定 workload 结果，不能给出 tok/s。
- Qwen3.8-27B 有 MTP 功能接受率和内存差异，但没有正式吞吐 sweep，不能声称 MTP 带来
  某个百分比的加速；这与 08-27 的 Flash-Next sweep 是不同 checkpoint。
- Qwen3.6 historical nightly 缺完整 image ID，无法做跨时间严格复现。
- DeepSeek 当前与 Anemll 是跨 runtime generation 的控制，不是单 commit kernel A/B。
- 文本模型和视频扩散模型的数字不可放在一张“排行榜”里排序。
