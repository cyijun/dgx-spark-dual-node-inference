# GB10 统一内存与 vLLM 容量判断

## 不要套用传统独显心智模型

DGX Spark 的 GB10 使用 CPU/GPU 统一内存。Linux、CUDA、模型权重、KV cache、文件缓存和
其他进程竞争同一套物理资源，因此：

- `nvidia-smi memory.total` 不是可靠的容量入口；
- `docker stats` 主要反映容器 CPU/cgroup 侧记账，不包含完整 CUDA 分配；
- Linux 的 `free` 很低不一定代表马上 OOM，应重点看 `available`；
- vLLM 启动日志中的 profiler 分解最接近推理实例自身的预算。

## 应记录哪些数字

```bash
make memory
```

脚本同时收集两节点：

- Linux `free -h`；
- CUDA compute process accounting；
- 模型加载内存；
- 权重加 non-Torch runtime 开销；
- 峰值 activation；
- CUDA Graph 实际内存；
- KV cache 分配与 token 容量。

这些视角不能直接相加，因为它们来自不同记账层。

## Qwen3.8 NVFP4 + MTP 实测

`gpu-memory-utilization=0.75` 时，每节点目标预算约 91.27 GiB。

| 项目 | Head | Worker |
|---|---:|---:|
| 主模型 + MTP draft | 约 11.07 GiB | 约 11.07 GiB |
| FP8 KV cache | 约 73.97 GiB | 约 73.81 GiB |
| 全局 KV token 容量 | 4,305,153 | 同一 TP 实例 |
| 262,144-token 理论 KV 并发 | 16.42× | 同一 TP 实例 |

未启用 MTP 时，模型约 10.67 GiB/节点，全局 KV 容量约 4,676,407 tokens。MTP draft
增加约 0.40 GiB/节点，并因为 cache 对齐填充使 token 容量下降约 7.9%。当前
`max-num-seqs=8` 仍比 KV 理论容量更保守。

## 2026-08-27 Flash 模型：统一内存峰值与双节点熔断

### GLM 自动重启的峰值

失败配置中，模型权重和 non-Torch runtime 已接近统一内存预算，多模态/视觉初始化以及
API 侧缓存又增加约 8 GiB 尾部占用。Linux 一度观测约 119 GiB used、仅约 2 GiB
available，随后 swap thrashing 并失去响应。

安全 profile 同时改变以下项目：

- `gpu_memory_utilization=0.80`；
- `max_model_len=8192`、`max_num_seqs=4`；
- `language_model_only`、`skip_mm_profiling`；
- multimodal processor cache 为 0；
- eager execution。

旧 vLLM safety profile 把 weights + non-Torch 由约 93.93 GiB 降为 91.02 GiB；当前 SGLang
原生 FP4 profile 的 Head/Worker 最低 available 为 10.82/11.76 GiB。这里的关键不是单个
参数，而是避免模型预算与第二份多模态/API 尾部共同越过物理内存。

### Qwen Flash NVFP4 与 FP8

| 项目 | NVFP4 | FP8 TP2/EP2 |
|---|---:|---:|
| 主模型/节点 | 72.29 GiB | 93.60 GiB |
| MTP/节点 | 0.29 GiB | 1.99 GiB |
| 静态内存比例 | 0.85 | 0.89 |
| Mamba cache entries | 213 | 64 |
| KV token capacity | 1,189,248 | 129,856 |
| Head 最低 available | 12.33 GiB（autotune） | 8.56 GiB |
| Worker 最低 available | 13.61 GiB（autotune） | 10.14 GiB |

FP8 的 0.85 profile 没有 OOM，而是 SGLang profiler 主动报告最小可行比例为 0.852 并退出。
提高到 0.89 后约有 4.4 GiB 用于 Mamba/KV，同时仍给 Linux/JIT 留出约 8～10 GiB 最低
余量。直接跳到 0.95 没有必要。

### 双节点 watchdog

本轮之后，内存守护必须同时读取 Head 和 Worker 的 `/proc/meminfo`：

```text
每 2 秒采样 HeadMemAvailable 和 WorkerMemAvailable
任一节点 < 4 GiB：保存 docker top/stats，停止 Head 和 Worker
SSH 暂时失败：记录 unavailable，不把未知值误当 0
```

权重读取时 Worker 可能先达到低点，JIT/warmup 时又可能是 Head 更低。只监控 API 节点
会漏掉一半风险。watchdog 是最后的熔断，不是提高 `mem_fraction_static` 的理由。

## 其他案例如何体现统一内存

| 案例 | 关键观测 | 工程结论 |
|---|---|---|
| MiniMax H3 单机 | model load 89.1659 GiB；minimum available 约 8.6～8.7 GiB | 在线 FP8 只是让路径可行，余量仍很窄 |
| MiniMax H3 双机 | rank 0 约 89.42 GiB，rank 1 约 41.44 GiB | pipeline 非对称，不能把总模型内存简单除以二 |
| DeepSeek V4 8 GiB KV | startup available 最低约 11/12 GiB | 10 GiB KV 越过 reserve，8 GiB 才是 C6/4096 安全值 |
| DeepSeek 官方 control 6 GiB KV | available 最低约 17/18 GiB，swap 不增长 | fixed KV bytes + host guard 比单一 utilization 更清楚 |

这些数字来自不同模型与 workload，不能做容量排行榜；它们共同说明每个 rank 都要单独
记录模型、KV、runtime 和宿主机余量。

## FP8 KV cache

该模型的量化配置声明 8-bit KV scheme，vLLM 使用 `--kv-cache-dtype auto` 时实际解析为：

```text
kv_cache_dtype=torch.float8_e4m3fn
```

Query 的 prefill/decode 仍为 BF16。不要把“KV cache 是 FP8”误解为所有计算都使用 FP8。

## 调参顺序

1. 先用保守的 0.70–0.75 内存利用率完成冷启动和真实请求。
2. 记录模型、activation、CUDA Graph 和 KV profile。
3. 再根据 Linux `available`、swap、并发目标小步提高。
4. 修改上下文、并发、MTP 或视觉输入预算后重新 profile。

统一内存环境中，给 OS、Docker、SSH、日志和临时编译保留余量通常比追求“显存 99%”更
重要。
