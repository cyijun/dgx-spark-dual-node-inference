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
