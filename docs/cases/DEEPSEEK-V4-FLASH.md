# DeepSeek V4 Flash：从 Anemll 首通到 vLLM 0.27 对照

DeepSeek V4 Flash 是这批实验里最深入的文本模型案例。它经历了两代路径：

1. Anemll 的 GB10 定制镜像先解决 SM121、MoE 和 DSpark 兼容；
2. 自建 vLLM 0.27.1 把 weight format、MoE backend、DS-MLA KV、MTP 和 CUDA Graph
   分开测量。

## 第一代：Anemll vLLM 0.25 系

| 项目 | 值 |
|---|---|
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| Model | `sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4` |
| Revision | `4b15c44c31f8dacdf7b0c2fa42e20d03c5f5859b` |
| Parallelism | TP=2，PP=1 |
| Speculation | DSpark，7 tokens，greedy draft |
| KV | FP8 |
| Context / max seq | 32,768 / 4 |
| Memory utilization | 0.88 |

镜像自带的 7-argument fused-MoE AOT 与当前调用不兼容。部署屏蔽它，使用 current-source
FlashInfer SM120 JIT，并复用镜像内 TileLang 的 CUTLASS/CUTE headers。

这条路径是成功 bring-up 记录，不是正式 benchmark 记录。

## 第二代：自建 vLLM 0.27.1

### 固定 runtime

| 项目 | 值 |
|---|---|
| Image | `infer-deploy/vllm:0.27.1-dsv4-nvfp4-ds-mla-sm121` |
| Abliterated vLLM source | `c2141dd49c8b88eaa4ff480f5d5afd30369d6775` |
| Abliterated FlashInfer source | `e6695831f237dc6bc022ddf29581ef272c862d7a` |
| Payload SHA-256 | `16ebe92097277557e224e74cef06c3cca9b3d3022842d8e0097292dc6a685115` |
| Official-control vLLM source | `d337c7046ea9670093e5e9ffe3179fc33b8e287c` |
| Official-control FlashInfer source | `6398edbbc6796d81781bd54827be860b65d8f38b` |
| Control payload SHA-256 | `b08166b8696faa7317fbab96cd5dd64338f96e402fb7e7b829c8968319fc2314` |

### Abliterated NVFP4 最终 serving envelope

- 48 NVFP4 safetensors shards；
- fixed 8 GiB `nvfp4_ds_mla` KV，24,157 tokens；
- `max_model_len=4096`、`max_num_seqs=6`、batch tokens 8192；
- DSpark MTP-5，probabilistic draft；
- target FlashInfer B12X，draft Marlin；
- `FULL_DECODE_ONLY` graphs，exact 12-token C2 batch 强制 eager；
- 108 GiB container cgroup，无 container swap；
- 10 GiB host available stop threshold，512 MiB swap-growth limit。

## Hybrid graph 为什么改善 C2

早期 graph profile 在 C2 的 exact target batch shape 上发生 padding，aggregate output
只有 27.95 tok/s。把 exact 12-token shape 强制 eager 后达到 41.58 tok/s，提升 48.7%。

| Concurrency | Hybrid | Earlier graph | Full eager |
|---:|---:|---:|---:|
| 1 | 28.95 | 32.06 | 26.58 |
| 2 | 41.58 | 27.95 | 38.22 |
| 4 | 51.22 | 50.46 | 46.01 |
| 6 | 61.74 | 63.96 | 56.23 |

所有点均为 GuideLLM 256-to-256、10 秒 warmup、60 秒 measured、零请求错误。

## W4A4 为什么没赢 W4A16

通过 in-place scale reuse，W4A16 target 不再需要额外保留约 4.3 GiB duplicate scales。
C6 结果：

| Target / draft | Output tok/s | Output tok/iter | Iter time |
|---|---:|---:|---:|
| B12X W4A16 / Marlin | 80.27 | 2.493 | 186.4 ms |
| B12X W4A16 / B12X | 69.43 | 2.442 | 211.0 ms |
| CUTLASS W4A4 / B12X | 68.40 | 2.498 | 219.1 ms |

NVFP4 checkpoint 本来就保持 packed weight。W4A4 相比 W4A16 主要减少 activation traffic，
却新增 reduction、scale、FP4 packing 和 intermediate traffic。C6 decode 的 target
verification 在 expert partition 前只有 36 routed rows，这些固定成本无法充分摊薄。

checkpoint layer、routed `M=36` microbenchmark 同样测得 W4A4 5.282 ms、W4A16
4.505 ms。结论限定为这个 GB10 workload：保存 NVFP4 weight，但 target experts 用
B12X W4A16。

## KV 名称中的陷阱

Anemll 0.1.1 的 `nvfp4_ds_mla` 名称不代表该模型的物理 KV 是 4-bit。其 sparse-MLA
路径实际每 token 为：

- 448 bytes FP8 NoPE KV；
- 128 bytes BF16 RoPE；
- 8 bytes UE8M0 block scales；
- 合计 584 bytes/token。

它调用 FP8 fused insert，并令 FlashMLA `is_fp8_kvcache=True`。官方 checkpoint 控制因此
明确写成 `fp8_ds_mla`，避免把 weight format 与 KV physical layout 混为一谈。

## 官方 checkpoint 控制

`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb` 的 dense/linear 为 FP8，target 和 MTP
experts 为 native MXFP4。target/draft 都使用 B12X W4A16，DSpark MTP-5 probabilistic，
prompt 使用 `thinking=true, reasoning_effort=low` 与参考记录对齐。

第一次 run 在测量窗口内编译额外 TileLang/Triton/CuTeDSL specialization，只保留为 cold
证据（67.99 tok/s，mean TTFT 9.91 s）。第二次无新增编译的 hot run 才是 canonical：

| C6 | Output tok/s | Output tok/iter | Iter time | TPOT | TTFT |
|---|---:|---:|---:|---:|---:|
| Current vLLM 0.27.1 hot | 97.20 | 2.706 | 167.0 ms | 61.98 ms | 743.58 ms |
| Anemll recorded | 108.18 | 2.873 | 159.3 ms | 55.67 ms | 704.29 ms |

当前 aggregate 低 10.15%。accepted tokens/iteration 低 5.82%，iteration time 慢
4.82%，不能把全部差距归因于 MoE GEMM。layer-0 draft-MoE 与 Anemll 数值 all-close，
cosine similarity 0.9999766；单层 GEMM 甚至略快，因此剩余差异包含 acceptance、
attention、scheduler、graph lifecycle 和 runtime overhead。

## 内存安全

- 10 GiB KV 试验越过 host reserve，因此 8 GiB 是 abliterated C6/4096 的安全默认；
- final formal benchmark 未跌破 guard，也没有 swap growth；
- official control 用 6 GiB physical FP8 DS-MLA KV，available 最低约 17/18 GiB；
- shutdown 后两节点恢复约 117～118 GiB available。

统一内存上，性能 profile 同时也是安全 profile。只写 `gpu-memory-utilization` 不足以复现。

## 可复用结论

- weight dtype、activation dtype 和 KV physical dtype 必须分别描述；
- speculative throughput = accepted output per target iteration × target iterations per second；
- cold JIT 结果应该保留，但不能当 steady-state；
- CUDA Graph 对特定 shape 可能更慢，hybrid dispatch 比全开/全关更合理；
- kernel microbenchmark 只能排除或定位局部瓶颈，不能替代 end-to-end；
- memory guard 应和 benchmark 同时运行，结束后验证 swap/OOM 与资源回收。
