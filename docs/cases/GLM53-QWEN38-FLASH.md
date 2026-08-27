# GLM-5.3 与 Qwen3.8 Flash：双机 TP、MTP 和 SM121 兼容

本案例记录 2026-08-27 在两台 DGX Spark 上完成的三套部署：

- `LibertAIDAI/GLM-5.3-Flash-NVFP4`；
- `RadixArk/Qwen3.8-Flash-Next-NVFP4`；
- `Qwen/Qwen3.8-Flash-Next-FP8`。

三者都使用两节点、每节点一张 GB10、跨机 RoCEv2，并完成模型 revision、镜像身份、
两 rank、NCCL `NET/IB`、API、真实生成、MTP 指标和固定 workload benchmark 验证。

这不是三种量化格式的严格模型质量或 kernel A/B。GLM 使用 vLLM，Qwen 使用 SGLang；
模型结构、上下文和 speculative 配置也不同。结果只能描述各自可复现的部署状态和本机性能。

## 固定输入

| 模型 | Revision | 运行时 |
|---|---|---|
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `9e0d74e3cef17f634e84fb8e2223707e02616290` | vLLM `0.1.dev20051+g487ecf187` |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | SGLang `0.0.0.dev1+gd91c3682b` |
| `Qwen/Qwen3.8-Flash-Next-FP8` | `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` | SGLang `0.0.0.dev1+gd91c3682b` |

两个 Qwen profile 固定父镜像
`lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`，
其 ARM64 image ID 为
`sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d`。

GLM 以
[`glm-5.3-flash-nvfp4-gb10`](https://github.com/cyijun/glm-5.3-flash-nvfp4-gb10)
为参考，父镜像 digest 固定为
`sha256:4251b561d111d817765ed4097512ce36811deac071a4a7411d20242df5c74a47`。

## GLM：TP=2 改变了稀疏注意力 shape

参考项目已有 GB10/SM121 的 GLM sparse MLA 路径，但 TP=2 后每个 rank 的 attention heads
由 64 变为 32，而 FlashInfer AOT/JIT dispatch 没有 `(H=32, top-k=2176)`：

- decode dispatch 表缺少 `(32, 2176)`；
- prefill 只实例化 `ComputeMode::FP8, 64, 2176, 64`；
- 直接复用 H=64 kernel 会在 shape 或 dispatch 阶段失败。

本仓库的 [`patches/20260827-flash-models/glm53/`](../../patches/20260827-flash-models/glm53/)
只增加 H=32/top-k=2176 dispatch 与 AOT 实例，不修改其他 architecture。最终配置：

```text
TP=2, PP=1, vLLM native mp
attention=FLASHINFER_MLA_SPARSE_SM120
KV=fp8_ds_mla, block_size=256
MoE=Marlin, eager
MTP=1 speculative token
max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=4096
gpu_memory_utilization=0.80
language_model_only=true, skip_mm_profiling=true, mm_processor_cache=0
```

### 为什么机器曾经自动重启

GB10 的 GPU 分配和 Linux 进程共享同一套物理内存。失败配置中，模型已经接近吃满统一
内存，多模态/视觉路径及 API 侧缓存又带来约 8 GiB 尾部增长；观测一度达到约 119 GiB
used、仅约 2 GiB available，随后出现 swap thrashing 和主机失去响应。

安全 profile 将 weights + non-Torch 观测从约 93.93 GiB 降到 91.02 GiB，并避免视觉/API
尾部增长。启动与 benchmark 期间 Head 最低 `MemAvailable` 为 12.13 GiB。这个 A/B 支持
“多模态初始化 + 过高统一内存预算共同造成峰值”的判断，但不是传统独显 OOM。

## Qwen：QSA、GDN 与 MTP 的 SM121 路径

两个 Qwen checkpoint 共用相同的 Qwen Sparse Attention、GDN 和 MTP 兼容层。

### QSA CuTe varlen 失败

原始 packed-varlen QSA 路径在 GB10 上会由 FlashAttention/CuTe 抛出
`MLIRError: weakly congruent`。关闭 autotune 只会把失败移动到 CUDA Graph capture；关闭
decode graph 后，真实 MTP draft decode 仍会进入同一个 CuTe 路径。

[`patches/20260827-flash-models/qwen38/`](../../patches/20260827-flash-models/qwen38/)
仅在 compute capability `(12, 1)`：

1. 把逻辑 top-k 索引映射为物理 KV slots；
2. 使用 SGLang 已有的 device-agnostic `qsa_sparse_attention` reference operation；
3. 保留 QSA 与 MTP 语义；其他架构继续使用优化 kernel。

这是正确性 fallback，不是性能优化。若上游 CuTe 后续原生支持该 shape，应重新测试并优先
移除 fallback。

### GDN dtype 约束

同一组后端组合存在相反要求：FlashInfer prefill 期望 checkpoint 的 FP32 状态，而 SM100+
FlashInfer decode 要求 Mamba SSM 为 BF16。可运行组合为：

```text
linear_attention_prefill=triton
linear_attention_decode=flashinfer
mamba_ssm_dtype=bfloat16
disable_flashinfer_autotune=true
cuda_graph_decode=disabled
```

Qwen MTP 使用原生 NEXTN：3 steps、top-k 1、4 draft tokens。真实生成后
`spec_verify_calls_total` 增长，证明 draft/verify 链实际运行，不只是加载了 MTP 权重。

## NVFP4 profile

NVFP4 使用 ModelOpt FP4、FlashInfer CUTLASS、page size 64、32,768 context、最多 8 个请求，
`mem_fraction_static=0.85`。每节点主模型加载约 72.29 GiB，MTP 约 0.29 GiB；基准期间
Head 最低 `MemAvailable` 为 13.68 GiB。

整轮结束时有 3,638 次 speculative verify。Prometheus gauge 记录的末批接受率为 52.83%，
平均接受长度为 2.58；它不是整轮累计接受率。

## FP8 profile：Attention TP=2，MoE EP=2

FP8 checkpoint 的 MoE intermediate size 是 640，block-wise 量化的 `block_n=128`。纯
MoE TP=2 会把每个 expert 切为 320，SGLang 因 `320 % 128 != 0` 在创建权重时主动拒绝。

解决方案不是填充 checkpoint，而是：

```text
attention TP=2
MoE EP=2, standard NCCL all-to-all
MoE TP size=1, so each local expert keeps intermediate_size=640
```

主权重最终占每节点 93.60 GiB，MTP 占 1.99 GiB。`mem_fraction_static=0.85` 低于带 KV
缓存的最小可行值 0.852，因此框架会安全退出。最终使用 0.89，并把 Mamba cache 上限设为
64：

- Mamba cache 约 2.75 GiB；
- target + draft KV 约 1.60 GiB；
- KV token capacity 129,856；
- 启动和 benchmark 全程最低 `MemAvailable`：Head 8.56 GiB，Worker 10.14 GiB。

整轮结束时有 3,776 次 verify，末批接受率 53.59%，平均接受长度 2.61。

## 基准方法

使用 llama-benchy `0.4.0`：

- baseline：PP512/PP2048，TG128，C1；
- concurrency：PP512，TG128，C1/C2/C4/C8；
- 每个点 3 次；`--exact-tg`；关闭 prefix cache；
- SGLang 的 streaming chat 不支持同时 `return_token_ids=true`，因此请求关闭该字段，
  由流式 usage 精确计数；
- 保存 prompt throughput、aggregate generation throughput、TTFT 和 MTP metrics。

| 模型 | PP512 C1 | PP2048 C1 | TG128 C1 | TG128 C2 | TG128 C4 | TG128 C8 |
|---|---:|---:|---:|---:|---:|---:|
| GLM-5.3 NVFP4 | 741.92 | 1380.07 | 25.75 | 36.94 | 57.75 | 50.21 |
| Qwen3.8 Flash NVFP4 | 1056.13 | 2151.06 | 26.58 | 41.07 | 68.16 | 93.66 |
| Qwen3.8 Flash FP8 | 898.38 | 1905.23 | 25.93 | 39.69 | 52.29 | 80.64 |

PP 与 TG 数字均为 tokens/s；C2/C4/C8 是 aggregate generation throughput。GLM 的 C8
超过 `max_num_seqs=4`，发生排队，不能把 50.21 tok/s 当作其最佳并发点。完整精简数据见
[`benchmarks/flash-models-20260827.csv`](../../benchmarks/flash-models-20260827.csv)。

## 内存保护

本轮使用双节点 watchdog，每 2 秒读取 Head 和 Worker 的 Linux `MemAvailable`。任一节点
低于 4 GiB 时同时停止两端容器，并在停止前保存进程和容器内存快照。FP8 启动 JIT 的瞬时
最低值出现在 Head，而权重读取中也曾出现 Worker 比 Head 更低；只监控发起端不够。

watchdog 不能替代保守预算。它只用于在统一内存开始失控、但系统仍能调度 shell/SSH 时
提供最后一道可恢复保护。
