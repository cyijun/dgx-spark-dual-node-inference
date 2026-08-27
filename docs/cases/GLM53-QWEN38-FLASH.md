# GLM-5.3 与 Qwen3.8 Flash：GB10 原生 NVFP4 路线

本案例记录 2026-08-27 在两台 DGX Spark 上对两个 NVFP4 checkpoint 的重新验证和性能优化。
结论先行：两个模型都已经进入 GB10 `SM121` 的原生 block-scaled FP4 Tensor Core kernel；
但 attention、MTP 和调优 profile 不同，不能仅凭文件名里的 `NVFP4` 横向比较整机 decode。

- `LibertAIDAI/GLM-5.3-Flash-NVFP4`：按更新后的模型卡改用 SGLang，正确生成；
- `RadixArk/Qwen3.8-Flash-Next-NVFP4`：保留 SGLang SM121 QSA fallback，启用并持久化
  FlashInfer autotune；
- 两者都验证了双 rank、NCCL `NET/IB`、OpenAI API、真实回答、固定 workload 和双节点
  `MemAvailable`。

## 固定输入

| 模型 | Revision | 运行时与镜像 |
|---|---|---|
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `aa28e1f54130286c95fee10d0705c74ce8743734` | SGLang `0.0.0.dev1+g033446bb05`；父镜像 `sha256:73f9294b…`；overlay `sha256:6b4a31ee…` |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | SGLang `0.0.0.dev1+gd91c3682b`；QSA overlay `sha256:c582904a…` |

GLM 新 revision 的 122 个 LFS 权重对象与旧 revision 相同，变化是模型卡、chat template 和
config metadata。两节点都固定完整 snapshot，不能只更新 Head 上的 `refs/main`。

## 之前实际用过哪些推理后端

| 对象 | Runtime | 权重/GEMM 与 MoE | Attention / linear attention | 推测解码 |
|---|---|---|---|---|
| GLM 旧基线 | vLLM 专用镜像、native `mp` | ModelOpt checkpoint；MoE Marlin repack | FlashInfer sparse MLA SM120 补丁 | MTP=1，累计接受率约 84% |
| GLM 原生 FP4 尝试 | vLLM + FlashInfer CUTLASS | NVFP4 CUTLASS kernel 已加载并 autotune | vLLM GLM5-Next NoPE 语义错误 | 输出为空/错误，拒绝用于性能结论 |
| GLM 最终 | SGLang TP=2 | ModelOpt NVFP4 + FlashInfer CUTLASS MoE | TileLang DSA NoPE；BF16 KV | 本轮无 MTP |
| Qwen 最终 | SGLang TP=2 | ModelOpt NVFP4 + FlashInfer CUTLASS dense/MoE | SM121 QSA reference；GDN prefill Triton、decode FlashInfer | 原生 NEXTN，3 steps / 4 drafts |
| DeepSeek 历史实验 | Anemll vLLM 0.25、自建 vLLM 0.27.1 | B12X W4A16、CUTLASS W4A4、Marlin 等 A/B | DS-MLA / `nvfp4_ds_mla` | DSpark MTP-5 |

DeepSeek 官方 checkpoint 的 experts 本身是 native MXFP4。把它放到 NVFP4/W4A4 路径会发生
格式转换或额外 activation/scale 开销，所以那轮“NVFP4 略慢”不能解释为 GB10 原生 FP4
Tensor Core 无效；它测到的是权重格式与 runtime 路径不匹配。`nvfp4_ds_mla` 也只是历史
KV backend 名称，不代表物理 KV 是 4-bit。

## GLM：模型卡更新推翻了旧 vLLM 路线

当前[模型卡](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)明确给出 2× GB10、
TP=2 的 SGLang 参数，并说明 vLLM 在 SM121 上会因 GLM5-Next 的 NoPE
(`qk_rope_head_dim=0`) 路径产生错误输出。旧 vLLM 实验虽然能加载 FlashInfer CUTLASS
原生 FP4 kernel，却在 attention 后返回空 content；因此“kernel 加载成功”不能算部署成功。

最终 profile 的关键参数为：

```text
TP=2, PP=1, API port=8888
attention=dsa
dsa_prefill=tilelang, dsa_decode=tilelang
moe_runner=flashinfer_cutlass
kv_cache=bfloat16
disable_shared_experts_fusion=true
reasoning_parser=glm45, tool_call_parser=glm47
mem_fraction_static=0.84
context_length=65536, max_running_requests=2
```

`--disable-shared-experts-fusion` 必须保留，因为 checkpoint 的 shared expert 是 BF16，只有
routed experts 是 NVFP4。TileLang 则是当前包含 `tail_dim == 0` DSA kernel 的后端。

### GB10 TileLang shared-memory 修正

stock TileLang tile 是 `block_I=64, num_stages=2, threads=256`，动态 shared memory 超过
GB10 的 101,376 B 上限。验证过的组合为 `32/1/128`。补丁位于
[`patches/20260827-flash-models/glm53/`](../../patches/20260827-flash-models/glm53/)。

这次不再使用旧的 H=32 FlashInfer sparse-MLA AOT 补丁。那个补丁解决了 vLLM TP=2 的
dispatch shape，却没有解决 NoPE attention 语义，保留为失败调查证据。

### 原生 FP4 Tensor Core 证据

运行日志确认 `quant=modelopt_fp4`、`quant_algo=NVFP4` 和
`moe_runner_backend=flashinfer_cutlass`。实际加载的
`fused_moe_120.so` / `fp4_gemm_cutlass_sm120.so` 包含：

```text
MainloopSm120...BlockScaled
float_e2m1_t
float_ue4m3_t
SM120_16x8x64
runFp4GemmImpl
```

这比仅检查 CLI 的 backend 名称更强：二进制实例同时证明 E2M1 FP4 data、UE4M3 scale 和
SM120/121 block-scaled Tensor Core 路线实际存在。真实 API 回答“巴黎”，reasoning/content
分离正确。

## Qwen：原生 FP4、autotune 与 CUDA Graph 边界

Qwen 主模型和 MTP draft 都是 ModelOpt NVFP4，dense/MoE 走 FlashInfer CUTLASS；QSA 在
GB10 上仍需 reference fallback，避开 flash-attn-4/CuTe packed-varlen 的
`MLIRError: weakly congruent`。GDN 的稳定组合是：

Qwen 镜像内的 `fused_moe_120.so` 同样包含 `MainloopSm120...BlockScaled`、E2M1/UE4M3 与
`SM120_16x8x64` 实例，并包含多组 `rmsnorm_silu_*_nvfp4` fused kernel；原生 FP4 证据不只
来自启动参数。

```text
linear_attention_prefill=triton
linear_attention_decode=flashinfer
mamba_ssm_dtype=bfloat16
cuda_graph_decode=disabled
```

本轮开启 `FLASHINFER_AUTOTUNE`，并把 cache 持久化到宿主机。target 与 draft 的
TRT-LLM fused-MoE GEMM1/GEMM2 完成 profile 选择；后续启动按 FlashInfer 版本、SM 和
runtime hash 命中独立 JSON cache。真实请求后 `spec_verify_calls_total` 增长，末批接受率
52.82%，平均接受长度 2.58。

### CUDA Graph 失败实验

第一次直接启用 decode Graph，在 QSA reference 的动态布尔索引处被 CUDA stream capture
拒绝。随后把它改成数学等价的 fixed-shape masked gather：CPU reference 对含无效 slot 和
全无效行均通过，target/draft 三类 Graph 也全部 capture 成功。

但真实生成随即损坏：MTP 接受率降到约 12%，回答出现与 prompt 无关的日期文本，
llama-benchy coherence test 失败。说明 QSA physical-slot/metadata 在 replay 时还有捕获边界，
不能只修掉 `nonzero`。失败补丁保存在
[`qsa-graph-safe-reference.patch`](../../patches/20260827-flash-models/qwen38/qsa-graph-safe-reference.patch)，
但 Dockerfile 不应用它；最终仍使用 eager decode。

## 固定 workload 结果

使用 llama-benchy `0.4.0`：PP512/PP2048、TG128、C1，以及 PP512/TG128 的
C1/C2/C4/C8；每点 3 次、`--exact-tg`、禁用 prefix cache。SGLang 请求设置
`return_token_ids=false`，由 streaming usage 精确计数。

| Profile | PP512 C1 | PP2048 C1 | TG128 C1 | TG128 C2 | TG128 C4 | TG128 C8 |
|---|---:|---:|---:|---:|---:|---:|
| GLM 旧 vLLM + Marlin + MTP | 741.92 | 1380.07 | 25.75 | 36.94 | 57.75 | 50.21 |
| GLM SGLang native NVFP4，无 MTP | 745.50 | 1522.76 | 14.72 | 27.50 | — | — |
| Qwen SGLang，autotune 关闭 | 1056.13 | 2151.06 | 26.58 | 41.07 | 68.16 | 93.66 |
| Qwen SGLang，autotune 开启 | 732.43 | 2157.12 | 30.13 | 43.63 | 56.24 | 77.80 |

PP 与 TG 都是 tokens/s；C2/C4/C8 为 aggregate decode。GLM 新路线的 PP2048 提升 10.3%，
但 decode 不能直接判定“原生 FP4 更慢”：旧 vLLM 有约 84% 接受率的 MTP，本轮 SGLang
没有启用 MTP，是一个更大的混杂变量。

Qwen autotune 对 C1/C2 分别提升 13.4%/6.2%，但 C4/C8 分别下降约 17.5%/16.9%。
FlashInfer profile 对 shape 敏感：低并发服务可以保留 autotune；以 C4/C8 为主的服务应
关闭它使用旧 profile，或按生产 shape 独立调优。PP512 本轮方差很大，不能据此宣称
prefill 回归；PP2048 基本持平。

完整数据见
[`benchmarks/flash-models-20260827.csv`](../../benchmarks/flash-models-20260827.csv)。

## 为什么之前出现 119 GiB 峰值并自动重启

GB10 的 GPU 和 CPU 共享同一套物理内存。失败的 GLM vLLM 配置中，模型与 Marlin repack
已把每节点 weights + non-Torch 占用推到约 93.93 GiB；多模态/视觉初始化和 API cache
又出现约 8 GiB 尾部增长。最终观测接近 119 GiB used、仅约 2 GiB available，随后 swap
thrashing，主机失去响应并自动重启。这不是传统独立显存的 CUDA OOM。

安全 profile 与 watchdog 的观测为：

| Profile | Head 最低 MemAvailable | Worker 最低 MemAvailable |
|---|---:|---:|
| GLM SGLang native NVFP4 | 10.82 GiB | 11.76 GiB |
| Qwen eager + autotune | 12.33 GiB | 13.61 GiB |
| Qwen 失败 Graph 实验 | 12.19 GiB | 13.27 GiB |

本轮 watchdog 每 2 秒检查两个节点，低于 6 GiB 同时停止容器。Graph 失败不是内存问题；
其最低 available 仍明显高于熔断线。

## 仍有优化空间

1. GLM 最大的 decode 机会是 SGLang 的兼容 MTP 路线；启用后必须重新记录接受率，才能与
   旧 vLLM 公平 A/B。
2. GLM TileLang `32/1/128` 是安全 tile，不保证全局最优。可在 101,376 B shared-memory
   约束内搜索 warp、block 和 stage，但每个候选都要对 gathered-attention reference。
3. Qwen 的根本瓶颈是 SM121 optimized QSA。只有替换 reference fallback 后，才值得重开
   CUDA Graph；当前 fixed-shape Python/Torch 修补已经证明“能 capture”仍可能错。
4. Qwen FlashInfer autotune 应按实际并发 shape 分 cache/A-B；不要把 C1 profile 外推到 C8。
5. 两条 200 Gb/s RoCE 当前只使用 fabric A。multi-rail 可能改善高并发 collective，但应先
   单独测 NCCL collective，不能和 kernel 改动同时上线。
