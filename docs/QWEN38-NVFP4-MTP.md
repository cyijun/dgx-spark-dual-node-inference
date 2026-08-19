# Qwen3.8-27B-NVFP4 + MTP 验收记录

这是多模型实践档案中的最新文本服务案例，也是根目录 `scripts/` 当前实现的可运行
reference profile。其他模型的历史与测试入口见 [部署历史](HISTORY.md) 和
[验证矩阵](VALIDATION.md)。

## 固定输入

| 项目 | 值 |
|---|---|
| 模型 | `unsloth/Qwen3.8-27B-NVFP4` |
| Revision | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` |
| 模型架构 | `Qwen3_5ForConditionalGeneration` |
| MTP 架构 | `Qwen3_5MTP` |
| MTP layers | 1 |
| MTP checkpoint | `model_mtp.safetensors`，约 811 MiB，15 个 `mtp.*` tensors |
| vLLM | `0.27.2rc1.dev110+gacb0f1dcd` |
| Image ID | `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b` |
| Context | 262,144 tokens |
| Topology | TP=2、PP=1、两节点原生 `mp` |

这是 2026-08-19 的环境快照。`nightly` tag 会移动，实际复现必须同时校验完整 image ID。
机器可读配置见 [`profiles/deployments.yaml`](../profiles/deployments.yaml)。

## 最终启动选择

```text
--distributed-executor-backend mp
--tensor-parallel-size 2
--nnodes 2
--spec-method mtp
--spec-tokens 1
--gpu-memory-utilization 0.75
--max-model-len 262144
--max-num-seqs 8
--max-num-batched-tokens 8192
```

没有显式传 `--quantization`。模型 config 声明 `compressed-tensors`，让 vLLM 自动解析才能
正确加载 Unsloth Dynamic NVFP4 权重。

## 实际后端

| 模块 | vLLM 自动选择 |
|---|---|
| 全注意力 | FlashInfer |
| Decode attention | FlashInfer XQA |
| KV cache | FP8 E4M3 |
| Query dtype | BF16 |
| NVFP4 GEMM | `FlashInferCutlassNvFp4LinearKernel` |
| FP8 GEMM | CUTLASS scaled MM |
| GDN prefill | Triton/FLA |
| GDN decode | CUDA kernel |
| 视觉编码器 | FlashAttention |
| 图编译 | Torch Inductor + CUDA Graph |
| TP collective | PyNCCL/NCCL over RoCE `NET/IB` |

`compressed-tensors` 是权重格式/loader，不是 GEMM 执行后端。

## 如何证明 MTP 真正启用

仅看到缓存中存在 `model_mtp.safetensors` 不够。验收日志同时出现：

```text
Resolved architecture: Qwen3_5MTP
SpeculativeConfig(method='mtp', ..., num_spec_tokens=1)
Loading drafter model...
Detected MTP model. Sharing target model embedding weights with the draft model.
Detected MTP model. Sharing target model lm_head weights with the draft model.
```

执行生成请求后，Prometheus metrics 出现：

```text
vllm:spec_decode_num_drafts_total 54
vllm:spec_decode_num_draft_tokens_total 54
vllm:spec_decode_num_accepted_tokens_total 41
```

该小样本接受率为 75.9%。它证明 draft/verify 路径正在工作，但不能替代固定 prompt、固定
输出长度、冷/热分离的正式吞吐基准。

## 带来的限制

- 当前运行时提示 speculative decoding 下 `min_p` 和 `logit_bias` 不生效。
- 单层 MTP 配置为 1 个 draft token；盲目配置更多 token 会复用同一层，接受率和速度未必
  更好。
- MTP 增加模型和 graph 内存，并需要 cache padding，KV token 容量略降。
- 改变 vLLM commit 后应重新确认 MTP 参数、模型 registry 和指标名称。

## 三个关键故障

### 1. 强制 `modelopt` 失败

症状：

```text
Quantization method specified in the model config (compressed-tensors)
does not match ... (modelopt)
```

修复：移除 `--quantization modelopt`，由模型 config 自动选择。

### 2. SSH 远端 JSON 丢引号

Head 能解析 `--speculative-config '{...}'`，Worker 经远程 shell 后变成无效 JSON。

修复：使用无 JSON 的等价参数：

```text
--spec-method mtp --spec-tokens 1
```

### 3. 把视频 processor 的 `[ERROR]` 当成启动失败

Transformers 对 `min_frames`/`max_frames` docstring 的提示虽然带 `[ERROR]`，在本次版本中
不是致命异常。真正的失败判据应是容器 exit code、Python traceback、`/health` 和实际请求。
