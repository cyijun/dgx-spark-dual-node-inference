# MiniMax H3：单机兼容到双机扩散并行

MiniMax H3 是这批实验里唯一的视频生成模型。它不走文本模型的 vLLM TP 模板，而是
vLLM-Omni + diffusion executor；验证标准也不是“能输出 token”，而是模型加载、完整视频
生成、音视频流、全量 decode、质量与内存。

> [!IMPORTANT]
> MiniMax H3 模型与输出受单独的 Community License 约束。仓库代码的开源许可证不自动
> 授权下载、运行或公开展示模型输出。本文不分发权重或生成媒体。

## 单 Spark：为什么最终选在线 FP8

### 失败路线

| 路线 | 结果 | 原因 |
|---|---|---|
| BF16 | 未成为可用单机路径 | 约 135 GiB checkpoint 缺少实际统一内存余量 |
| INT8 | 失败 | 进入 SM121 不支持的 kernel |
| 初版在线 FP8 | 加载/运行错误 | grouped QKV、loader、activation quantizer、AdaLN dtype |
| FlashAttention 4 varlen | 首请求失败 | CuTe/CUTLASS DSL 的 packed H3 shape 兼容问题 |

成功路径在 pinned vLLM-Omni 镜像上增加四类聚焦修正：

1. grouped checkpoint QKV rows 在 native loader 前归一化；
2. 保留 vLLM native weight-loader signature；
3. FP8 activation quantizer 绑定到 SM121 支持的 native CUDA op；
4. FP8 weight conversion 后，AdaLN activation 继续使用 BF16。

六个敏感 projection 不做在线 FP8。full-compute 默认选择 cuDNN attention + regional
compile；Cache-DiT 作为可选近似 profile。

## 单机实测

| 指标 | 结果 |
|---|---:|
| Model load | 89.1659 GiB |
| Cold load | 519～543 s；最终 release readiness 约 589 s |
| SDPA/eager baseline warm | 152.911 s |
| cuDNN/compile full-compute warm | 111.373 s |
| cuDNN/compile + Cache-DiT 0.10 warm | 80.579 s |
| Matched cached vs full-compute | SSIM 0.881367；video PSNR 26.613264 dB |
| 输出 | 768×448，24 fps，H.264 + AAC stereo |
| Regression tests | 5 passed |

最终 full-compute 观测 peak used 113.01 GiB、minimum available 8.68 GiB、peak swap
3.98 GiB，零 OOM/restart。这个余量很窄，不应与另一个大模型并存。

## 双 Spark：为什么需要自定义 executor

当时的 vLLM-Omni：

- diffusion executor factory 不接受 CLI 层暴露的 Ray backend；
- local multiprocess 把 global rank 直接映射为本机 CUDA index；
- 在“一机一 GPU、两机两 rank”上，rank 1 会错误选择本机不存在的 `cuda:1`。

双机扩展只补 control/executor 边界：

~~~text
client -> Spark 1 API -> Ray executor
                         |-- rank 0 / Spark 1 / cuda:0 --\
                         |                               | Ulysses SP + NCCL/RoCE
                         |-- rank 1 / Spark 2 / cuda:0 --/
                                      |
                              rank 0 encode -> client
~~~

Ray 固定 actor 到两个 node；每个 actor 用 local rank 0 / `cuda:0`，模型仍看到 global
rank 0/1。PyTorch distributed 与 NCCL 是 tensor data plane，Ray 只是 control plane。

## 双机实测

### Public-release smoke

| Profile | Cold ready | Compile warm-up | Warm request |
|---|---:|---:|---:|
| cuDNN/compile full-compute | 588.98 s | 70.337 s | 46.574 s |
| cuDNN/compile balanced Cache-DiT | 584.91 s | 55.412 s | 30.578 s |

四个结果均为 56-frame、768×448、24 fps H.264/AAC MP4；完整 FFmpeg decode、非静音
audio 和 midpoint inspection 通过。Ray 两 node/两 GPU 正常，容器零 restart/OOM，
请求时两 GPU 同时约 94%～96% utilization。

### 内存为什么不对称

| Rank | Model load | Process GPU memory after load |
|---|---:|---:|
| rank 0 / API | 89.1659 GiB | 89.42 GiB |
| rank 1 / peer | 41.1969 GiB | 41.44 GiB |

rank 0 除了被切分的 DiT，还持有 encoders、VAE 和 output path。双机并行不是简单把全部
内存除以二。

### 质量 workload

双机 full-compute 的 1344×768、50-step request 为 1353.506 秒；balanced Cache-DiT
为 608.991 秒，低 55.0%、generation rate 为 2.22×。matched same-seed SSIM 0.8879、
video PSNR 27.04 dB，完整 decode 与视觉检查通过，但像素不同。

## 可复用结论

- 文本 TP 启动器不能直接套给 diffusion pipeline。
- 分布式初始化要区分 global rank、local rank 和 local device。
- “两个 GPU 都可见”不证明参与同一请求；需要 NCCL、rank 和同步 utilization 证据。
- 视频验收必须检查容器格式、每条 stream、完整 decode、音频和视觉内容。
- approximate cache 的速度必须和 same-seed quality 一起报告。
- 非对称 pipeline 要分别测每个 rank 的模型和 runtime 内存。

## 完整项目

- [单 Spark 兼容与优化](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)
- [双 Spark executor 与验收](https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark)

这两个项目保存完整 patch、测试、复现和模型许可证说明；本仓库只保留跨模型视角的摘要。
