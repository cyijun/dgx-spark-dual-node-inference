# 部署历史：从“能装下”到“能解释”

本页按证据产生时间整理这轮 DGX Spark 实验。日期来自已发布项目记录、结果文件中的
时间戳，或本地部署/benchmark 产物的保存时间。它表示“这份证据何时留下”，不一定等于
模型或框架的发布日期。

## 2026-08-03～08-04：MiniMax H3 单机兼容

第一个问题不是分布式，而是让约 135 GiB 的 FL2VA checkpoint 在一台 128 GB-class
统一内存设备上稳定加载。

失败链：

1. BF16 缺少可用统一内存余量；
2. INT8 进入 SM121 不支持的 kernel；
3. 初版在线 FP8 暴露 grouped QKV、loader signature、activation quantizer 和 AdaLN dtype
   四类兼容问题；
4. FlashAttention 4 的 CuTe varlen 路径不适合当时的 packed H3 shape。

最终路径使用 pinned `vllm/vllm-omni:minimax-h3`、在线 dynamic FP8、六个敏感 projection
保留高精度、cuDNN attention 和 regional compile。五项针对性回归测试通过，固定视频请求
完成 HTTP、完整 FFmpeg 解码、音视频流和内存验收。

公开完整配方：

- [MiniMax-H3-DGX-Spark](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)

## 2026-08-03～08-04：MiniMax H3 双机扩散并行

vLLM-Omni 当时的本地多进程路径会把 global rank 1 映射到单机不存在的 `cuda:1`；
CLI 虽接受 Ray backend，diffusion executor factory 却拒绝它。因此双机版本新增一个很薄的
Ray executor：

- Ray 负责跨主机 actor placement；
- 每个节点始终使用 local `cuda:0`；
- 全局 rank 0/1 通过 Ulysses sequence parallel 和 NCCL/RoCE 协同一个 denoising 轨迹；
- rank 0 保留 encoders、VAE、API 和输出编码路径。

两节点验收证明 NCCL 使用 `NET/IB`，两个 GPU 同时高负载，输出通过完整媒体解码。
相近固定请求的两次早期结果为 68.783/64.888 秒，单机对照为 154.956 秒，约 2.3×。

公开完整配方：

- [MiniMax-H3-2x-DGX-Spark](https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark)

## 2026-08-04：Qwen3.6-27B，先做 TP2/TP1 对照

同一 revision `0893e1606ff3d5f97a441f405d5fc541a6bdf404` 分别以 TP=1 和 TP=2
部署，主要目的是确认两台小 GPU 上的跨机 TP 是否值得。

共同参数：

- NVIDIA NVFP4 checkpoint，`--quantization modelopt`；
- 131,072 token context；
- `gpu-memory-utilization=0.75`、`max-num-seqs=8`；
- Qwen3 reasoning parser；
- 双机路径使用 vLLM 原生 `mp` 与显式 RoCE/NCCL 绑定。

llama-benchy 保留了 PP512/2048/8192、TG32/TG128 和 C1/C2/C4/C8。TP=2 在 TG128 C1
从 12.21 提升到 21.66 tok/s；PP2048 C1 从 1208.62 提升到 1847.89 tok/s。并发提高后
跨机 TP 的相对增益缩小，说明不能只报单请求峰值。

历史限制：当时只保存了可移动 nightly tag；TP2 README 记录 vLLM
`0.26.1rc1.dev306`，但没有保留完整 image ID。因此结果可作为本机对照，不能宣称仅凭
当前 tag 可精确复现。

## 2026-08-05：Qwen3.6-35B-A3B MoE

同一天继续验证 MoE 模型 `nvidia/Qwen3.6-35B-A3B-NVFP4`，revision
`491c2f1ea524c639598bf8fa787a93fed5a6fbce`。双机 TP=2、131k context、
`modelopt` 和原生 `mp` 均延续 Qwen3.6 路径。

代表结果：

- TG128 C1：97.97 tok/s；
- TG128 C8 aggregate：243.73 tok/s；
- PP8192：约 8.45k～8.50k tok/s。

另有一轮文件名标记为 `b12x` 的复测；其 TG128 C1 为 98.18 tok/s，C8 为
228.63 tok/s。由于该轮没有独立保存 image ID、后端解析日志或启动参数差异，本仓库只把它
当作“第二轮测量”，不把差异归因于某个 kernel。

## 2026-08-05：DeepSeek V4 Flash 的 Anemll 首通

第一条可运行路径采用 `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`，服务
`sakamakismile/DeepSeek-V4-Flash-0731-Abliterated-NVFP4`：

- TP=2，vLLM `0.25.2.dev0+g752a3a504`；
- DSpark speculative decoding，7 draft tokens，greedy draft sampling；
- FP8 KV cache，32,768 context；
- `gpu-memory-utilization=0.88`，最多 4 seq；
- 屏蔽镜像内不兼容的 7-argument fused-MoE AOT，改用当前源码 JIT。

这一步解决的是“能加载、能服务”的兼容性问题。目录没有留下独立、固定 workload 的正式
结果文件，因此不从日志片段补写性能数字。

## 2026-08-09：BigBang-v1 BF16 双机 TP

`endless-frontier/BigBang-v1` revision
`fe313c9057ca2c51a07c7cf141915dc8cc3f620e` 包含 15 个 BF16 shard，
权重总量 71,903,645,408 bytes。部署沿用原生 `mp`、TP=2、RoCE 和 131k context，
并启用 Qwen3 reasoning 与 Qwen3 Coder tool-call parser。

启动脚本会校验两节点 revision、15/15 shards、断链 symlink、镜像 ID 相等和动态 GID。
保留了服务/API 配方，但没有留存正式 benchmark，因此本仓库不提供吞吐结论。

## 2026-08-17～08-18：DeepSeek V4 深入优化

第二轮 DeepSeek 工作不再停留于“框架能跑”，而是构建
`infer-deploy/vllm:0.27.1-dsv4-nvfp4-ds-mla-sm121`，逐项分离：

- abliterated NVFP4 与官方 FP8/MXFP4 checkpoint；
- target MoE 的 CUTLASS W4A4 与 B12X W4A16；
- draft MoE 的 Marlin 与 B12X；
- eager、CUDA graph 和指定 shape 强制 eager 的 hybrid；
- true FP4 DS-MLA 与物理 FP8 DS-MLA KV；
- cold JIT、prewarm 和 hot steady-state。

8 GiB KV 的 abliterated hybrid 在 C1/C2/C4/C6 分别得到
28.95/41.58/51.22/61.74 output tok/s，全部零请求错误。之后 in-place scale reuse 让
NVFP4 checkpoint 的 B12X W4A16 target 在 C6 达到 80.27 tok/s。

最终官方 checkpoint 对照统一了权重族、物理 KV 格式、target/draft MoE、prompt 策略、
硬件、并发和 workload。热态 C6 为 97.20 tok/s，Anemll 记录为 108.18 tok/s；剩余差距
同时包含 5.82% accepted tokens/iteration 和 4.82% iteration time 两部分。

## 2026-08-19：Qwen3.8 NVFP4 + MTP

最新可复用模板升级到 vLLM `0.27.2rc1.dev110+gacb0f1dcd`，完整 image ID 和模型 revision
均固定。启动过程先后解决三类问题：

1. 强制 `modelopt` 与模型声明的 `compressed-tensors` 冲突；
2. speculative JSON 经 SSH 远端 shell 丢引号；
3. “checkpoint 有 MTP 文件”与“运行时真正启用 MTP”被混为一谈。

最终使用 `--spec-method mtp --spec-tokens 1`。日志确认 `Qwen3_5MTP`、draft model
加载与权重共享；96-token 验收请求后累计接受 41/54 个 draft tokens（75.9%）。
MTP 使每节点模型内存约从 10.67 增至 11.07 GiB，FP8 KV 全局 token 容量约下降 7.9%。

## 如何继续追加历史

新增案例时至少保存：

1. 日期、硬件/驱动快照；
2. 模型 repo、revision 和 shard/缓存完整性；
3. 镜像 tag、完整 ID、运行时 source/package 版本；
4. 并行、context、KV、量化、speculation、graph 与内存限制；
5. 冷启动、首请求、热态结果；
6. 原始结果文件、错误数、服务健康、内存/swap/OOM；
7. 哪些对照严格匹配，哪些只具方向性。
