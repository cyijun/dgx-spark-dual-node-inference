# 故障排查

先看两侧容器状态，不要只看 Head：

```bash
make status
docker logs --tail 200 <prefix>-head
ssh user@spark-worker.local docker logs --tail 200 <prefix>-worker
```

| 症状 | 常见原因 | 处理 |
|---|---|---|
| Head 等待、Worker 很快 exit 2 | 远端 CLI 引号被 shell 处理 | 避免 JSON；MTP 用 `--spec-method`/`--spec-tokens` |
| `compressed-tensors` 与 `modelopt` 不匹配 | 套用了其他模型的量化参数 | 清空 `.env` 中 `QUANTIZATION`，让模型 config 决定 |
| NCCL 走 socket 或管理网 | 接口/HCA 未显式绑定 | 检查 `VLLM_HOST_IP`、Gloo、NCCL NIC/HCA |
| `NCCL WARN` 或 rendezvous timeout | 两节点 IP、端口、GID、RDMA device 不一致 | 重新执行 `make preflight`，不要复用旧 GID |
| 两节点镜像 tag 相同但行为不同 | tag 指向不同 image ID | 比较完整 `.Id`，同步镜像并设置 `EXPECTED_IMAGE_ID` |
| 找不到模型文件或 snapshot 断链 | 只复制了 snapshot symlink，没有复制 blobs | 同步整个 `models--org--repo` 缓存目录 |
| 模型 revision 不一致 | 两节点缓存不是共享文件系统 | `make sync-model && make verify-model` |
| API 端口“被占用”但另一个地址可用 | 服务绑定在同端口的其他具体 IP | 检查实际 bind address；模板按 `API_BIND` 尝试绑定 |
| `/health` 未就绪数分钟 | 首次 Torch compile、FlashInfer autotune、CUDA Graph | 确认进程仍活跃并等待 `make wait`，不要只看固定 sleep |
| 日志显示 `No available shared memory broadcast block` | 某 rank 正在长时间编译/调优 | 先观察另一 rank 是否继续前进；不等同于 hang |
| `docker stats` 很低但系统内存很高 | CUDA 统一内存不在该视图完整展示 | 使用 `make memory` 和 vLLM profiler 日志 |
| MTP 文件存在但无接受率指标 | 没启用 speculative config，或尚无完成请求 | 查 `Qwen3_5MTP`、`SpeculativeConfig`，执行 `make smoke` |
| speculative decoding 的 `min_p` 无效 | 当前 vLLM 限制 | 不依赖 `min_p`/`logit_bias`，或关闭 MTP |
| QSA 抛出 CuTe `weakly congruent` | SM121 不支持当前 packed-varlen shape | 先区分 autotune/graph/真实 decode；使用限定 SM121 的 physical-slot reference fallback |
| FP8 MoE 报 `320 is not divisible by block_n=128` | 640 intermediate 被 MoE TP=2 切半 | 保持 Attention TP=2，将 MoE 改为 EP=2，使 local expert 保留 640 |
| 权重加载后提示 static fraction 没有 KV 空间 | 模型 + MTP 已超过静态预算 | 使用 profiler 给出的 minimum viable，限制 Mamba cache，再小步提高 fraction |
| SGLang streaming chat 拒绝 `return_token_ids=true` | 当前 streaming API 不支持同时返回 token IDs | benchmark 关闭 `return_token_ids`，使用 streaming usage 计数并保留 exact TG |
| 只有 Head 内存正常，Worker 重启 | 两 rank 的加载/JIT 峰值不同步 | watchdog 同时采集两节点，任一节点越线就停止整个实例 |
| API 端口 probe 误报，但只看到其他本地地址监听 | 具体地址绑定与 socket reuse 语义不同 | probe 绑定实际 API 地址并设置 `SO_REUSEADDR`，仍需检查 loopback 的真实 listener |

## 分层排查顺序

1. **文件层**：模型 revision、broken symlink、incomplete 文件、两节点 checksum。
2. **镜像层**：ARM64、完整 image ID、镜像内 vLLM/CUDA 版本。
3. **网络层**：IP、NIC、HCA、ACTIVE/LINK_UP、GID、peer ping。
4. **分布式层**：rank、world size、master address/port、NCCL `NET/IB`。
5. **模型层**：architecture、quantization loader、权重加载、KV profiling。
6. **服务层**：`/health`、`/v1/models`、真实生成、MTP metrics。

按层排查比同时改十个环境变量更快，也更容易保留可复现记录。
