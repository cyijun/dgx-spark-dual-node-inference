# 架构选择与当前模板启动顺序

这个实践档案包含三类并行架构，不能用一套启动参数互换：

| 案例 | 控制面 | 模型并行 | 数据面 |
|---|---|---|---|
| Qwen / BigBang / DeepSeek 文本模型 | vLLM native multiprocessing | Tensor Parallel | PyNCCL/NCCL over RoCE |
| MiniMax H3 单机 | vLLM-Omni | 无跨机并行 | 本地 CUDA |
| MiniMax H3 双机 | Ray actor placement | Ulysses sequence parallel | PyTorch distributed/NCCL over RoCE |

下面的拓扑与脚本描述仓库根目录当前可运行的 Qwen3.8/vLLM TP=2 模板。MiniMax H3
的自定义 diffusion executor 见 [案例页](cases/MINIMAX-H3.md) 和对应独立项目。

## 拓扑

```text
Head node                                  Worker node
┌──────────────────────────────┐           ┌──────────────────────────────┐
│ vLLM API + EngineCore        │           │ vLLM headless worker         │
│ TP rank 0 / cuda:0           │◄─────────►│ TP rank 1 / cuda:0           │
│ 127.0.0.1:8888               │  NCCL/IB  │ no HTTP API                  │
│ 10.100.32.1 / RoCE fabric    │           │ 10.100.32.2 / RoCE fabric    │
└──────────────────────────────┘           └──────────────────────────────┘
```

Head 同时承担 API、调度和 TP rank 0。Worker 以 `--headless` 运行 rank 1。两个 rank
通过 `--master-addr`/`--master-port` 建立进程组，通过 NCCL 在指定 RoCE fabric 上完成
collective。

## 当前模板为什么使用 vLLM 原生 multiprocessing

本案例只有两个节点、每节点一张 GPU。vLLM 的多节点 `mp` executor 已能直接表达这个
拓扑，不需要额外启动 Ray head、Ray worker 或 dashboard。减少控制面组件也减少了端口、
版本和故障面。

这不意味着 Ray 永远不合适。MiniMax H3 双机实验正是因为当时 vLLM-Omni 的 diffusion
路径没有可用的跨主机 multiprocessing executor，才使用 Ray 做跨机 actor placement。
选择的是模型和框架实际支持的执行边界，不是统一偏好某个 launcher。

## 启动顺序

1. Head 执行 `preflight.sh`，同时通过 SSH 检查 Worker。
2. 分别探测两节点当前 IPv4 RoCEv2 GID index。
3. 先启动 Worker rank 1，使其等待 rendezvous。
4. 再启动 Head rank 0 和 API。
5. `wait-ready.sh` 同时监控容器状态和 `/health`。
6. `/health` 成功后检查 `/v1/models`，再执行真实生成请求。

脚本采用 fail-closed：镜像 ID、模型 revision、RDMA link、API bind 或已有容器不符合预期
时不会继续启动。

## 数据路径与控制路径

- SSH、Docker 控制、日志读取属于控制路径，可以走管理网络。
- `VLLM_HOST_IP`、Gloo、NCCL socket 和 IB HCA 全部绑定 RoCE fabric。
- Docker 使用 host network，避免容器 bridge 被误选为跨节点数据路径。
- 模型缓存在两节点本地 NVMe 上，不依赖共享文件系统。

## 容器边界

容器获得 GPU、host IPC、`IPC_LOCK` 和检测到的 `/dev/infiniband/rdma_cm`、`uverbs*`
设备。模型目录只读挂载，并设置 Hugging Face/Transformers 离线模式，避免启动时因为
网络或远端 revision 变化而产生不可复现行为。
