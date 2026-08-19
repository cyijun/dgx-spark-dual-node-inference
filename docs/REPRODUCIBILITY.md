# 复现方法

## 固定四类输入

一个可复现的双机部署至少要固定：

1. **硬件/驱动**：ARM64、GB10/SM121、驱动与内核版本；
2. **镜像**：完整 image ID，不只记录可移动 tag；
3. **模型**：repo、revision、两节点完整 cache；
4. **运行参数**：TP/PP、上下文、并发、内存比例、量化和 speculative config。

`.env.example` 保存了本案例的第 2–4 类输入。硬件与动态系统信息应在每次验收时重新采集。

## 建议验收流程

```bash
make audit
make preflight
make verify-model
make up
make wait
make smoke
make memory
make mtp-metrics
```

记录两侧：

```bash
uname -a
docker version
docker image inspect <image>
nvidia-smi
rdma link show
free -h
```

## 冷启动与热请求分开

首次启动可能包含：

- checkpoint page-in；
- Torch/Dynamo/Inductor compile；
- FlashInfer autotune；
- CUDA Graph capture；
- multimodal encoder warmup。

因此启动时长、首个请求和热态请求必须分别报告。固定 sleep 不是 readiness 检查；以
`/health`、容器状态、模型 identity 和真实请求为准。

## 模型同步

Hugging Face cache snapshot 通常由相对 symlink 指向 `blobs/`。只复制 snapshot 目录会产生
断链。仓库按完整模型 cache 目录同步：

```bash
make sync-model
make verify-model
```

`verify-model` 使用 rsync checksum dry-run，会读取所有权重文件，适合部署前验收，不适合
每秒执行。

## 版本漂移策略

升级镜像或模型 revision 时：

1. 新旧配置分开保存；
2. 两节点同步新镜像并确认 ID；
3. 在空闲端口完成冷启动和 smoke；
4. 比较内存、KV 容量、后端选择、MTP 接受率和请求行为；
5. 验收后再替换原服务。

不要让 `nightly` tag 在无人检查时自动拉取并重启生产服务。
