# 安全边界

## 默认只监听 loopback

`.env.example` 使用：

```dotenv
API_BIND=127.0.0.1
API_PORT=8888
API_KEY=
ALLOW_REMOTE_API=false
```

这意味着只有 Head 本机可以直接访问 API。推荐通过 SSH tunnel 或已有的私有 overlay
网络代理访问，而不是把未认证的 OpenAI-compatible API 暴露到公网。

## 非 loopback 需要显式解锁

脚本拒绝在以下条件不满足时绑定非 loopback 地址：

- `ALLOW_REMOTE_API=true`；
- `API_KEY` 至少 24 个字符。

这只是最低限度护栏，不替代 TLS、ACL、速率限制、审计和密钥轮换。

## SSH 与 Worker

- 使用专用、最小权限的 SSH key。
- 将 Worker host key 固定在 `known_hosts`，不要在自动化里关闭 host-key 检查。
- Worker 启动命令通过 SSH 发送；不要把 token 或复杂未转义 JSON 放进命令行。
- 仓库的 `.env` 被 `.gitignore` 排除，发布前仍应执行 `make audit`。

## 容器权限

模板使用 host network、host IPC、GPU、`IPC_LOCK` 和 RDMA character devices。这些权限
是当前多节点 NCCL 路径的一部分，也扩大了容器边界。只运行可信镜像，并通过完整 image
ID 固定验收版本。

模型缓存只读挂载。脚本不会删除模型、Hugging Face cache 或镜像；`make down` 仅停止并
删除以 `CONTAINER_PREFIX` 命名的两个服务容器。

## 发布前检查

```bash
make audit
git status --short
git diff --cached
```

不要提交：

- `.env`；
- Hugging Face/API token；
- 私钥；
- 私有 DNS/tailnet 名称；
- 模型权重、日志或生成内容；
- 未经授权的镜像层或第三方代码。
