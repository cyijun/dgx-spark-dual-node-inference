# 2026-08-27 Flash 模型 SM121 补丁

这里保留本轮实际验证过的两个最小派生镜像配方：

- `glm53/`：为 FlashInfer sparse MLA 增加 TP=2 所需的 H=32/top-k=2176 dispatch 和 AOT；
- `qwen38/`：仅在 GB10 `(SM121)` 用 SGLang reference QSA 避开 CuTe packed-varlen 错误。

构建前必须确认父镜像 digest/image ID。两个节点应从同一构建产物分发，或分别验证 patch
marker 和镜像内源码；不要只比较可移动 tag。

这些是特定模型 revision、运行时 source 和 SM121 的兼容补丁。上游实现升级后应重新验证，
不应无条件长期保留。
