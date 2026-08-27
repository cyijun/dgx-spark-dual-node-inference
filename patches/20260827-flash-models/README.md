# 2026-08-27 Flash 模型 SM121 补丁

这里保留本轮实际验证过的两个最小派生镜像配方：

- `glm53/`：在 SGLang TileLang NoPE DSA 上把 GB10 不可启动的默认 tile
  `64/2/256` 降为验证过的 `32/1/128`；
- `qwen38/`：仅在 GB10 `(SM121)` 用 SGLang reference QSA 避开 CuTe packed-varlen 错误。

`qwen38/qsa-graph-safe-reference.patch` 是失败实验留档，不由 Dockerfile 应用。它把动态索引
改成固定 shape masked gather 后能完成 CUDA Graph capture，但回放时真实生成语义损坏，
不能作为可用优化。

`glm53/flashinfer-glm53-tp2.patch` 是早期 vLLM 调查留档，不再由 Dockerfile 应用：它确实
补齐了 TP=2 的 H=32 sparse MLA dispatch，也能进入 SM120 kernel，但 vLLM 的 GLM5-Next
NoPE attention 语义仍会产生错误输出。当前可用路线是同目录的 SGLang TileLang 补丁。

构建前必须确认父镜像 digest/image ID。两个节点应从同一构建产物分发，或分别验证 patch
marker 和镜像内源码；不要只比较可移动 tag。

这些是特定模型 revision、运行时 source 和 SM121 的兼容补丁。上游实现升级后应重新验证，
不应无条件长期保留。
