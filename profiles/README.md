# 部署 profile 清单

[`deployments.yaml`](deployments.yaml) 是历史部署的机器可读索引，目的是回答：

- 跑的是哪个 model revision；
- 用了哪个 image/runtime；
- 是 TP、Ulysses SP 还是单机；
- context、KV、量化和 speculative 参数是什么；
- 留下了哪类验证；
- 哪些 provenance 缺失。

它不是一个通用 orchestrator，也不会由脚本自动启动所有模型。不同案例的 runtime
边界不同：

- Qwen、BigBang 和 DeepSeek 文本模型主要是 vLLM；
- MiniMax H3 使用 vLLM-Omni 和 diffusion executor；
- DeepSeek vLLM 0.27 路径依赖定制 source/payload 与 memory guard；
- 当前根目录 `.env.example` + `scripts/` 只实现 Qwen3.8 的可运行参考模板。

复制某个历史参数前，应先检查当前镜像内 `--help`、模型 config、硬件余量和动态 RoCE GID。
历史 API 端口只属于该次实验；仓库的新部署默认使用 8888。
