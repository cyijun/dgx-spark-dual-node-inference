# RoCEv2 与 NCCL 网络实战

## 把“网卡”和“HCA”分清楚

同一条 RoCE 链路有两个名称：

- Linux 网络接口，例如 `enp1s0f1np1`，供 IP、Gloo 和 NCCL socket bootstrap 使用。
- RDMA HCA，例如 `rocep1s0f1`，供 NCCL IB transport 使用。

两者都区分大小写。不要根据看起来相似的名称推测，应该同时检查：

```bash
ip -br addr
rdma link show
ls /sys/class/infiniband
```

## 为什么不能写死 GID index

RoCEv2 的 GID table 会受启动顺序、链路事件和地址配置影响。即使两台机器本次都使用
index 3，也不能保证重启后仍相同。

仓库的 `detect-rocev2-gid.sh` 会逐项匹配：

1. GID type 是 `RoCE v2`；
2. ndev 等于配置的 fabric NIC；
3. GID 是 IPv4-mapped 形式。

每个节点独立探测，然后分别注入 `NCCL_IB_GID_INDEX`。

## 建议的单 rail 起步配置

```dotenv
HEAD_IP=10.100.32.1
WORKER_IP=10.100.32.2
FABRIC_NIC=enp1s0f1np1
FABRIC_HCA=rocep1s0f1
```

对应运行时环境：

```text
VLLM_HOST_IP=<本节点 fabric IP>
GLOO_SOCKET_IFNAME=<fabric NIC>
NCCL_SOCKET_IFNAME=<fabric NIC>
NCCL_IB_HCA=<fabric HCA>
NCCL_IB_GID_INDEX=<本节点动态探测值>
NCCL_IB_DISABLE=0
NCCL_NET=IB
```

先把一条 rail 验证正确，再考虑 multi-rail。未经基准验证就同时开放多个 HCA，可能让
NCCL 选择与预期不同，排障也更困难。

## 验证 NCCL 确实走 RoCE

启动阶段保留 `NCCL_DEBUG=INFO`，日志应出现类似：

```text
NCCL INFO NET/IB : Using .../RoCE
NCCL INFO Using network IB
```

如果出现 socket transport、管理网 IP 或 Docker bridge 地址，先停止服务并修正绑定，
不要在错误数据路径上继续做性能调优。

## 常用只读检查

```bash
cat /sys/class/infiniband/<HCA>/ports/1/state
cat /sys/class/infiniband/<HCA>/ports/1/phys_state
cat /sys/class/infiniband/<HCA>/ports/1/rate
```

预期分别是 `ACTIVE`、`LINK_UP` 和对应链路速率。
