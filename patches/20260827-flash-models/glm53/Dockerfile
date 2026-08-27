ARG BASE_IMAGE=ghcr.io/cyijun/glm-5.3-flash-nvfp4-gb10@sha256:4251b561d111d817765ed4097512ce36811deac071a4a7411d20242df5c74a47
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="GLM-5.3 Flash NVFP4 TP2 runtime for GB10" \
      org.opencontainers.image.description="Adds the H=32/top-k=2176 SM121 sparse-MLA specializations required by TP=2" \
      io.glm53-gb10.tp2-attention-heads="32" \
      io.glm53-gb10.tp2-topk-capacity="2176" \
      io.glm53-gb10.parent-digest="sha256:4251b561d111d817765ed4097512ce36811deac071a4a7411d20242df5c74a47"

COPY flashinfer-glm53-tp2.patch /opt/glm53-gb10-overlay/flashinfer-glm53-tp2.patch
COPY verify-tp2-runtime.py /opt/glm53-gb10-overlay/verify-tp2-runtime.py

RUN patch --batch --forward -p0 \
      -d /usr/local/lib/python3.12/dist-packages \
      < /opt/glm53-gb10-overlay/flashinfer-glm53-tp2.patch \
    && rm -rf /root/.cache/flashinfer/0.6.17/121a/cached_ops/sparse_mla_sm120 \
    && FLASHINFER_CUDA_ARCH_LIST=12.1a python3 -c \
      'from flashinfer.jit.mla import gen_sparse_mla_sm120_module; gen_sparse_mla_sm120_module().build()' \
    && install -m 0755 \
      /root/.cache/flashinfer/0.6.17/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so \
      /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so \
    && GLM53_GB10_PATCH_DISABLE=1 python3 /opt/glm53-gb10-overlay/verify-tp2-runtime.py
