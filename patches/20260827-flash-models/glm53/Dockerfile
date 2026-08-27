FROM lmsysorg/sglang@sha256:73f9294b78e38d8cc297bfed16daec8ac192b126a2d1fb9055e259a632c68f00

LABEL org.opencontainers.image.title="GLM-5.3 Flash NVFP4 SGLang SM121 runtime" \
      org.opencontainers.image.description="GB10 TileLang NoPE DSA low-shared-memory tile" \
      org.opencontainers.image.base.digest="sha256:73f9294b78e38d8cc297bfed16daec8ac192b126a2d1fb9055e259a632c68f00" \
      org.opencontainers.image.model="LibertAIDAI/GLM-5.3-Flash-NVFP4" \
      org.opencontainers.image.model.revision="aa28e1f54130286c95fee10d0705c74ce8743734"

COPY sglang-tilelang-dsa-gb10.patch /tmp/sglang-tilelang-dsa-gb10.patch
RUN patch --batch --forward -p1 -d /sgl-workspace/sglang \
      < /tmp/sglang-tilelang-dsa-gb10.patch \
    && rm /tmp/sglang-tilelang-dsa-gb10.patch

COPY verify-gb10-runtime.py /tmp/verify-gb10-runtime.py
RUN python3 /tmp/verify-gb10-runtime.py \
    && rm /tmp/verify-gb10-runtime.py
