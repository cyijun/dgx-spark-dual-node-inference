ARG BASE_IMAGE=lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1
FROM ${BASE_IMAGE}

COPY qsa-sm121-reference.patch /tmp/qsa-sm121-reference.patch
RUN cd /sgl-workspace/sglang \
    && patch -p1 < /tmp/qsa-sm121-reference.patch \
    && rm /tmp/qsa-sm121-reference.patch

LABEL infer-deploy.patch="qwen4-exp-qsa-sm121-reference"
