#!/usr/bin/env python3
import ast
from pathlib import Path


SOURCE = Path(
    "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/"
    "tilelang_kernel.py"
)

tree = ast.parse(SOURCE.read_text())
function = next(
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "sparse_attention_fwd_kernel_v1"
)
defaults = {
    arg.arg: ast.literal_eval(default)
    for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
    if default is not None
}
expected = {"block_I": 32, "num_stages": 1, "threads": 128}
actual = {name: defaults[name] for name in expected}
if actual != expected:
    raise SystemExit(f"unexpected TileLang DSA defaults: {actual}")

source_text = SOURCE.read_text()
if "sparse_attention_fwd_kernel_v1\n            if tail_dim == 0" not in source_text:
    raise SystemExit("NoPE tail_dim==0 no longer dispatches to kernel_v1")

print(f"verified GB10 NoPE TileLang DSA tile: {actual}")
