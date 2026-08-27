from pathlib import Path

import flashinfer.mla._sparse_mla_sm120 as sparse_sm120


root = Path("/usr/local/lib/python3.12/dist-packages")
decode_source = root / "flashinfer/data/csrc/sparse_mla_sm120_decode_dsv3_2.cu"
prefill_source = root / "flashinfer/data/csrc/sparse_mla_sm120_prefill.cu"
aot_module = root / "flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"

assert (32, 2176) in sparse_sm120._DECODE_DSV3_2_DISPATCH
assert "DSV3_2_DISPATCH(32, 2176)" in decode_source.read_text()
assert "ComputeMode::FP8, 32, 2176, 64" in prefill_source.read_text()
assert aot_module.is_file() and aot_module.stat().st_size > 0
print(f"verified TP2 H=32/top-k=2176 AOT: {aot_module}")
