#!/usr/bin/env python
"""Standalone repro: CoreML miscompiles transpose + identity slice_by_index.

Observed on macOS 26.5 (coremlc 3520.5) via onnxruntime 1.27's CoreML EP
(MLProgram, any compute-unit setting including CPUOnly — this is a compiler
bug, not a precision issue).

A 2-op ONNX model — Transpose(perm=[0,2,1]) then a FULL-RANGE identity
Slice — produces garbage under CoreML: the output is the transposed stream
with (1024 mod 14) = 2 elements skipped per row, i.e. the fused/eliminated
kernel conflates the two dimensions. maxDelta vs CPU ~ 147 on N(0,25) input.

Make the slice non-identity (ends=13) and the same model is exact (2e-7).
Remove the slice and the transpose alone is exact. The generated MIL program
(dumped via the EP's ModelCacheDirectory) is semantically CORRECT — the
corruption happens during CoreML's own graph optimization/execution, so this
should be reported to Apple (and worked around in exports by never emitting
no-op slices).

Usage:  python scripts/coreml_transpose_slice_repro.py
Expect: "identity slice: maxDelta=1.5e+02  BUG" and
        "ends=13 slice : maxDelta~1e-07   ok"
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def make_model(end: int, fname: str) -> None:
    inits = [
        numpy_helper.from_array(np.array([0], np.int64), "st"),
        numpy_helper.from_array(np.array([end], np.int64), "en"),
        numpy_helper.from_array(np.array([2], np.int64), "ax"),
        numpy_helper.from_array(np.array([1], np.int64), "sp"),
    ]
    g = helper.make_graph(
        [
            helper.make_node("Transpose", ["x"], ["t"], perm=[0, 2, 1]),
            helper.make_node("Slice", ["t", "st", "en", "ax", "sp"], ["y"]),
        ],
        "repro",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 14, 1024])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1024, end])],
        initializer=inits,
    )
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), fname)


def run(fname: str, providers) -> np.ndarray:
    s = ort.InferenceSession(fname, ort.SessionOptions(), providers=providers)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((1, 14, 1024)) * 25).astype(np.float32)
    return x, np.asarray(s.run(["y"], {"x": x})[0])


COREML = [
    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUOnly"}),
    "CPUExecutionProvider",
]

for end, label in [(14, "identity slice"), (13, "ends=13 slice ")]:
    make_model(end, "repro_ts.onnx")
    x, y_cml = run("repro_ts.onnx", COREML)
    ref = x.transpose(0, 2, 1)[:, :, :end]
    d = np.abs(y_cml - ref).max()
    print(f"{label}: maxDelta={d:.1e}  {'BUG' if d > 1e-3 else 'ok'}")
