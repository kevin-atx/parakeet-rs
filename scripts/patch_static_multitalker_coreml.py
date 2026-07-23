#!/usr/bin/env python
"""Remove the identity Slice from the static multitalker encoder export.

Why this exists (2026-07-22, macOS 26.5 / coremlc 3520.5):

Apple's CoreML compiler miscompiles `transpose` followed by a FULL-RANGE
(no-op) `slice_by_index`: the fused kernel drops (inner_dim mod outer_dim)
elements per row — for the encoder's [1,14,1024] -> transpose -> slice tail
that's 2 elements skipped per row (1024 mod 14 = 2), silently decorrelating
the `encoded` output while every chunk still returns Ok. Under ORT's CoreML
EP the whole model then "executes" perfectly and transcribes NOTHING.

The trigger in encoder.int8.onnx is a single node: `/Slice_8`, an identity
slice (starts=0, ends=14, axes=2, steps=1 on a length-14 axis) between
`/Transpose_4` and the `encoded` graph output. On a static-shape graph it is
dead weight; removing it restores numerically-correct CoreML execution
(verified: identical 74-word transcript CPU vs CoreML on real speech, and
per-chunk encoder maxDelta drops 0.40 -> 0.03).

A non-identity slice (e.g. ends=13) compiles CORRECTLY, as does
LN+transpose without the slice — the bug needs the eliminable no-op.
See scripts/coreml_transpose_slice_repro.py for the standalone 2-op repro
suitable for an Apple / onnxruntime bug report.

Usage:
  python scripts/patch_static_multitalker_coreml.py <encoder.int8.onnx> <out.onnx>
"""
import sys

import onnx


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    g = m.graph

    sl = next((n for n in g.node if n.name == "/Slice_8"), None)
    if sl is None:
        print("no /Slice_8 node found — model already patched?")
        sys.exit(1)
    tr = next(n for n in g.node if n.output[0] == sl.input[0])
    assert tr.op_type == "Transpose", f"expected Transpose feeding /Slice_8, got {tr.op_type}"
    assert sl.output[0] == "encoded", f"expected /Slice_8 -> encoded, got {sl.output[0]}"

    # Rewire: the Transpose now directly produces the graph output.
    tr.output[0] = "encoded"
    g.node.remove(sl)

    # Drop the slice's now-orphaned Constant feeders.
    used = {i for n in g.node for i in n.input}
    for cn in [n for n in g.node if n.op_type == "Constant" and n.output[0] not in used]:
        g.node.remove(cn)

    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, dst)
    print(f"patched: removed /Slice_8, {tr.name} -> encoded; saved {dst}")


if __name__ == "__main__":
    main()
