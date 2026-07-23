#!/usr/bin/env python
"""Produce the CoreML-ready fp16 multitalker encoder from the int8 static export.

Why each stage exists (validated 2026-07-23, macOS 26.5 / M1 Pro / ORT 1.27):

1. DEQUANTIZE — the int8 export uses dynamic quantization
   (DynamicQuantizeLinear + MatMulInteger/ConvInteger), none of which the
   ORT CoreML EP supports: the graph shatters into ~300 partitions and runs
   SLOWER than the CPU EP. Rewriting every cluster to plain MatMul/Conv with
   dequantized fp32 weights removes runtime activation quantization entirely
   (output matches int8-CPU within its own quantization noise, meanDelta
   ~5e-3 on `encoded`).

2. DE-WHERE — the CoreML EP rejects `Where` and every bool-tensor shape op
   (Unsqueeze/Slice/Transpose on bool), so attention masking fragments the
   graph at every layer (75 partitions even after dequant). All 72 float
   Wheres follow two patterns and convert exactly:
       Where(c, -10000, S) -> S + float(c) * -10000   (pre-softmax mask)
       Where(c, 0, X)      -> X * (1 - float(c))      (zero-fill)
   Bit-exact vs the original on CPU; partitions collapse to ~4 (the int64
   shape/mask cluster stays on CPU as a side branch off the compute path).

3. IDENTITY-SLICE REMOVAL — Apple's CoreML compiler miscompiles
   transpose + full-range slice_by_index (see
   scripts/coreml_transpose_slice_repro.py); the one trigger node /Slice_8
   is dead weight on a static graph.

4. FP16 — MLProgram fp16 halves the file (fits the 2 GB protobuf limit that
   the fp32 external-data form trips in ORT's CoreML compile path) and is
   the ANE/GPU-native precision. keep_io_types=True keeps the runtime
   interface fp32 — no loader changes. Includes two converter fix-ups:
   stale Cast `to` attrs and the fp32 constants stage 2 introduced.
   No overflow: encoder activations peak ~800, well inside fp16 range;
   validated NaN-free with maxDelta 7.5e-4 vs fp32 on real speech.

Measured (per 1.12 s chunk, encoder only, M1 Pro): int8 CPU 53 ms ·
fp16 CoreML GPU 44 ms · fp16 CoreML CPUOnly 50 ms · fp16 CoreML ANE 89 ms.
End-to-end transcripts match CPU at 99%+ word level.

Usage:
  python scripts/make_multitalker_coreml_fp16.py <encoder.int8.onnx> <encoder.fp16.onnx>

Requires: onnx, numpy, onnxconverter-common.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def dequantize(g):
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    new_inits, remove, count = [], set(), 0
    for n in list(g.node):
        if n.op_type not in ("MatMulInteger", "ConvInteger"):
            continue
        dql = prod.get(n.input[0])
        assert dql is not None and dql.op_type == "DynamicQuantizeLinear", n.name
        x = dql.input[0]
        wq = numpy_helper.to_array(inits[n.input[1]])
        wzp = numpy_helper.to_array(inits[n.input[3]]) if len(n.input) > 3 else np.uint8(0)
        (cast,) = [c for c in cons.get(n.output[0], []) if c.op_type == "Cast"]
        (out_mul,) = [c for c in cons.get(cast.output[0], []) if c.op_type == "Mul"]
        scales_mul = prod.get(out_mul.input[1]) or prod.get(out_mul.input[0])
        assert scales_mul.op_type == "Mul", n.name
        wscale = numpy_helper.to_array(
            inits[next(t for t in scales_mul.input if t in inits)])
        assert wscale.ndim == 0 and np.asarray(wzp).ndim == 0, \
            f"non-scalar quant params at {n.name}"

        wdq_name = n.input[1] + "_dequant"
        new_inits.append(numpy_helper.from_array(
            (wq.astype(np.float32) - np.float32(wzp)) * np.float32(wscale), wdq_name))

        if n.op_type == "MatMulInteger":
            repl = helper.make_node("MatMul", [x, wdq_name], [out_mul.output[0]],
                                    name=n.name.replace("_quant", "_dequant"))
        else:
            kw = {a.name: (list(a.ints) if len(a.ints) else a.i) for a in n.attribute}
            repl = helper.make_node("Conv", [x, wdq_name], [out_mul.output[0]],
                                    name=n.name.replace("_quant", "_dequant"), **kw)
        g.node.append(repl)
        remove.update(id(v) for v in (n, cast, out_mul, scales_mul))
        count += 1

    kept = [n for n in g.node if id(n) not in remove]
    del g.node[:]
    g.node.extend(kept)
    used = {i for n in g.node for i in n.input}
    for i in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(i)
    g.initializer.extend(new_inits)
    return count


def dewhere(g):
    inits = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def const_val(t):
        if t in inits:
            return numpy_helper.to_array(inits[t])
        n = prod.get(t)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
        return None

    g.initializer.append(numpy_helper.from_array(np.float32(-10000.0), "mask_neg_const"))
    g.initializer.append(numpy_helper.from_array(np.float32(1.0), "mask_one_const"))
    cast_cache, count = {}, 0
    for n in list(g.node):
        if n.op_type != "Where":
            continue
        cond, a, b = n.input
        av = const_val(a)
        if av is None or av.size != 1 or av.dtype != np.float32:
            continue  # int64 shape-machinery Where; stays on CPU harmlessly
        if cond not in cast_cache:
            out = cond + "_f32mask"
            g.node.append(helper.make_node("Cast", [cond], [out],
                                           to=TensorProto.FLOAT,
                                           name=cond + "_f32mask_cast"))
            cast_cache[cond] = out
        m_f = cast_cache[cond]
        base = n.name.replace("/", "_")
        if float(av) == -10000.0:
            g.node.append(helper.make_node("Mul", [m_f, "mask_neg_const"],
                                           [n.output[0] + "_addmask"], name=base + "_maskmul"))
            g.node.append(helper.make_node("Add", [b, n.output[0] + "_addmask"],
                                           [n.output[0]], name=base + "_maskadd"))
        elif float(av) == 0.0:
            g.node.append(helper.make_node("Sub", ["mask_one_const", m_f],
                                           [n.output[0] + "_keep"], name=base + "_maskinv"))
            g.node.append(helper.make_node("Mul", [b, n.output[0] + "_keep"],
                                           [n.output[0]], name=base + "_maskmul"))
        else:
            raise AssertionError(f"unexpected Where const {float(av)} at {n.name}")
        g.node.remove(n)
        count += 1
    return count


def drop_identity_slice(g):
    sl = next((n for n in g.node if n.name == "/Slice_8"), None)
    if sl is None:
        return 0
    tr = next(n for n in g.node if n.output[0] == sl.input[0])
    assert tr.op_type == "Transpose"
    tr.output[0] = sl.output[0]
    g.node.remove(sl)
    return 1


def dead_sweep(g):
    graph_outputs = {o.name for o in g.output}
    while True:
        used = {i for n in g.node for i in n.input} | graph_outputs
        dead = [n for n in g.node if not any(o in used for o in n.output)]
        if not dead:
            return
        for n in dead:
            g.node.remove(n)


def to_fp16(m):
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(m, keep_io_types=True,
                                           disable_shape_infer=True)
    g = m16.graph
    # converter leaves Cast `to` attrs stale and skips constants dewhere added
    vi_type = {vi.name: vi.type.tensor_type.elem_type
               for vi in list(g.value_info) + list(g.output) + list(g.input)}
    for n in g.node:
        if n.op_type == "Cast":
            to_attr = next(a for a in n.attribute if a.name == "to")
            declared = vi_type.get(n.output[0])
            if declared in (TensorProto.FLOAT, TensorProto.FLOAT16) and to_attr.i != declared:
                to_attr.i = declared
            if n.name.endswith("_f32mask_cast") and to_attr.i == TensorProto.FLOAT:
                to_attr.i = TensorProto.FLOAT16
    for i, init in enumerate(g.initializer):
        if init.name in ("mask_neg_const", "mask_one_const"):
            arr = numpy_helper.to_array(init).astype(np.float16)
            g.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
    return m16


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    n_dq = dequantize(m.graph)
    n_wh = dewhere(m.graph)
    n_sl = drop_identity_slice(m.graph)
    dead_sweep(m.graph)
    print(f"dequantized {n_dq} clusters, rewrote {n_wh} Wheres, "
          f"removed {n_sl} identity slice(s)")
    m16 = to_fp16(m)
    onnx.save(m16, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
