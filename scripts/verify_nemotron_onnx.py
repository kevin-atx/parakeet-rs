"""Check an exported Nemotron diarization step graph against PyTorch over a
whole recording.

Runs the same streaming loop twice — once with the PyTorch model, once with
ONNX Runtime feeding its outputs into the SAME Python speaker cache — so the
graph is validated on real audio and real cache states, independently of the
Rust cache port. Reports the worst per-step logit difference, the share of
80 ms frames whose p > 0.5 decision differs, and writes the ONNX run's 80 ms
probabilities next to the input for scoring.

    python verify_nemotron_onnx.py MODEL.onnx AUDIO.f32 OUT_PROBS.f32
        [--chunk-len 340 --right-context 40 --fifo-len 40 --update-period 300]
"""
import argparse

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
    Nemotron3DiarizationSpeakerCache,
)

ap = argparse.ArgumentParser()
ap.add_argument("onnx_path")
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--model", default="nvidia/nemotron-3-diarization")
ap.add_argument("--chunk-len", type=int, default=340)
ap.add_argument("--right-context", type=int, default=40)
ap.add_argument("--fifo-len", type=int, default=40)
ap.add_argument("--update-period", type=int, default=300)
a = ap.parse_args()

torch.set_num_threads(4)
processor = AutoProcessor.from_pretrained(a.model)
model = AutoModelForAudioFrameClassification.from_pretrained(a.model, attn_implementation="eager").eval()
audio = np.fromfile(a.audio, dtype=np.float32)
feats = processor(audio, sampling_rate=16000).input_features
E = feats.shape[1] // 8
feats = feats[:, : E * 8]
sess = ort.InferenceSession(a.onnx_path, providers=["CPUExecutionProvider"])
pool = lambda lg: torch.as_tensor(lg).sigmoid()[0].numpy().reshape(-1, 8, 8).mean(axis=1)


def run(use_onnx):
    cache = Nemotron3DiarizationSpeakerCache(model.config.streaming_config, fifo_length=a.fifo_len,
                                             speaker_cache_update_period=a.update_period)
    probs, logits_all, s = np.zeros((E, 8), np.float32), [], 0
    with torch.inference_mode():
        while s < E:
            e = min(s + a.chunk_len, E)
            la = min(a.right_context, E - e)
            x = feats[:, s * 8: (e + la) * 8]
            if not use_onnx:
                out = model(input_features=x, speaker_cache=cache, num_lookahead_frames=la or None)
                chunk = out.logits.numpy()
            else:
                cached = cache.get_embeds(model.model.audio_tower.embedder(x))
                lg, emb = sess.run(None, {"chunk_features": x.numpy(), "cached_embeds": cached.numpy()})
                lg_t = torch.as_tensor(lg)
                cache.update(torch.cat([cached, torch.as_tensor(emb)], dim=1), lg_t,
                             model.silence_embeds, e - s)
                c = cached.shape[1]
                chunk = lg[:, c * 8: (c + e - s) * 8]
            logits_all.append(chunk)
            probs[s:e] = pool(chunk)
            s = e
    return probs, logits_all


ref_p, ref_l = run(False)
onx_p, onx_l = run(True)
per_step = [np.abs(r - o).max() for r, o in zip(ref_l, onx_l)]
worst = max(per_step)
first = next((i for i, d in enumerate(per_step) if d > 1e-3), None)
dp = np.abs(ref_p - onx_p)
flips = ((ref_p > 0.5) != (onx_p > 0.5)).any(axis=1).mean()
print(f"per-step max |Δlogit|: first step above 1e-3 = {first} (of {len(per_step)}); "
      f"steps 0-4: {', '.join(f'{d:.1e}' for d in per_step[:5])}; "
      f"|Δp| median {np.median(dp):.1e}, 99.9th pct {np.quantile(dp, 0.999):.1e}")
onx_p.astype(np.float32).tofile(a.out)
print(f"{len(ref_l)} steps over {E * 0.08 / 60:.1f} min: worst |Δlogit| {worst:.2e}, "
      f"frames with a different p>0.5 decision {flips:.4%}, max |Δp| {np.abs(ref_p - onx_p).max():.2e}")
