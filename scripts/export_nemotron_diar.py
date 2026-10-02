"""Export Nemotron 3 Diarization's streaming STEP to ONNX for parakeet-rs.

One graph, no state:
    chunk_features [1, T, 128]   10 ms log-mel frames of the chunk plus its
                                 look-ahead; T is a multiple of 8
    cached_embeds  [1, C, 512]   speaker cache ++ FIFO (C may be 0)
 -> logits         [1, (C + T/8) * 8, num_speakers]   10 ms speaker logits for
                                 every input frame (cache, FIFO, chunk, look-ahead)
    chunk_embeds   [1, T/8, 512] the chunk's encoder-input embeddings, which
                                 join the FIFO

Everything stateful — FIFO, the arrival-order speaker cache and its
compression — belongs to the caller (src/nemotron_diar.rs), exactly as for
Sortformer v2 (export_diar_sortformer.py). Positions restart at every step,
as in the reference implementation, and are computed inside the graph.

Metadata: model_family, num_speakers, the default streaming geometry, the
cache policy constants, and silence_embeds (512 f32le, hex) — the learned
vector a compressed cache uses for its silence slots.

    python export_nemotron_diar.py OUT.onnx [--model nvidia/nemotron-3-diarization]
        [--chunk-len 340 --right-context 40 --fifo-len 40 --update-period 300]
"""
import argparse

import numpy as np
import onnx
import torch
from transformers import AutoModelForAudioFrameClassification

ap = argparse.ArgumentParser()
ap.add_argument("output_path")
ap.add_argument("--model", default="nvidia/nemotron-3-diarization")
ap.add_argument("--chunk-len", type=int, default=340)
ap.add_argument("--right-context", type=int, default=40)
ap.add_argument("--fifo-len", type=int, default=40)
ap.add_argument("--update-period", type=int, default=300)
args = ap.parse_args()

model = AutoModelForAudioFrameClassification.from_pretrained(args.model, attn_implementation="eager").eval()
cfg = model.config
sc = cfg.streaming_config
SUB = cfg.audio_config.subsampling_factor
N_MELS = cfg.audio_config.num_mel_bins
HIDDEN = cfg.audio_config.hidden_size


class Step(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, chunk_features, cached_embeds):
        chunk_embeds = self.m.model.audio_tower.embedder(chunk_features)
        x = torch.cat([cached_embeds, chunk_embeds], dim=1)
        position_ids = torch.arange(x.shape[1], device=x.device)[None, :]
        hidden = self.m.model(inputs_embeds=x, position_ids=position_ids).last_hidden_state
        return self.m.classifier(hidden), chunk_embeds


step = Step(model).eval()
example = (
    torch.randn(1, (args.chunk_len + args.right_context) * SUB, N_MELS),
    torch.randn(1, sc.speaker_cache_length + args.fifo_len, HIDDEN),
)
with torch.inference_mode():
    ref_logits, ref_embeds = step(*example)

torch.onnx.export(
    step,
    example,
    args.output_path,
    input_names=["chunk_features", "cached_embeds"],
    output_names=["logits", "chunk_embeds"],
    dynamic_axes={
        "chunk_features": {1: "mel_time"},
        "cached_embeds": {1: "cache_time"},
        "logits": {1: "logit_time"},
        "chunk_embeds": {1: "chunk_time"},
    },
    opset_version=17,
    dynamo=False,
)

m = onnx.load(args.output_path)
silence = model.silence_embeds.detach().float().numpy().astype("<f4")
meta = {
    "model_family": "nemotron3_diarization",
    "num_speakers": cfg.head_config.num_speakers,
    "subsampling_factor": SUB,
    "emb_dim": HIDDEN,
    "chunk_len": args.chunk_len,
    "right_context": args.right_context,
    "fifo_len": args.fifo_len,
    "spkcache_update_period": args.update_period,
    "spkcache_len": sc.speaker_cache_length,
    "silence_frames_per_speaker": sc.speaker_cache_silence_frames_per_speaker,
    "prediction_score_threshold": sc.prediction_score_threshold,
    "strong_boost_rate": sc.strong_boost_rate,
    "weak_boost_rate": sc.weak_boost_rate,
    "min_positive_scores_rate": sc.min_positive_scores_rate,
    "latest_frames_score_boost": sc.latest_frames_score_boost,
    "silence_embeds": silence.tobytes().hex(),
}
for k, v in meta.items():
    m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
onnx.save(m, args.output_path)

# Round-trip the random example through ONNX Runtime before declaring success.
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(args.output_path, providers=["CPUExecutionProvider"])
o_logits, o_embeds = sess.run(None, {"chunk_features": example[0].numpy(), "cached_embeds": example[1].numpy()})
print(f"exported {args.output_path}: logits {o_logits.shape}, chunk_embeds {o_embeds.shape}; "
      f"max |Δlogit| vs torch {np.abs(o_logits - ref_logits.numpy()).max():.2e}, "
      f"max |Δembed| {np.abs(o_embeds - ref_embeds.numpy()).max():.2e}")
