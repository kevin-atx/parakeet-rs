//! NVIDIA Nemotron 3 Diarization — streaming speaker diarization, up to 8 speakers.
//!
//! Same streaming design as Sortformer v2 (`sortformer.rs`): one stateless ONNX
//! *step* per stride (`scripts/export_nemotron_diar.py`), with the FIFO, the
//! arrival-order speaker cache and its compression held here. The cache logic
//! is a port of transformers' `Nemotron3DiarizationSpeakerCache`; the
//! differences from v2's port are the speaker count (from metadata, 8), the
//! cache length (264), ONE silence slot per speaker filled with the model's
//! LEARNED silence embedding (v2 keeps a running mean), and probabilities for
//! an uncompressed cache re-estimated at every step.
//!
//! Output is on the same 80 ms frame axis as v2: the model's 10 ms logits go
//! through `sigmoid` and an 8-to-1 mean — the pooling its own speaker cache
//! uses — so `FRAME_SAMPLES` and every consumer keep their meaning.
//!
//! Mel features are computed CONTINUOUSLY over the stream (global preemphasis,
//! centre padding only at the stream start), so a stride sees exactly the
//! features a whole-recording pass would — not v2's per-window approximation,
//! which re-pads every window's edges.

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use crate::sortformer::{empty_profile, slot_profiles_n, RawDiarizationPredictions, SlotProfile, Sortformer};
use ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView2, Axis};
use ort::session::Session;
use std::path::Path;

const SAMPLE_RATE: usize = 16_000;
const HOP: usize = 160;
const N_FFT: usize = 512;
const N_MELS: usize = 128;
/// 10 ms mel frames per 80 ms encoder frame.
const SUB: usize = 8;
/// Leading context a mel slice carries so its first wanted frame's window
/// (`N_FFT / 2` before its centre) lies inside the slice. Two hops, so slice
/// frame `k` is stream frame `m0 + k - 2`.
const CONTEXT: usize = 2 * HOP;
const HALF_WIN: usize = N_FFT / 2;
/// `ONNX metadata model_family` this module accepts.
pub const MODEL_FAMILY: &str = "nemotron3_diarization";

/// The `model_family` an ONNX diarizer declares, if any (Sortformer v2
/// exports declare none).
pub fn model_family(session: &Session) -> Option<String> {
    session.metadata().ok().and_then(|m| m.custom("model_family"))
}

/// Geometry and cache policy, read from the ONNX metadata the export writes.
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub num_speakers: usize,
    pub emb_dim: usize,
    pub chunk_len: usize,
    pub right_context: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub spkcache_update_period: usize,
    pub silence_frames_per_speaker: usize,
    pub prediction_score_threshold: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_positive_scores_rate: f32,
    pub latest_frames_score_boost: f32,
    pub silence_embeds: Array1<f32>,
}

impl NemotronConfig {
    fn from_session(session: &Session) -> Result<Self> {
        let md = session
            .metadata()
            .map_err(|e| Error::Model(format!("nemotron diarizer: no ONNX metadata: {e}")))?;
        let get = |k: &str| {
            md.custom(k)
                .ok_or_else(|| Error::Model(format!("nemotron diarizer: ONNX metadata lacks '{k}'")))
        };
        let family = get("model_family")?;
        if family != MODEL_FAMILY {
            return Err(Error::Model(format!(
                "nemotron diarizer: model_family is '{family}', expected '{MODEL_FAMILY}'"
            )));
        }
        let usize_of = |k: &str| -> Result<usize> {
            get(k)?.parse().map_err(|e| Error::Model(format!("nemotron diarizer: '{k}': {e}")))
        };
        let f32_of = |k: &str| -> Result<f32> {
            get(k)?.parse().map_err(|e| Error::Model(format!("nemotron diarizer: '{k}': {e}")))
        };
        let emb_dim = usize_of("emb_dim")?;
        let silence = decode_f32_hex(&get("silence_embeds")?)?;
        if silence.len() != emb_dim {
            return Err(Error::Model(format!(
                "nemotron diarizer: silence_embeds has {} values, emb_dim is {emb_dim}",
                silence.len()
            )));
        }
        Ok(Self {
            num_speakers: usize_of("num_speakers")?,
            emb_dim,
            chunk_len: usize_of("chunk_len")?,
            right_context: usize_of("right_context")?,
            fifo_len: usize_of("fifo_len")?,
            spkcache_len: usize_of("spkcache_len")?,
            spkcache_update_period: usize_of("spkcache_update_period")?,
            silence_frames_per_speaker: usize_of("silence_frames_per_speaker")?,
            prediction_score_threshold: f32_of("prediction_score_threshold")?,
            strong_boost_rate: f32_of("strong_boost_rate")?,
            weak_boost_rate: f32_of("weak_boost_rate")?,
            min_positive_scores_rate: f32_of("min_positive_scores_rate")?,
            latest_frames_score_boost: f32_of("latest_frames_score_boost")?,
            silence_embeds: Array1::from(silence),
        })
    }

    /// Cache slots each speaker is budgeted, excluding its silence slots.
    fn budget(&self) -> usize {
        (self.spkcache_len / self.num_speakers).saturating_sub(self.silence_frames_per_speaker)
    }
}

fn decode_f32_hex(hex: &str) -> Result<Vec<f32>> {
    let bad = || Error::Model("nemotron diarizer: silence_embeds is not hex f32le".into());
    if !hex.len().is_multiple_of(8) {
        return Err(bad());
    }
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| bad()))
        .collect::<Result<_>>()?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

/// Everything that changes as audio is fed — saved and restored whole around
/// an independent pass (`diarize_full`, a provisional peek).
#[derive(Clone)]
pub struct NemotronState {
    /// Raw samples from virtual index `raw_vstart`. The virtual axis is the
    /// stream shifted by `CONTEXT`: the stream starts with `CONTEXT` zeros,
    /// which is the centre padding a whole-recording STFT puts before sample 0.
    raw: Vec<f32>,
    raw_vstart: usize,
    /// Mel frames `[mel_start, mel_start + mel.nrows())` of the stream.
    mel: Array2<f32>,
    mel_start: usize,
    /// Encoder frames already emitted by completed strides.
    emitted: usize,
    cache_embeds: Array2<f32>,
    cache_probs: Array2<f32>,
    is_compressed: bool,
    fifo: Array2<f32>,
}

impl NemotronState {
    fn new(cfg: &NemotronConfig) -> Self {
        Self {
            raw: vec![0.0; CONTEXT],
            raw_vstart: 0,
            mel: Array2::zeros((0, N_MELS)),
            mel_start: 0,
            emitted: 0,
            cache_embeds: Array2::zeros((0, cfg.emb_dim)),
            cache_probs: Array2::zeros((0, cfg.num_speakers)),
            is_compressed: false,
            fifo: Array2::zeros((0, cfg.emb_dim)),
        }
    }

    fn mel_end(&self) -> usize {
        self.mel_start + self.mel.nrows()
    }
}

/// Streaming Nemotron 3 Diarization engine. API mirrors [`Sortformer`]'s raw
/// paths, which is all the multitalker pipeline uses.
pub struct NemotronDiarizer {
    session: Session,
    pub cfg: NemotronConfig,
    mel_basis: Array2<f32>,
    state: NemotronState,
    emit_embeddings: bool,
}

impl NemotronDiarizer {
    pub fn new<P: AsRef<Path>>(model_path: P, execution_config: Option<ModelConfig>) -> Result<Self> {
        Self::from_session(execution_config.unwrap_or_default().build_session(model_path.as_ref())?)
    }

    /// Build from an already-loaded session — so a caller that opened the
    /// file to read [`model_family`] does not load the weights twice.
    pub fn from_session(session: Session) -> Result<Self> {
        let cfg = NemotronConfig::from_session(&session)?;
        let state = NemotronState::new(&cfg);
        Ok(Self {
            session,
            cfg,
            mel_basis: crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE),
            state,
            emit_embeddings: false,
        })
    }

    pub fn num_speakers(&self) -> usize {
        self.cfg.num_speakers
    }

    /// Streaming latency in seconds: (chunk_len + right_context) × 80 ms.
    pub fn latency(&self) -> f32 {
        (self.cfg.chunk_len + self.cfg.right_context) as f32 * 0.08
    }

    pub fn set_emit_frame_embeddings(&mut self, on: bool) {
        self.emit_embeddings = on;
    }

    pub fn reset_state(&mut self) {
        self.state = NemotronState::new(&self.cfg);
    }

    pub fn save_state(&self) -> NemotronState {
        self.state.clone()
    }

    pub fn restore_state(&mut self, state: NemotronState) {
        self.state = state;
    }

    /// What the speaker cache currently believes each slot sounds like.
    pub fn slot_profiles(&self) -> Vec<SlotProfile> {
        if self.state.cache_embeds.nrows() == 0 {
            return (0..self.cfg.num_speakers).map(|s| empty_profile(s, self.cfg.emb_dim)).collect();
        }
        slot_profiles_n(self.state.cache_embeds.view(), self.state.cache_probs.view(), self.cfg.num_speakers)
    }

    /// Buffer audio and run every stride whose look-ahead has arrived.
    /// Row 0 of the first-ever result is frame 0 of the stream; results are
    /// contiguous across calls.
    pub fn feed_raw(&mut self, audio_16k_mono: &[f32]) -> Result<RawDiarizationPredictions> {
        let Self { session, cfg, mel_basis, state, emit_embeddings } = self;
        state.raw.extend_from_slice(audio_16k_mono);
        compute_mel(mel_basis, state, false)?;
        let mut preds = Vec::new();
        let mut embs = Vec::new();
        while state.mel_end() >= (state.emitted + cfg.chunk_len + cfg.right_context) * SUB {
            let (s, e) = (state.emitted, state.emitted + cfg.chunk_len);
            let (p, emb) = run_stride(session, cfg, state, s, e, cfg.right_context)?;
            preds.push(p);
            if *emit_embeddings {
                embs.push(emb);
            }
            state.emitted = e;
        }
        trim_mel(state);
        Ok(raw_predictions(cfg, preds, emit_embeddings.then_some(embs)))
    }

    /// Provisional predictions for buffered audio no completed stride covers,
    /// on a copy of the stream state (so the same audio is processed
    /// authoritatively later). The tail is closed the way a whole-recording
    /// pass closes the end of a file: zero samples after the last real one,
    /// and an incomplete encoder frame padded as the model's own feature
    /// stacking pads it.
    pub fn peek_buffered_raw(&mut self) -> Result<RawDiarizationPredictions> {
        let Self { session, cfg, mel_basis, state, emit_embeddings } = self;
        let mut st = state.clone();
        compute_mel(mel_basis, &mut st, true)?;
        let end = st.mel_end().div_ceil(SUB);
        if end <= st.emitted {
            return Ok(raw_predictions(cfg, vec![], emit_embeddings.then_some(vec![])));
        }
        pad_mel_to_encoder_frames(&mut st);
        let s = st.emitted;
        let (p, emb) = run_stride(session, cfg, &mut st, s, end, 0)?;
        Ok(raw_predictions(cfg, vec![p], emit_embeddings.then_some(vec![emb])))
    }

    /// Diarize a standalone buffer from the CURRENT speaker cache and FIFO
    /// (the stream's audio buffer is not involved). Mutates the cache like a
    /// stream would; callers wanting the stream untouched save and restore
    /// around it, as `MultitalkerASR::diarize_full` does.
    pub fn diarize_chunk_raw(&mut self, audio_16k_mono: &[f32]) -> Result<RawDiarizationPredictions> {
        let Self { session, cfg, mel_basis, state, emit_embeddings } = self;
        if audio_16k_mono.is_empty() {
            return Ok(raw_predictions(cfg, vec![], emit_embeddings.then_some(vec![])));
        }
        // The buffer's own features (centre-padded at both of its ends, like a
        // whole-recording pass over it) on a scratch state that shares the
        // stream's cache and FIFO; only those are written back, so the
        // stream's audio and mel buffers are untouched.
        let mel = Sortformer::mel_features_with_basis(mel_basis, audio_16k_mono)?;
        let mut tmp = state.clone();
        tmp.mel = mel.slice(s![0, .., ..]).to_owned();
        tmp.mel_start = 0;
        pad_mel_to_encoder_frames(&mut tmp);
        let total = tmp.mel_end() / SUB;
        let (mut preds, mut embs) = (Vec::new(), Vec::new());
        let mut s = 0;
        while s < total {
            let e = (s + cfg.chunk_len).min(total);
            let la = cfg.right_context.min(total - e);
            let (p, emb) = run_stride(session, cfg, &mut tmp, s, e, la)?;
            preds.push(p);
            if *emit_embeddings {
                embs.push(emb);
            }
            s = e;
        }
        state.cache_embeds = tmp.cache_embeds;
        state.cache_probs = tmp.cache_probs;
        state.is_compressed = tmp.is_compressed;
        state.fifo = tmp.fifo;
        Ok(raw_predictions(cfg, preds, emit_embeddings.then_some(embs)))
    }
}

fn raw_predictions(cfg: &NemotronConfig, preds: Vec<Array2<f32>>, embs: Option<Vec<Array2<f32>>>) -> RawDiarizationPredictions {
    let cat = |parts: &[Array2<f32>], width: usize| {
        if parts.is_empty() {
            Array2::zeros((0, width))
        } else {
            concatenate(Axis(0), &parts.iter().map(|p| p.view()).collect::<Vec<_>>())
                .expect("rows share a width")
        }
    };
    let predictions = cat(&preds, cfg.num_speakers);
    RawDiarizationPredictions {
        num_valid_frames: predictions.nrows(),
        predictions,
        embeddings: embs.map(|e| cat(&e, cfg.emb_dim)),
    }
}

/// Compute every stream mel frame whose window has fully arrived (or, with
/// `close_tail`, every frame a whole-recording pass would produce for the
/// audio so far, zero-padding past the last sample).
fn compute_mel(mel_basis: &Array2<f32>, st: &mut NemotronState, close_tail: bool) -> Result<()> {
    if close_tail {
        st.raw.extend(std::iter::repeat_n(0.0, HALF_WIN));
    }
    let total_v = st.raw_vstart + st.raw.len();
    if total_v < CONTEXT + HALF_WIN {
        return Ok(());
    }
    // Frame m is centred at virtual m*HOP + CONTEXT; its window ends HALF_WIN later.
    let last = (total_v - CONTEXT - HALF_WIN) / HOP;
    let first = st.mel_end();
    if last < first {
        return Ok(());
    }
    let lo = first * HOP - st.raw_vstart;
    let hi = last * HOP + CONTEXT + HALF_WIN - st.raw_vstart;
    let frames = Sortformer::mel_features_with_basis(mel_basis, &st.raw[lo..hi])?;
    let wanted = frames.slice(s![0, 2..2 + (last - first + 1), ..]);
    st.mel = concatenate(Axis(0), &[st.mel.view(), wanted])
        .map_err(|e| Error::Model(format!("nemotron diarizer: mel append: {e}")))?;
    // Keep the samples the next frame's slice starts from.
    let keep_from = (last + 1) * HOP;
    if keep_from > st.raw_vstart {
        let drop = (keep_from - st.raw_vstart).min(st.raw.len());
        st.raw.drain(..drop);
        st.raw_vstart += drop;
    }
    Ok(())
}

/// Drop mel frames every completed stride has consumed.
fn trim_mel(st: &mut NemotronState) {
    let consumed = st.emitted * SUB;
    if consumed > st.mel_start {
        let drop = (consumed - st.mel_start).min(st.mel.nrows());
        st.mel = st.mel.slice(s![drop.., ..]).to_owned();
        st.mel_start += drop;
    }
}

/// Zero-pad the mel buffer to whole encoder frames, as the model's feature
/// stacking does for an incomplete final group.
fn pad_mel_to_encoder_frames(st: &mut NemotronState) {
    let short = (SUB - st.mel_end() % SUB) % SUB;
    if short > 0 {
        st.mel = concatenate(Axis(0), &[st.mel.view(), Array2::zeros((short, N_MELS)).view()])
            .expect("mel rows share a width");
    }
}

/// One step over encoder frames `[s, e)` with `la` look-ahead frames: run the
/// graph, update the FIFO and speaker cache, return the chunk's 80 ms
/// probabilities and embeddings.
fn run_stride(
    session: &mut Session,
    cfg: &NemotronConfig,
    st: &mut NemotronState,
    s: usize,
    e: usize,
    la: usize,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let lo = s * SUB - st.mel_start;
    let hi = (e + la) * SUB - st.mel_start;
    let feats = st.mel.slice(s![lo..hi, ..]).to_owned();
    step(session, cfg, st, feats.view(), e - s)
}

fn step(
    session: &mut Session,
    cfg: &NemotronConfig,
    st: &mut NemotronState,
    feats: ArrayView2<f32>,
    n_chunk: usize,
) -> Result<(Array2<f32>, Array2<f32>)> {
    let n_cache = st.cache_embeds.nrows();
    let n_fifo = st.fifo.nrows();
    let c = n_cache + n_fifo;
    let cached = concatenate(Axis(0), &[st.cache_embeds.view(), st.fifo.view()])
        .map_err(|e| Error::Model(format!("nemotron diarizer: cache concat: {e}")))?
        .insert_axis(Axis(0));
    let feats3: Array3<f32> = feats.to_owned().insert_axis(Axis(0));

    let (probs, chunk_embeds_all) = {
        let outputs = session.run(ort::inputs!(
            "chunk_features" => ort::value::TensorRef::<f32>::from_array_view(feats3.view())?,
            "cached_embeds" => ort::value::TensorRef::<f32>::from_array_view(cached.view())?
        ))?;
        let (l_shape, l_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("nemotron diarizer: logits: {e}")))?;
        let (e_shape, e_data) = outputs["chunk_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("nemotron diarizer: chunk_embeds: {e}")))?;
        let (l_dims, e_dims) = (l_shape.as_ref(), e_shape.as_ref());
        let (lt, ls) = (l_dims[1] as usize, l_dims[2] as usize);
        let probs = pool_probs(l_data, lt, ls);
        let embeds = Array2::from_shape_vec((e_dims[1] as usize, e_dims[2] as usize), e_data.to_vec())
            .map_err(|e| Error::Model(format!("nemotron diarizer: embeds shape: {e}")))?;
        (probs, embeds)
    };
    if probs.nrows() < c + n_chunk || chunk_embeds_all.nrows() < n_chunk {
        return Err(Error::Model(format!(
            "nemotron diarizer: step returned {} prob rows / {} embeds for {c} cached + {n_chunk} chunk frames",
            probs.nrows(),
            chunk_embeds_all.nrows()
        )));
    }

    // --- FIFO and speaker-cache update (Nemotron3DiarizationSpeakerCache.update) ---
    let chunk_embeds = chunk_embeds_all.slice(s![..n_chunk, ..]).to_owned();
    let mut fifo_embeds = concatenate(Axis(0), &[st.fifo.view(), chunk_embeds.view()])
        .map_err(|e| Error::Model(format!("nemotron diarizer: fifo concat: {e}")))?;
    let popped = num_popped(cfg, fifo_embeds.nrows());
    if popped > 0 {
        let fifo_probs = probs.slice(s![n_cache..n_cache + fifo_embeds.nrows(), ..]);
        // An uncompressed cache still holds plain chunk frames, whose
        // probabilities this step re-estimates; a compressed one is out of
        // order, so the stored probabilities are the only ones.
        let stored = if st.is_compressed {
            st.cache_probs.clone()
        } else {
            probs.slice(s![..n_cache, ..]).to_owned()
        };
        let mut cache_e = concatenate(Axis(0), &[st.cache_embeds.view(), fifo_embeds.slice(s![..popped, ..])])
            .expect("embeds share a width");
        let mut cache_p = concatenate(Axis(0), &[stored.view(), fifo_probs.slice(s![..popped, ..])])
            .expect("probs share a width");
        fifo_embeds = fifo_embeds.slice(s![popped.., ..]).to_owned();
        if cache_e.nrows() > cfg.spkcache_len {
            (cache_e, cache_p) = compress(cfg, cache_e.view(), cache_p.view());
            st.is_compressed = true;
        }
        st.cache_embeds = cache_e;
        st.cache_probs = cache_p;
    }
    st.fifo = fifo_embeds;
    Ok((probs.slice(s![c..c + n_chunk, ..]).to_owned(), chunk_embeds))
}

/// 10 ms logits `[t, s]` → 80 ms probabilities: sigmoid, then the mean of each
/// group of 8 (the model's `_pool_probs`).
fn pool_probs(logits: &[f32], t10: usize, spk: usize) -> Array2<f32> {
    let frames = t10 / SUB;
    let mut out = Array2::zeros((frames, spk));
    for f in 0..frames {
        for k in 0..SUB {
            let row = (f * SUB + k) * spk;
            for s in 0..spk {
                out[[f, s]] += 1.0 / (1.0 + (-logits[row + s]).exp());
            }
        }
    }
    out / SUB as f32
}

/// No frames move to the speaker cache until the FIFO overflows; then at
/// least `spkcache_update_period` of the oldest move, plus whatever restores
/// the FIFO's capacity.
fn num_popped(cfg: &NemotronConfig, fifo_frames: usize) -> usize {
    if fifo_frames <= cfg.fifo_len {
        return 0;
    }
    cfg.spkcache_update_period.max(fifo_frames - cfg.fifo_len).min(fifo_frames)
}

/// Per-frame, per-speaker importance (`_get_frame_scores`).
fn frame_scores(cfg: &NemotronConfig, probs: ArrayView2<f32>) -> Array2<f32> {
    let (frames, spk) = probs.dim();
    let thr = cfg.prediction_score_threshold;
    let min_pos = (cfg.budget() as f32 * cfg.min_positive_scores_rate).floor() as usize;
    let mut scores = Array2::from_elem((frames, spk), f32::NEG_INFINITY);
    let mut positive = vec![0usize; spk];
    for f in 0..frames {
        let lc_sum: f32 = (0..spk).map(|s| (1.0 - probs[[f, s]]).max(thr).ln()).sum();
        for s in 0..spk {
            let p = probs[[f, s]];
            if p > 0.5 {
                let v = p.max(thr).ln() - (1.0 - p).max(thr).ln() + lc_sum - 0.5f32.ln();
                scores[[f, s]] = v;
                if v > 0.0 {
                    positive[s] += 1;
                }
            }
        }
    }
    for f in 0..frames {
        for s in 0..spk {
            let v = scores[[f, s]];
            if v.is_finite() && v <= 0.0 && positive[s] >= min_pos {
                scores[[f, s]] = f32::NEG_INFINITY;
            }
        }
    }
    scores
}

/// Add `boost` to each speaker's top-`k` frames (`_boost_scores`). Ties go to
/// the earlier frame, so the selection is deterministic.
fn boost_scores(scores: &mut Array2<f32>, k: usize, boost: f32) {
    let (frames, spk) = scores.dim();
    for s in 0..spk {
        let mut order: Vec<usize> = (0..frames).collect();
        order.sort_by(|&a, &b| scores[[b, s]].total_cmp(&scores[[a, s]]).then(a.cmp(&b)));
        for &f in order.iter().take(k.min(frames)) {
            scores[[f, s]] += boost;
        }
    }
}

/// Indices of the frames a compression keeps, in cache order: a frame index
/// `< frames`, or `frames` for the silence slot (`_compress`'s selection).
fn compress_selection(cfg: &NemotronConfig, probs: ArrayView2<f32>) -> Vec<usize> {
    let (frames, spk) = probs.dim();
    let mut scores = frame_scores(cfg, probs);
    for f in cfg.spkcache_len.min(frames)..frames {
        for s in 0..spk {
            scores[[f, s]] += cfg.latest_frames_score_boost;
        }
    }
    let ln2 = std::f32::consts::LN_2;
    let budget = cfg.budget() as f32;
    boost_scores(&mut scores, (budget * cfg.strong_boost_rate).floor() as usize, 2.0 * ln2);
    boost_scores(&mut scores, (budget * cfg.weak_boost_rate).floor() as usize, ln2);

    let scored = frames + cfg.silence_frames_per_speaker;
    let sentinel = scored * spk;
    // Speaker-major flat index: spk * scored + frame; silence rows score +inf.
    let mut flat: Vec<(f32, usize)> = (0..spk)
        .flat_map(|s| (0..scored).map(move |f| (s, f)))
        .map(|(s, f)| (if f < frames { scores[[f, s]] } else { f32::INFINITY }, s * scored + f))
        .collect();
    flat.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut picked: Vec<usize> = flat
        .iter()
        .take(cfg.spkcache_len)
        .map(|&(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i })
        .collect();
    picked.sort_unstable();
    picked
        .into_iter()
        .map(|i| if i == sentinel { frames } else { (i % scored).min(frames) })
        .collect()
}

/// Keep the `spkcache_len` most important frames, grouped by speaker and in
/// their original order within a speaker (`_compress`).
fn compress(cfg: &NemotronConfig, embeds: ArrayView2<f32>, probs: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
    let frames = embeds.nrows();
    let sel = compress_selection(cfg, probs);
    let mut out_e = Array2::zeros((sel.len(), embeds.ncols()));
    let mut out_p = Array2::zeros((sel.len(), probs.ncols()));
    for (row, &f) in sel.iter().enumerate() {
        if f == frames {
            out_e.slice_mut(s![row, ..]).assign(&cfg.silence_embeds);
        } else {
            out_e.slice_mut(s![row, ..]).assign(&embeds.slice(s![f, ..]));
            out_p.slice_mut(s![row, ..]).assign(&probs.slice(s![f, ..]));
        }
    }
    (out_e, out_p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(num_speakers: usize, spkcache_len: usize) -> NemotronConfig {
        NemotronConfig {
            num_speakers,
            emb_dim: 2,
            chunk_len: 340,
            right_context: 40,
            fifo_len: 40,
            spkcache_len,
            spkcache_update_period: 300,
            silence_frames_per_speaker: 1,
            prediction_score_threshold: 0.25,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_positive_scores_rate: 0.5,
            latest_frames_score_boost: 0.05,
            silence_embeds: Array1::from(vec![-1.0, -1.0]),
        }
    }

    #[test]
    fn hex_round_trips_little_endian_f32() {
        let v = [1.5f32, -0.25, 3.0e-7];
        let hex: String = v.iter().flat_map(|x| x.to_le_bytes()).map(|b| format!("{b:02x}")).collect();
        assert_eq!(decode_f32_hex(&hex).unwrap(), v);
        assert!(decode_f32_hex("abc").is_err());
    }

    #[test]
    fn fifo_pops_nothing_until_it_overflows_then_at_least_the_update_period() {
        let c = cfg(8, 264);
        assert_eq!(num_popped(&c, 40), 0);
        assert_eq!(num_popped(&c, 41), 41); // period 300 > frames: everything
        assert_eq!(num_popped(&c, 380), 340.max(300).min(380));
        assert_eq!(num_popped(&c, 1000), 960);
    }

    #[test]
    fn probabilities_pool_eight_ten_ms_frames_into_one() {
        // logit 0 → 0.5 everywhere; one large logit in frame 1's group.
        let mut logits = vec![0.0f32; 16 * 2];
        logits[(8 + 3) * 2] = 100.0;
        let p = pool_probs(&logits, 16, 2);
        assert_eq!(p.dim(), (2, 2));
        assert!((p[[0, 0]] - 0.5).abs() < 1e-6);
        assert!((p[[1, 0]] - (7.0 * 0.5 + 1.0) / 8.0).abs() < 1e-6);
    }

    #[test]
    fn non_speech_frames_never_score() {
        let c = cfg(2, 4);
        // Frame 2's 0.6 is speech but scores below zero, so (as in the
        // reference) it is disabled once a speaker has enough positive frames
        // — with this tiny cache the bar is zero.
        let probs = Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.1, 0.9, 0.4, 0.6]).unwrap();
        let sc = frame_scores(&c, probs.view());
        assert!(sc[[0, 0]].is_finite() && sc[[0, 1]] == f32::NEG_INFINITY);
        assert!(sc[[1, 0]] == f32::NEG_INFINITY && sc[[1, 1]].is_finite());
        assert!(sc[[2, 0]] == f32::NEG_INFINITY && sc[[2, 1]] == f32::NEG_INFINITY);
    }

    #[test]
    fn compression_keeps_capacity_in_frame_order_with_a_silence_slot_per_speaker() {
        // 2 speakers, cache 6: budget 2 per speaker. Speaker 0 talks in frames
        // 0..6, speaker 1 in 6..10, with rising confidence.
        let c = cfg(2, 6);
        let mut probs = Array2::zeros((10, 2));
        for f in 0..6 {
            probs[[f, 0]] = 0.6 + 0.05 * f as f32;
        }
        for f in 6..10 {
            probs[[f, 1]] = 0.6 + 0.05 * (f - 6) as f32;
        }
        let sel = compress_selection(&c, probs.view());
        assert_eq!(sel.len(), 6);
        // Speaker-major, frame order inside a speaker, silence (index 10) last
        // for each speaker's block.
        let s0: Vec<usize> = sel.iter().copied().take_while(|&f| f < 6 || f == 10).collect();
        assert!(s0.windows(2).all(|w| w[0] <= w[1]));
        assert!(sel.contains(&10), "every speaker gets its silence slot: {sel:?}");
        let embeds = Array2::from_shape_fn((10, 2), |(f, _)| f as f32);
        let (e, p) = compress(&c, embeds.view(), probs.view());
        assert_eq!(e.nrows(), 6);
        for (row, &f) in sel.iter().enumerate() {
            if f == 10 {
                assert_eq!(e.row(row).to_vec(), vec![-1.0, -1.0]);
                assert_eq!(p.row(row).sum(), 0.0);
            } else {
                assert_eq!(e[[row, 0]], f as f32);
            }
        }
    }
}
