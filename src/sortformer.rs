//! NVIDIA Sortformer v2 Streaming Speaker Diarization
//!
//! This module implements NVIDIA's Sortformer v2 streaming model for speaker diarization.
//!
//! Key features:
//! - Streaming inference with ~10s chunks (124 frames at 80ms each)
//! - FIFO buffer for context management
//! - Smart speaker cache compression (keeps important frames, not just recent)
//! - Silence profile tracking
//! - Post-processing: median filtering, hysteresis thresholding
//! - Supports up to 4 speakers
//!
//! Reference: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
//! Note that, my ONNX export:
//! CHUNK_LEN = 124
//! FIFO_LEN = 124
//! CACHE_LEN = 188
//! FEAT_DIM = 128
//! EMB_DIM = 512
//! Note, my stft code is adapted from: https://librosa.org/doc/main/generated/librosa.stft.html

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ort::session::Session;
use realfft::RealFftPlanner;
use std::f32::consts::PI;
use std::path::Path;

// Model constants
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const SAMPLE_RATE: usize = 16000;

// Streaming constants (defaults, overridden by ONNX metadata if present)
const CHUNK_LEN: usize = 124; // Frames per chunk (~10s at 80ms)
const FIFO_LEN: usize = 124; // FIFO buffer length
const SPKCACHE_LEN: usize = 188; // Speaker cache length
const RIGHT_CONTEXT: usize = 1; // Future frames for lookahead
const SUBSAMPLING: usize = 8; // Audio frames -> model frames
/// Width of the model's per-frame embedding (`chunk_pre_encode_embs`), and of
/// every vector in the speaker cache and the silence profile.
pub const EMB_DIM: usize = 512;
pub const NUM_SPEAKERS: usize = 4; // Model supports 4 speakers
/// Samples of 16kHz audio per model frame — the EXACT quantity, and the one
/// to compute with.
///
/// A frame is `SUBSAMPLING` mel hops, each `HOP_LENGTH` samples, so 1280
/// exactly. Deriving it the other way round, from the float duration, is a
/// live trap: `0.08f32` is really 0.0799999982..., widening to `f64` keeps
/// that error, and `as usize` TRUNCATES — so
/// `(FRAME_DURATION as f64 * 16000.0) as usize` yields **1279**. One sample
/// short per frame is 62.5us of drift each frame, ~2.8 seconds across an
/// hour of audio, silently misaligning frames against any other timeline.
pub const FRAME_SAMPLES: usize = SUBSAMPLING * HOP_LENGTH;

/// Seconds of audio per model frame, DERIVED from [`FRAME_SAMPLES`].
///
/// Convenient for reporting; do not multiply it back out to recover a sample
/// or frame count — use [`FRAME_SAMPLES`], which is exact.
pub const FRAME_DURATION: f32 = FRAME_SAMPLES as f32 / SAMPLE_RATE as f32;

const _: () = assert!(FRAME_SAMPLES == 1280);

/// A cache frame counts towards a slot's profile only when that slot is above
/// this probability and every other slot is below it — i.e. the frame is
/// *exclusively* that speaker's. Overlapped frames describe two people and
/// would blur exactly the distinction a profile exists to measure.
const SLOT_EXCLUSIVE_THRESHOLD: f32 = 0.5;

// Cache compression params (from NeMo)
const SPKCACHE_SIL_FRAMES_PER_SPK: usize = 3;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;
const SIL_THRESHOLD: f32 = 0.2;
const MAX_INDEX: usize = 99999;

/// Post-processing configuration for speaker diarization. (NVIDIA official configs from v2 YAMLs)
///
/// Controls how raw model predictions are converted into speaker segments.
/// NVIDIA provides pre-tuned configs for different datasets (CallHome, DIHARD3, AMI).
///
/// # Parameters
/// - `onset`: Probability threshold to START a speaker segment (higher = more strict)
/// - `offset`: Probability threshold to END a speaker segment (lower = longer segments)
/// - `pad_onset`: Seconds to subtract from segment start times
/// - `pad_offset`: Seconds to add to segment end times
/// - `min_duration_on`: Minimum segment length in seconds (filters short blips)
/// - `min_duration_off`: Minimum gap between segments before merging
/// - `median_window`: Smoothing window size (odd number, higher = smoother)
///
/// # Pre-tuned Configs
/// - `callhome()` - (default)
/// - `dihard3()`
///
/// # Custom Config
/// Use `custom(onset, offset)` to create your own config for fine-tuning.
///
/// See: https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/speaker_tasks/diarization/conf/neural_diarizer
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    pub onset: f32,
    pub offset: f32,
    pub pad_onset: f32,
    pub pad_offset: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    pub median_window: usize,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self::callhome()
    }
}

impl DiarizationConfig {
    /// CallHome dataset config for v2 (default)
    /// From: diar_streaming_sortformer_4spk-v2_callhome-part1.yaml
    pub fn callhome() -> Self {
        Self {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            median_window: 11,
        }
    }

    /// DIHARD3 dataset config for v2
    /// From: diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml
    pub fn dihard3() -> Self {
        Self {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            median_window: 11,
        }
    }

    /// Create a custom config for fine-tuning diarization behavior.
    ///
    /// # Arguments
    /// * `onset` - Probability threshold to start a segment (0.0-1.0, typical: 0.5-0.7)
    /// * `offset` - Probability threshold to end a segment (0.0-1.0, typical: 0.4-0.6)
    ///
    /// # Example
    /// ```rust
    /// use parakeet_rs::sortformer::DiarizationConfig;
    ///
    /// // More sensitive detection (lower thresholds)
    /// let sensitive = DiarizationConfig::custom(0.5, 0.4);
    ///
    /// // Stricter detection (higher thresholds, fewer false positives)
    /// let strict = DiarizationConfig::custom(0.7, 0.6);
    ///
    /// // Full customization
    /// let mut config = DiarizationConfig::custom(0.6, 0.5);
    /// config.min_duration_on = 0.3;  // Ignore segments shorter than 300ms
    /// config.median_window = 15;      // More smoothing
    /// ```
    pub fn custom(onset: f32, offset: f32) -> Self {
        Self {
            onset,
            offset,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.1,
            min_duration_off: 0.1,
            median_window: 11,
        }
    }
}

/// Speaker segment with start/end as sample offsets at 16 kHz, and speaker ID.
///
///
/// ```rust,ignore
/// let secs = seg.start as f64 / 16_000.0;
/// let nanos = seg.start as u64 * 1_000_000_000 / 16_000;
/// ```
#[derive(Debug, Clone)]
pub struct SpeakerSegment {
    /// Start position in samples at 16 kHz
    pub start: u64,
    /// End position in samples at 16 kHz
    pub end: u64,
    pub speaker_id: usize,
}

/// Raw per-frame speaker activity predictions (sigmoid outputs).
/// Used by the multitalker pipeline to derive speaker masks for the ASR encoder.
#[derive(Debug, Clone)]
pub struct RawDiarizationPredictions {
    /// Per-frame speaker activity probabilities, shape [num_frames, NUM_SPEAKERS].
    /// Values in [0.0, 1.0].
    pub predictions: Array2<f32>,
    /// Number of valid frames (may be <= predictions.nrows()).
    pub num_valid_frames: usize,
    /// Per-frame model embeddings, shape [num_frames, EMB_DIM], row-aligned 1:1
    /// with `predictions`. `None` unless
    /// [`set_emit_frame_embeddings(true)`](Sortformer::set_emit_frame_embeddings)
    /// was called — the model computes them either way, but retaining them
    /// costs a copy per stride, so callers opt in.
    ///
    /// These are the model's `chunk_pre_encode_embs` output: the representation
    /// the transformer consumes and the currency of the speaker cache. They are
    /// **pre**-encoder, so unlike a speaker-verification embedding their metric
    /// behaviour under cosine distance is not something the training objective
    /// guarantees. Measure before relying on it.
    pub embeddings: Option<Array2<f32>>,
}

/// Cosine similarity, `0.0` if either vector is degenerate.
fn cosine(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let (na, nb) = (a.dot(&a).sqrt(), b.dot(&b).sqrt());
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        a.dot(&b) / (na * nb)
    }
}

/// Cache rows where `speaker_id` is above `SLOT_EXCLUSIVE_THRESHOLD` and every
/// other slot is below it.
///
/// `cache_rows` clamps the scan: the two arrays are grown and compressed
/// together so they should agree, but a mismatch would index one past the end
/// of the other.
fn slot_member_rows_from(
    preds: ArrayView2<f32>,
    cache_rows: usize,
    speaker_id: usize,
) -> Vec<usize> {
    if speaker_id >= NUM_SPEAKERS {
        return Vec::new();
    }
    let slots = preds.ncols().min(NUM_SPEAKERS);
    if speaker_id >= slots {
        return Vec::new();
    }
    (0..preds.nrows().min(cache_rows))
        .filter(|&t| {
            preds[[t, speaker_id]] >= SLOT_EXCLUSIVE_THRESHOLD
                && (0..slots).all(|s| s == speaker_id || preds[[t, s]] < SLOT_EXCLUSIVE_THRESHOLD)
        })
        .collect()
}

/// The cache frames belonging exclusively to one slot, shape [frames, dim].
fn slot_members_from(
    cache: ArrayView2<f32>,
    preds: ArrayView2<f32>,
    speaker_id: usize,
) -> Array2<f32> {
    let dim = cache.ncols();
    let rows = slot_member_rows_from(preds, cache.nrows(), speaker_id);
    let mut out = Array2::zeros((rows.len(), dim));
    for (i, &t) in rows.iter().enumerate() {
        out.slice_mut(s![i, ..]).assign(&cache.slice(s![t, ..]));
    }
    out
}

/// A slot the model has not used: reported rather than omitted, so callers can
/// index `slot_profiles()` by slot id. `dim` keeps the placeholder centroid the
/// same width as the real ones.
fn empty_profile(speaker_id: usize, dim: usize) -> SlotProfile {
    SlotProfile {
        speaker_id,
        frames: 0,
        centroid: Array1::zeros(dim),
        mean_cosine: 0.0,
        min_cosine: 0.0,
    }
}

/// Profile every slot from a speaker cache and its labels.
///
/// Free function, not a method, so the selection and centroid maths are
/// testable without an ONNX session.
fn slot_profiles_from(cache: ArrayView2<f32>, preds: ArrayView2<f32>) -> Vec<SlotProfile> {
    (0..NUM_SPEAKERS)
        .map(|speaker_id| {
            let members = slot_members_from(cache, preds, speaker_id);
            let frames = members.nrows();
            if frames == 0 {
                return empty_profile(speaker_id, cache.ncols());
            }

            let mut centroid = members.sum_axis(Axis(0));
            centroid /= frames as f32;
            let norm = centroid.dot(&centroid).sqrt();
            if norm > 0.0 {
                centroid /= norm;
            }

            let (mut sum, mut min) = (0.0f32, f32::MAX);
            for row in members.rows() {
                let c = cosine(row, centroid.view());
                sum += c;
                min = min.min(c);
            }

            SlotProfile {
                speaker_id,
                frames,
                centroid,
                mean_cosine: sum / frames as f32,
                min_cosine: min,
            }
        })
        .collect()
}

/// What the speaker cache currently believes one slot sounds like.
///
/// Built from the frames the model itself selected and retained as that
/// slot's exemplars, so it reports the diarizer's own notion of the speaker
/// rather than a re-derivation from the audio.
#[derive(Debug, Clone)]
pub struct SlotProfile {
    pub speaker_id: usize,
    /// Cache frames attributed exclusively to this slot (see
    /// `SLOT_EXCLUSIVE_THRESHOLD`). Zero for a slot the model has not used.
    pub frames: usize,
    /// L2-normalised mean of those frames. All-zero when `frames == 0`.
    pub centroid: Array1<f32>,
    /// Mean cosine of member frames to `centroid` — how tightly the slot's
    /// exemplars agree. `0.0` when `frames == 0`.
    pub mean_cosine: f32,
    /// Cosine of the single most deviant member frame.
    ///
    /// Catches a LOPSIDED slot — a few frames of a second voice barely move
    /// `mean_cosine` but are the minimum by construction. On an evenly split
    /// slot the centroid sits equidistant from both voices and this equals
    /// `mean_cosine`, so the two are worth reading together rather than
    /// picking one.
    pub min_cosine: f32,
}

/// Streaming Sortformer v2 speaker diarization engine
/// Saved Sortformer streaming state for save/restore around independent diarization passes.
pub struct SortformerState {
    spkcache: Array3<f32>,
    spkcache_preds: Option<Array3<f32>>,
    fifo: Array3<f32>,
    fifo_preds: Array3<f32>,
    mean_sil_emb: Array2<f32>,
    n_sil_frames: usize,
    audio_buffer: Vec<f32>,
    elapsed_samples: usize,
}

pub struct Sortformer {
    session: Session,
    config: DiarizationConfig,
    // Streaming constants (read from ONNX metadata, fallback to defaults)
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub spkcache_len: usize,
    pub right_context: usize,
    // Streaming state. note that, Same way as Nemo
    spkcache: Array3<f32>,               // (1, 0..spkcache_len, EMB_DIM)
    spkcache_preds: Option<Array3<f32>>, // (1, 0..spkcache_len, NUM_SPEAKERS)
    fifo: Array3<f32>,                   // (1, 0..fifo_len, EMB_DIM)
    fifo_preds: Array3<f32>,             // (1, 0..fifo_len, NUM_SPEAKERS)
    mean_sil_emb: Array2<f32>,           // (1, EMB_DIM)
    n_sil_frames: usize,
    // Buffered streaming state (used by feed/flush)
    audio_buffer: Vec<f32>,
    elapsed_samples: usize,
    // Mel filterbank (cached)
    mel_basis: Array2<f32>,
    /// Retain `chunk_pre_encode_embs` and hand it back on the raw-prediction
    /// paths. Configuration, not state: deliberately outside `SortformerState`
    /// so a save/restore around an independent pass cannot silently turn it
    /// off.
    emit_embeddings: bool,
}

impl Sortformer {
    /// a new Sortformer instance from ONNX model path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_config(model_path, None, DiarizationConfig::default())
    }

    /// Create with custom config
    pub fn with_config<P: AsRef<Path>>(
        model_path: P,
        execution_config: Option<ModelConfig>,
        config: DiarizationConfig,
    ) -> Result<Self> {
        let config_to_use = execution_config.unwrap_or_default();
        let session = config_to_use.build_session(model_path.as_ref())?;

        // Read streaming constants from ONNX metadata (fallback to defaults)
        let (chunk_len, fifo_len, spkcache_len, right_context) =
            if let Ok(metadata) = session.metadata() {
                let c = metadata
                    .custom("chunk_len")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(CHUNK_LEN);
                let f = metadata
                    .custom("fifo_len")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(FIFO_LEN);
                let s = metadata
                    .custom("spkcache_len")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(SPKCACHE_LEN);
                let r = metadata
                    .custom("right_context")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(RIGHT_CONTEXT);
                (c, f, s, r)
            } else {
                (CHUNK_LEN, FIFO_LEN, SPKCACHE_LEN, RIGHT_CONTEXT)
            };

        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        let mut instance = Self {
            session,
            config,
            chunk_len,
            fifo_len,
            spkcache_len,
            right_context,
            spkcache: Array3::zeros((1, 0, EMB_DIM)),
            spkcache_preds: None,
            fifo: Array3::zeros((1, 0, EMB_DIM)),
            fifo_preds: Array3::zeros((1, 0, NUM_SPEAKERS)),
            mean_sil_emb: Array2::zeros((1, EMB_DIM)),
            n_sil_frames: 0,
            audio_buffer: Vec::new(),
            elapsed_samples: 0,
            mel_basis,
            emit_embeddings: false,
        };
        instance.reset_state();
        Ok(instance)
    }

    /// Retain the model's per-frame embeddings and return them on the
    /// raw-prediction paths ([`feed_raw`](Self::feed_raw),
    /// [`peek_buffered_raw`](Self::peek_buffered_raw),
    /// [`diarize_chunk_raw`](Self::diarize_chunk_raw)).
    ///
    /// Off by default. The embeddings are computed on every inference either
    /// way — this only controls whether a copy is kept, which costs
    /// `EMB_DIM * 4` bytes per 80ms frame (~25 KB per second of audio).
    /// Consume them per call; accumulating a whole conversation is ~90 MB/hour.
    ///
    /// The binarising paths ([`diarize`](Self::diarize), [`feed`](Self::feed),
    /// [`flush`](Self::flush)) return segments and are unaffected.
    pub fn set_emit_frame_embeddings(&mut self, on: bool) {
        self.emit_embeddings = on;
    }

    /// Whether per-frame embeddings are being retained.
    pub fn emits_frame_embeddings(&self) -> bool {
        self.emit_embeddings
    }

    /// The speaker cache's exemplar frames for one slot, shape
    /// [frames, EMB_DIM].
    ///
    /// These are the frames the model selected and kept as this slot's
    /// identity — not a re-derivation from audio, and not filtered by anything
    /// downstream. Empty for an unused slot or before the cache first fills.
    ///
    /// Unlike [`set_emit_frame_embeddings`](Self::set_emit_frame_embeddings)
    /// this needs no opt-in: the cache is maintained regardless, so reading it
    /// costs only the copy.
    pub fn slot_frame_embeddings(&self, speaker_id: usize) -> Array2<f32> {
        match &self.spkcache_preds {
            Some(p) => slot_members_from(
                self.spkcache.slice(s![0, .., ..]),
                p.slice(s![0, .., ..]),
                speaker_id,
            ),
            None => Array2::zeros((0, EMB_DIM)),
        }
    }

    /// Per-slot summary of the speaker cache: what each slot sounds like and
    /// how tightly its exemplars agree.
    ///
    /// Always returns `NUM_SPEAKERS` entries, including unused slots
    /// (`frames == 0`), so a caller can index by slot id without a lookup.
    ///
    /// `mean_cosine` and `min_cosine` are offered as a contamination signal —
    /// a slot holding two people should show a looser cluster than one holding
    /// one. That is a hypothesis about a **pre**-encoder representation, not a
    /// property the model was trained to have; calibrate it against ground
    /// truth before gating anything on it.
    pub fn slot_profiles(&self) -> Vec<SlotProfile> {
        match &self.spkcache_preds {
            Some(p) => slot_profiles_from(
                self.spkcache.slice(s![0, .., ..]),
                p.slice(s![0, .., ..]),
            ),
            None => (0..NUM_SPEAKERS).map(|s| empty_profile(s, EMB_DIM)).collect(),
        }
    }

    /// The running mean of frames the model judged to be silence, or `None`
    /// before any have been seen. Useful as the reference "no one is talking"
    /// point in the same space as the slot centroids.
    pub fn silence_profile(&self) -> Option<Array1<f32>> {
        (self.n_sil_frames > 0).then(|| self.mean_sil_emb.slice(s![0, ..]).to_owned())
    }


    /// Streaming latency in seconds: (chunk_len + right_context) * 80ms.
    /// eg. chunk_len=124, right_context=1 -> 10.0s
    pub fn latency(&self) -> f32 {
        (self.chunk_len + self.right_context) as f32 * FRAME_DURATION
    }

    /// Reset streaming state
    pub fn reset_state(&mut self) {
        self.spkcache = Array3::zeros((1, 0, EMB_DIM));
        self.spkcache_preds = None;
        self.fifo = Array3::zeros((1, 0, EMB_DIM));
        self.fifo_preds = Array3::zeros((1, 0, NUM_SPEAKERS));
        self.mean_sil_emb = Array2::zeros((1, EMB_DIM));
        self.n_sil_frames = 0;
        self.audio_buffer.clear();
        self.elapsed_samples = 0;
    }

    /// Save all streaming state so it can be restored after an independent diarization pass.
    pub fn save_state(&self) -> SortformerState {
        SortformerState {
            spkcache: self.spkcache.clone(),
            spkcache_preds: self.spkcache_preds.clone(),
            fifo: self.fifo.clone(),
            fifo_preds: self.fifo_preds.clone(),
            mean_sil_emb: self.mean_sil_emb.clone(),
            n_sil_frames: self.n_sil_frames,
            audio_buffer: self.audio_buffer.clone(),
            elapsed_samples: self.elapsed_samples,
        }
    }

    /// Restore previously saved streaming state.
    pub fn restore_state(&mut self, state: SortformerState) {
        self.spkcache = state.spkcache;
        self.spkcache_preds = state.spkcache_preds;
        self.fifo = state.fifo;
        self.fifo_preds = state.fifo_preds;
        self.mean_sil_emb = state.mean_sil_emb;
        self.n_sil_frames = state.n_sil_frames;
        self.audio_buffer = state.audio_buffer;
        self.elapsed_samples = state.elapsed_samples;
    }

    /// Main diarization entry point
    pub fn diarize(
        &mut self,
        mut audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Vec<SpeakerSegment>> {
        // Resample if needed
        if sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, sample_rate
            )));
        }

        // Convert to mono
        if channels > 1 {
            audio = audio
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect();
        }

        // Reset state for new audio
        self.reset_state();

        // Extract mel features and run streaming inference
        let features = self.extract_mel_features(&audio)?;
        let (full_preds, _) = self.process_features(&features)?;

        // Apply median filtering
        let filtered_preds = if self.config.median_window > 1 {
            self.median_filter(&full_preds)
        } else {
            full_preds
        };

        // Binarize to segments and clip to audio length
        let n_audio_samples = audio.len() as u64;
        let mut segments = self.binarize(&filtered_preds);
        for seg in &mut segments {
            seg.end = seg.end.min(n_audio_samples);
        }
        segments.retain(|s| s.end > s.start);

        Ok(segments)
    }

    /// Streaming diarization: process one audio chunk without resetting state.
    ///
    /// Unlike `diarize()`, this method preserves internal state (FIFO, speaker cache,
    /// silence profile) across calls, enabling true streaming diarization.
    ///
    /// For full `right_context` benefit, buffer at least
    /// `(chunk_len + right_context) * 80ms` of audio before each call, then stride
    /// by `chunk_len * 80ms`. Shorter buffers still work (padded with zeros) but
    /// the lookahead sees silence instead of real future audio.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio chunk at 16kHz mono (any length, typically 2-30s)
    ///
    /// # Returns
    /// Speaker segments with sample offsets relative to this chunk (starting at 0)
    pub fn diarize_chunk(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>> {
        if audio_16k_mono.is_empty() {
            return Ok(vec![]);
        }

        let features = self.extract_mel_features(audio_16k_mono)?;
        let (full_preds, _) = self.process_features(&features)?;

        let filtered_preds = if self.config.median_window > 1 {
            self.median_filter(&full_preds)
        } else {
            full_preds
        };

        // Clip to audio length in samples
        let n_audio_samples = audio_16k_mono.len() as u64;
        let mut segments = self.binarize(&filtered_preds);
        for seg in &mut segments {
            seg.end = seg.end.min(n_audio_samples);
        }
        segments.retain(|s| s.end > s.start);

        Ok(segments)
    }

    /// Streaming diarization returning raw predictions without post-processing.
    ///
    /// Unlike `diarize_chunk()`, this method returns the raw sigmoid outputs
    /// (per-frame speaker activity probabilities) without median filtering or
    /// binarisation. Used by the multitalker ASR pipeline to derive speaker
    /// masks for the encoder.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio chunk at 16kHz mono (any length, typically 2-30s)
    ///
    /// # Returns
    /// Raw predictions with shape [num_frames, NUM_SPEAKERS], values in [0.0, 1.0]
    pub fn diarize_chunk_raw(
        &mut self,
        audio_16k_mono: &[f32],
    ) -> Result<RawDiarizationPredictions> {
        if audio_16k_mono.is_empty() {
            return Ok(self.empty_raw());
        }

        let features = self.extract_mel_features(audio_16k_mono)?;
        let (full_preds, embeddings) = self.process_features(&features)?;
        let num_valid_frames = full_preds.nrows();

        Ok(RawDiarizationPredictions {
            predictions: full_preds,
            num_valid_frames,
            embeddings,
        })
    }

    /// A no-frames result that still honours the current emit setting, so a
    /// caller that asked for embeddings never has to distinguish "none this
    /// call" from "not enabled".
    fn empty_raw(&self) -> RawDiarizationPredictions {
        RawDiarizationPredictions {
            predictions: Array2::zeros((0, NUM_SPEAKERS)),
            num_valid_frames: 0,
            embeddings: self.emit_embeddings.then(|| Array2::zeros((0, EMB_DIM))),
        }
    }

    /// Feed audio samples for buffered streaming diarization.
    ///
    /// Buffers audio internally and runs inference only when enough data has
    /// accumulated for a full `(chunk_len + right_context)` window. Returns
    /// segments with **absolute** timestamps (accumulated across calls).
    ///
    /// Each successful inference produces `chunk_len * 80ms` worth of predictions
    /// from exactly one `streaming_update` call — no redundant re-chunking.
    ///
    /// # Arguments
    /// * `audio_16k_mono` - Audio samples at 16kHz mono (any length)
    ///
    /// # Returns
    /// Speaker segments from any chunks that were ready, or empty vec if still buffering.
    pub fn feed(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerSegment>> {
        self.audio_buffer.extend_from_slice(audio_16k_mono);

        let mut all_segments = Vec::new();

        for (sample_offset, chunk_preds, _) in self.drain_ready_strides()? {
            // Apply median filtering
            let filtered_preds = if self.config.median_window > 1 {
                self.median_filter(&chunk_preds)
            } else {
                chunk_preds
            };

            // Binarize with absolute sample offset
            let chunk_samples = (self.chunk_len * SUBSAMPLING * HOP_LENGTH) as u64;
            let mut segments = self.binarize(&filtered_preds);
            for seg in &mut segments {
                seg.start += sample_offset;
                seg.end = (seg.end + sample_offset).min(sample_offset + chunk_samples);
            }
            segments.retain(|s| s.end > s.start);
            all_segments.extend(segments);
        }

        Ok(all_segments)
    }

    /// Drain complete `(chunk_len + right_context)` windows from the internal
    /// audio buffer, running one `streaming_update` per window at the model's
    /// native stride. Returns each stride's raw predictions (`chunk_len`
    /// frames) with the absolute sample offset of its first frame, and its
    /// row-aligned embeddings when those are being emitted.
    #[allow(clippy::type_complexity)]
    fn drain_ready_strides(&mut self) -> Result<Vec<(u64, Array2<f32>, Option<Array2<f32>>)>> {
        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let stride_samples = self.chunk_len * SUBSAMPLING * HOP_LENGTH;
        let feed_samples = feed_size * HOP_LENGTH;

        let mut strides = Vec::new();
        while self.audio_buffer.len() >= feed_samples {
            let window = &self.audio_buffer[..feed_samples];
            let features = self.extract_mel_features(window)?;
            // STFT center=True produces feed_size+1 mel frames from feed_samples audio,
            // so we always have enough frames: just slice to feed_size...
            let chunk_feat = features.slice(s![.., ..feed_size, ..]).to_owned();

            let (chunk_preds, chunk_embs) = self.streaming_update(&chunk_feat, feed_size)?;
            strides.push((self.elapsed_samples as u64, chunk_preds, chunk_embs));

            // Advance: stride by chunk_len, keep right_context overlap
            self.audio_buffer.drain(..stride_samples);
            self.elapsed_samples += stride_samples;
        }

        Ok(strides)
    }

    /// Buffered streaming diarization returning raw predictions.
    ///
    /// Raw-prediction counterpart of [`feed`](Self::feed): buffers audio and
    /// runs inference only when a full native-stride window has accumulated,
    /// so — unlike [`diarize_chunk_raw`](Self::diarize_chunk_raw) — short
    /// inputs never pay for a zero-padded full-window inference. Returns the
    /// concatenated raw predictions of any strides that completed (possibly
    /// zero rows while still buffering). Row 0 of the first-ever result is
    /// frame 0 of the stream; results are contiguous across calls.
    ///
    /// Use [`peek_buffered_raw`](Self::peek_buffered_raw) to get provisional
    /// predictions for the not-yet-strided remainder.
    pub fn feed_raw(&mut self, audio_16k_mono: &[f32]) -> Result<RawDiarizationPredictions> {
        self.audio_buffer.extend_from_slice(audio_16k_mono);

        let strides = self.drain_ready_strides()?;
        let n = strides.len();
        let mut preds = Vec::with_capacity(n);
        let mut embs = Vec::with_capacity(n);
        for (_, p, e) in strides {
            preds.push(p);
            embs.extend(e);
        }

        let predictions = Self::concat_predictions(&preds);
        let num_valid_frames = predictions.nrows();
        let embeddings = self
            .emit_embeddings
            .then(|| Self::concat_rows(&embs, EMB_DIM));

        Ok(RawDiarizationPredictions {
            predictions,
            num_valid_frames,
            embeddings,
        })
    }

    /// Provisional raw predictions for audio buffered by [`feed_raw`](Self::feed_raw)
    /// that has not yet completed a native stride.
    ///
    /// Runs inference over the buffered remainder (zero-padded to the stride
    /// window) with the streaming state saved before and restored after, so
    /// the same audio is re-processed authoritatively when its stride
    /// completes on a later `feed_raw` call. Frame 0 of the result is the
    /// first frame after the last completed stride.
    pub fn peek_buffered_raw(&mut self) -> Result<RawDiarizationPredictions> {
        if self.audio_buffer.is_empty() {
            return Ok(self.empty_raw());
        }

        let remainder = self.audio_buffer.clone();
        let saved = self.save_state();
        let result = self.diarize_chunk_raw(&remainder);
        self.restore_state(saved);

        result
    }

    /// Flush remaining buffered audio at end of stream.
    ///
    /// processes any leftover audio in the buffer with zero paddings.
    /// we call this once when the audio stream ends to get final segments.
    pub fn flush(&mut self) -> Result<Vec<SpeakerSegment>> {
        if self.audio_buffer.is_empty() {
            return Ok(vec![]);
        }

        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let remaining = std::mem::take(&mut self.audio_buffer);

        let features = self.extract_mel_features(&remaining)?;
        let total_mel = features.shape()[1];
        let current_len = total_mel.min(feed_size);

        let chunk_feat = if current_len < feed_size {
            let mut padded = Array3::zeros((1, feed_size, N_MELS));
            padded
                .slice_mut(s![.., ..current_len, ..])
                .assign(&features.slice(s![.., ..current_len, ..]));
            padded
        } else {
            features.slice(s![.., ..feed_size, ..]).to_owned()
        };

        let (chunk_preds, _) = self.streaming_update(&chunk_feat, current_len)?;

        let filtered_preds = if self.config.median_window > 1 {
            self.median_filter(&chunk_preds)
        } else {
            chunk_preds
        };

        let sample_offset = self.elapsed_samples as u64;
        let remaining_samples = remaining.len() as u64;
        let mut segments = self.binarize(&filtered_preds);
        for seg in &mut segments {
            seg.start += sample_offset;
            seg.end = (seg.end + sample_offset).min(sample_offset + remaining_samples);
        }
        segments.retain(|s| s.end > s.start);

        self.elapsed_samples += remaining.len();

        Ok(segments)
    }

    /// run streaming inference over mel features, returning concatenated per chunk predictions.
    /// note: this shared by `diarize`, `diarize_chunk`, and `diarize_chunk_raw`.
    fn process_features(
        &mut self,
        features: &Array3<f32>,
    ) -> Result<(Array2<f32>, Option<Array2<f32>>)> {
        let total_frames = features.shape()[1];
        let chunk_stride = self.chunk_len * SUBSAMPLING;
        let feed_size = (self.chunk_len + self.right_context) * SUBSAMPLING;
        let num_chunks = total_frames.div_ceil(chunk_stride);

        let mut all_chunk_preds = Vec::new();
        let mut all_chunk_embs = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_stride;
            let end = (start + feed_size).min(total_frames);
            let current_len = end - start;

            let mut chunk_feat = features.slice(s![.., start..end, ..]).to_owned();

            if current_len < feed_size {
                let mut padded = Array3::zeros((1, feed_size, N_MELS));
                padded
                    .slice_mut(s![.., ..current_len, ..])
                    .assign(&chunk_feat);
                chunk_feat = padded;
            }

            let (chunk_preds, chunk_embs) = self.streaming_update(&chunk_feat, current_len)?;
            all_chunk_preds.push(chunk_preds);
            all_chunk_embs.extend(chunk_embs);
        }

        // Every stride emits embeddings or none does, so a partial collection
        // would mean rows no longer align with predictions — refuse rather
        // than hand back a silently misaligned matrix.
        let embs = (self.emit_embeddings && all_chunk_embs.len() == all_chunk_preds.len())
            .then(|| Self::concat_rows(&all_chunk_embs, EMB_DIM));

        Ok((Self::concat_predictions(&all_chunk_preds), embs))
    }

    /// NeMo's streaming_update with smart cache compression
    fn streaming_update(
        &mut self,
        chunk_feat: &Array3<f32>,
        current_len: usize,
    ) -> Result<(Array2<f32>, Option<Array2<f32>>)> {
        let spkcache_len = self.spkcache.shape()[1];
        let fifo_len = self.fifo.shape()[1];

        // Prepare inputs
        let chunk_lengths = Array1::from_vec(vec![current_len as i64]);
        let spkcache_lengths = Array1::from_vec(vec![spkcache_len as i64]);
        let fifo_lengths = Array1::from_vec(vec![fifo_len as i64]);

        // Use empty arrays as fallbacks when lengths are zero (avoids cloning self fields)
        let empty_3d = Array3::<f32>::zeros((1, 0, EMB_DIM));
        let fifo_ref = if fifo_len > 0 { &self.fifo } else { &empty_3d };
        let spkcache_ref = if spkcache_len > 0 {
            &self.spkcache
        } else {
            &empty_3d
        };

        // Create borrowed tensor views instead of cloning arrays
        let chunk_value = ort::value::TensorRef::<f32>::from_array_view(chunk_feat.view())?;
        let chunk_lengths_value = ort::value::Value::from_array(chunk_lengths)?;
        let spkcache_value = ort::value::TensorRef::<f32>::from_array_view(spkcache_ref.view())?;
        let spkcache_lengths_value = ort::value::Value::from_array(spkcache_lengths)?;
        let fifo_value = ort::value::TensorRef::<f32>::from_array_view(fifo_ref.view())?;
        let fifo_lengths_value = ort::value::Value::from_array(fifo_lengths)?;

        // Run ONNX inference and extract all data in a block to release borrow
        let (preds, new_embs, chunk_len) = {
            let outputs = self.session.run(ort::inputs!(
                "chunk" => chunk_value,
                "chunk_lengths" => chunk_lengths_value,
                "spkcache" => spkcache_value,
                "spkcache_lengths" => spkcache_lengths_value,
                "fifo" => fifo_value,
                "fifo_lengths" => fifo_lengths_value
            ))?;

            // Extract outputs
            let (preds_shape, preds_data) = outputs["spkcache_fifo_chunk_preds"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract preds: {e}")))?;
            let (embs_shape, embs_data) = outputs["chunk_pre_encode_embs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract embs: {e}")))?;

            // Convert to ndarray
            let preds_dims = preds_shape.as_ref();
            let embs_dims = embs_shape.as_ref();

            let preds = Array3::from_shape_vec(
                (
                    preds_dims[0] as usize,
                    preds_dims[1] as usize,
                    preds_dims[2] as usize,
                ),
                preds_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape preds: {e}")))?;

            let new_embs = Array3::from_shape_vec(
                (
                    embs_dims[0] as usize,
                    embs_dims[1] as usize,
                    embs_dims[2] as usize,
                ),
                embs_data.to_vec(),
            )
            .map_err(|e| Error::Model(format!("Failed to reshape embs: {e}")))?;

            // Calculate valid frames
            let valid_frames = current_len.div_ceil(SUBSAMPLING);

            (preds, new_embs, valid_frames)
        };

        // Extract predictions for different parts
        let fifo_preds = if fifo_len > 0 {
            preds
                .slice(s![0, spkcache_len..spkcache_len + fifo_len, ..])
                .to_owned()
        } else {
            Array2::zeros((0, NUM_SPEAKERS))
        };

        // only keep chunk_len predictions/embeddings... right_context frames
        // participaded in attenttion (__providing lookahead__) but are discarded here.
        let keep = self.chunk_len.min(chunk_len);
        let chunk_preds = preds
            .slice(s![
                0,
                spkcache_len + fifo_len..spkcache_len + fifo_len + keep,
                ..
            ])
            .to_owned();
        let chunk_embs = new_embs.slice(s![0, ..keep, ..]).to_owned();

        // The caller's copy, taken before the FIFO consumes the array. Same
        // `keep` rows as `chunk_preds`, so the two stay row-aligned by
        // construction rather than by a later assumption.
        let emitted = self.emit_embeddings.then(|| chunk_embs.clone());

        // Append chunk embeddings to FIFO
        self.fifo = Self::concat_axis1(&self.fifo, &chunk_embs.insert_axis(Axis(0)));

        // Update FIFO predictions
        if fifo_len > 0 {
            let combined = Self::concat_axis1_2d(&fifo_preds, &chunk_preds);
            self.fifo_preds = combined.insert_axis(Axis(0));
        } else {
            self.fifo_preds = chunk_preds.clone().insert_axis(Axis(0));
        }

        let fifo_len_after = self.fifo.shape()[1];

        // Move from FIFO to cache when FIFO exceeds limit
        if fifo_len_after > self.fifo_len {
            let mut pop_out_len = self.chunk_len;
            pop_out_len = pop_out_len.max(chunk_len.saturating_sub(self.fifo_len) + fifo_len);
            pop_out_len = pop_out_len.min(fifo_len_after);

            let pop_out_embs = self.fifo.slice(s![.., ..pop_out_len, ..]).to_owned();
            let pop_out_preds = self.fifo_preds.slice(s![.., ..pop_out_len, ..]).to_owned();

            // Update silence profile
            self.update_silence_profile(&pop_out_embs, &pop_out_preds);

            // Remove from FIFO
            self.fifo = self.fifo.slice(s![.., pop_out_len.., ..]).to_owned();
            self.fifo_preds = self.fifo_preds.slice(s![.., pop_out_len.., ..]).to_owned();

            // Append to cache
            self.spkcache = Self::concat_axis1(&self.spkcache, &pop_out_embs);

            if let Some(ref cache_preds) = self.spkcache_preds {
                self.spkcache_preds = Some(Self::concat_axis1(cache_preds, &pop_out_preds));
            }

            // Smart compression when cache exceeds limit
            if self.spkcache.shape()[1] > self.spkcache_len {
                if self.spkcache_preds.is_none() {
                    // Initialize cache predictions from initial output
                    let initial_cache_preds = preds.slice(s![.., ..spkcache_len, ..]).to_owned();
                    let combined = Self::concat_axis1(&initial_cache_preds, &pop_out_preds);
                    self.spkcache_preds = Some(combined);
                }

                // Use smart compression
                self.compress_spkcache();
            }
        }

        Ok((chunk_preds, emitted))
    }

    /// Update mean silence embedding
    fn update_silence_profile(&mut self, embs: &Array3<f32>, preds: &Array3<f32>) {
        let preds_2d = preds.slice(s![0, .., ..]);

        for t in 0..preds_2d.shape()[0] {
            let sum: f32 = (0..NUM_SPEAKERS).map(|s| preds_2d[[t, s]]).sum();
            if sum < SIL_THRESHOLD {
                // This is a silence frame
                let emb = embs.slice(s![0, t, ..]);

                // Update running mean
                let old_sum: Vec<f32> = self
                    .mean_sil_emb
                    .slice(s![0, ..])
                    .iter()
                    .map(|&x| x * self.n_sil_frames as f32)
                    .collect();

                self.n_sil_frames += 1;

                for i in 0..EMB_DIM {
                    self.mean_sil_emb[[0, i]] = (old_sum[i] + emb[i]) / self.n_sil_frames as f32;
                }
            }
        }
    }

    /// Smart cache compression
    fn compress_spkcache(&mut self) {
        let cache_preds = match &self.spkcache_preds {
            Some(p) => p.clone(),
            None => return,
        };

        let n_frames = self.spkcache.shape()[1];
        let per_spk = self.spkcache_len / NUM_SPEAKERS;
        if per_spk <= SPKCACHE_SIL_FRAMES_PER_SPK {
            // truncate if cache too small for compression
            self.spkcache = self.spkcache.slice(s![.., ..self.spkcache_len, ..]).to_owned();
            if let Some(ref p) = self.spkcache_preds {
                self.spkcache_preds = Some(p.slice(s![.., ..self.spkcache_len, ..]).to_owned());
            }
            return;
        }
        let spkcache_len_per_spk = per_spk - SPKCACHE_SIL_FRAMES_PER_SPK;
        let strong_boost_per_spk = (spkcache_len_per_spk as f32 * STRONG_BOOST_RATE) as usize;
        let weak_boost_per_spk = (spkcache_len_per_spk as f32 * WEAK_BOOST_RATE) as usize;
        let min_pos_scores_per_spk = (spkcache_len_per_spk as f32 * MIN_POS_SCORES_RATE) as usize;

        // Calculate quality scores
        let preds_2d = cache_preds.slice(s![0, .., ..]).to_owned();
        let mut scores = self.get_log_pred_scores(&preds_2d);

        // Disable low scores
        scores = self.disable_low_scores(&preds_2d, scores, min_pos_scores_per_spk);

        // Boost important frames
        scores = self.boost_topk_scores(scores, strong_boost_per_spk, 2.0);
        scores = self.boost_topk_scores(scores, weak_boost_per_spk, 1.0);

        // Add silence frames placeholder
        if SPKCACHE_SIL_FRAMES_PER_SPK > 0 {
            let mut padded = Array2::from_elem(
                (n_frames + SPKCACHE_SIL_FRAMES_PER_SPK, NUM_SPEAKERS),
                f32::NEG_INFINITY,
            );
            padded.slice_mut(s![..n_frames, ..]).assign(&scores);
            for i in n_frames..n_frames + SPKCACHE_SIL_FRAMES_PER_SPK {
                for j in 0..NUM_SPEAKERS {
                    padded[[i, j]] = f32::INFINITY;
                }
            }
            scores = padded;
        }

        // Select top frames
        let (topk_indices, is_disabled) = self.get_topk_indices(&scores, n_frames);

        // Gather embeddings
        let (new_embs, new_preds) = self.gather_spkcache(&topk_indices, &is_disabled);

        self.spkcache = new_embs;
        self.spkcache_preds = Some(new_preds);
    }

    /// Calculate quality scores
    fn get_log_pred_scores(&self, preds: &Array2<f32>) -> Array2<f32> {
        let mut scores = Array2::zeros(preds.dim());

        for t in 0..preds.shape()[0] {
            let mut log_1_probs_sum = 0.0f32;
            for s in 0..NUM_SPEAKERS {
                let p = preds[[t, s]].max(PRED_SCORE_THRESHOLD);
                let log_1_p = (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
                log_1_probs_sum += log_1_p;
            }

            for s in 0..NUM_SPEAKERS {
                let p = preds[[t, s]].max(PRED_SCORE_THRESHOLD);
                let log_p = p.ln();
                let log_1_p = (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
                scores[[t, s]] = log_p - log_1_p + log_1_probs_sum - 0.5f32.ln();
            }
        }

        scores
    }

    /// Disable non-speech and overlapped speech
    fn disable_low_scores(
        &self,
        preds: &Array2<f32>,
        mut scores: Array2<f32>,
        min_pos_scores_per_spk: usize,
    ) -> Array2<f32> {
        // Count positive scores per speaker
        let mut pos_count = [0usize; NUM_SPEAKERS];
        for t in 0..scores.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                if scores[[t, s]] > 0.0 {
                    pos_count[s] += 1;
                }
            }
        }

        for t in 0..preds.shape()[0] {
            for s in 0..NUM_SPEAKERS {
                let is_speech = preds[[t, s]] > 0.5;

                if !is_speech {
                    scores[[t, s]] = f32::NEG_INFINITY;
                } else {
                    let is_pos = scores[[t, s]] > 0.0;
                    if !is_pos && pos_count[s] >= min_pos_scores_per_spk {
                        scores[[t, s]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        scores
    }

    /// Boost top K frames per speaker
    fn boost_topk_scores(
        &self,
        mut scores: Array2<f32>,
        n_boost_per_spk: usize,
        scale_factor: f32,
    ) -> Array2<f32> {
        for s in 0..NUM_SPEAKERS {
            // Get column for this speaker
            let col: Vec<(usize, f32)> = (0..scores.shape()[0])
                .map(|t| (t, scores[[t, s]]))
                .collect();

            // Sort by score descending
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boost top K
            for item in sorted.iter().take(n_boost_per_spk.min(sorted.len())) {
                let t = item.0;
                if scores[[t, s]] != f32::NEG_INFINITY {
                    scores[[t, s]] -= scale_factor * 0.5f32.ln();
                }
            }
        }

        scores
    }

    /// Get indices of top frames
    fn get_topk_indices(
        &self,
        scores: &Array2<f32>,
        n_frames_no_sil: usize,
    ) -> (Vec<usize>, Vec<bool>) {
        let n_frames = scores.shape()[0];

        // Flatten scores as (S, T) then reshape to (S*T,)
        // This means we iterate: speaker 0 all times, then speaker 1 all times, etc.
        // flat_index = speaker * n_frames + time
        let mut flat_scores: Vec<(usize, f32)> = Vec::with_capacity(n_frames * NUM_SPEAKERS);
        for s in 0..NUM_SPEAKERS {
            for t in 0..n_frames {
                let flat_idx = s * n_frames + t;
                flat_scores.push((flat_idx, scores[[t, s]]));
            }
        }

        // Sort by score descending to get top-K
        flat_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top spkcache_len and replace invalid scores with MAX_INDEX
        let mut topk_flat: Vec<usize> = flat_scores
            .iter()
            .take(self.spkcache_len)
            .map(|(idx, score)| {
                if *score == f32::NEG_INFINITY {
                    MAX_INDEX
                } else {
                    *idx
                }
            })
            .collect();

        // Sort flat indices ascending (this puts MAX_INDEX at the end)
        topk_flat.sort();

        // Compute is_disabled and convert to frame indices
        let mut is_disabled = vec![false; self.spkcache_len];
        let mut frame_indices = vec![0usize; self.spkcache_len];

        for (i, &flat_idx) in topk_flat.iter().enumerate() {
            if flat_idx == MAX_INDEX {
                // Invalid entries are disabled
                is_disabled[i] = true;
                frame_indices[i] = 0; // We set disabled to 0
            } else {
                // convert to frame index
                let frame_idx = flat_idx % n_frames;

                // check if frame is beyond valid range
                if frame_idx >= n_frames_no_sil {
                    is_disabled[i] = true;
                    frame_indices[i] = 0; // same as abov: set disabled to 0
                } else {
                    frame_indices[i] = frame_idx;
                }
            }
        }

        (frame_indices, is_disabled)
    }

    /// Gather selected frames
    fn gather_spkcache(
        &self,
        indices: &[usize],
        is_disabled: &[bool],
    ) -> (Array3<f32>, Array3<f32>) {
        let mut new_embs = Array3::zeros((1, self.spkcache_len, EMB_DIM));
        let mut new_preds = Array3::zeros((1, self.spkcache_len, NUM_SPEAKERS));

        let cache_preds = self.spkcache_preds.as_ref().unwrap();

        for (i, (&idx, &disabled)) in indices.iter().zip(is_disabled.iter()).enumerate() {
            if i >= self.spkcache_len {
                break;
            }

            if disabled {
                // Use silence embedding
                new_embs
                    .slice_mut(s![0, i, ..])
                    .assign(&self.mean_sil_emb.slice(s![0, ..]));
                // Predictions stay zero
            } else if idx < self.spkcache.shape()[1] {
                new_embs
                    .slice_mut(s![0, i, ..])
                    .assign(&self.spkcache.slice(s![0, idx, ..]));
                new_preds
                    .slice_mut(s![0, i, ..])
                    .assign(&cache_preds.slice(s![0, idx, ..]));
            }
        }

        (new_embs, new_preds)
    }

    /// Concatenate along axis 1 for 3D arrays
    fn concat_axis1(a: &Array3<f32>, b: &Array3<f32>) -> Array3<f32> {
        if a.shape()[1] == 0 {
            return b.clone();
        }
        if b.shape()[1] == 0 {
            return a.clone();
        }
        ndarray::concatenate(Axis(1), &[a.view(), b.view()]).unwrap()
    }

    /// Concatenate along axis 0 for 2D arrays
    fn concat_axis1_2d(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        if a.shape()[0] == 0 {
            return b.clone();
        }
        if b.shape()[0] == 0 {
            return a.clone();
        }
        ndarray::concatenate(Axis(0), &[a.view(), b.view()]).unwrap()
    }

    /// Concatenate predictions
    fn concat_predictions(preds: &[Array2<f32>]) -> Array2<f32> {
        Self::concat_rows(preds, NUM_SPEAKERS)
    }

    /// Stack row-compatible blocks, with `cols` giving the shape of the empty
    /// result so callers get a correctly-shaped zero-row matrix rather than
    /// having to special-case "nothing yet".
    fn concat_rows(blocks: &[Array2<f32>], cols: usize) -> Array2<f32> {
        if blocks.is_empty() {
            return Array2::zeros((0, cols));
        }
        if blocks.len() == 1 {
            return blocks[0].clone();
        }

        let views: Vec<_> = blocks.iter().map(|p| p.view()).collect();
        ndarray::concatenate(Axis(0), &views).unwrap()
    }

    /// Apply median filter to predictions
    fn median_filter(&self, preds: &Array2<f32>) -> Array2<f32> {
        let window = self.config.median_window;
        let half = window / 2;
        let mut filtered = preds.clone();

        for spk in 0..NUM_SPEAKERS {
            for t in 0..preds.shape()[0] {
                let start = t.saturating_sub(half);
                let end = (t + half + 1).min(preds.shape()[0]);

                let mut values: Vec<f32> = (start..end).map(|i| preds[[i, spk]]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                filtered[[t, spk]] = values[values.len() / 2];
            }
        }

        filtered
    }

    /// Binarize predictions to segments (padding applied during thresholding)
    fn binarize(&self, preds: &Array2<f32>) -> Vec<SpeakerSegment> {
        let mut segments = Vec::new();
        let num_frames = preds.shape()[0];

        // pre cobvert cfg thresh from secs to samples
        let pad_onset_samples = (self.config.pad_onset * SAMPLE_RATE as f32) as u64;
        let pad_offset_samples = (self.config.pad_offset * SAMPLE_RATE as f32) as u64;
        let min_dur_on_samples = (self.config.min_duration_on * SAMPLE_RATE as f32) as u64;
        let min_dur_off_samples = (self.config.min_duration_off * SAMPLE_RATE as f32) as u64;
        let samples_per_frame = FRAME_SAMPLES as u64;

        for spk in 0..NUM_SPEAKERS {
            let mut in_seg = false;
            let mut seg_start = 0;
            let mut temp_segments = Vec::new();

            for t in 0..num_frames {
                let p = preds[[t, spk]];

                if p >= self.config.onset && !in_seg {
                    in_seg = true;
                    seg_start = t;
                } else if p < self.config.offset && in_seg {
                    in_seg = false;

                    let start_s = (seg_start as u64 * samples_per_frame)
                        .saturating_sub(pad_onset_samples);
                    let end_s = t as u64 * samples_per_frame + pad_offset_samples;

                    if end_s - start_s >= min_dur_on_samples {
                        temp_segments.push(SpeakerSegment {
                            start: start_s,
                            end: end_s,
                            speaker_id: spk,
                        });
                    }
                }
            }

            // Handle segment at end
            if in_seg {
                let start_s = (seg_start as u64 * samples_per_frame)
                    .saturating_sub(pad_onset_samples);
                let end_s = num_frames as u64 * samples_per_frame + pad_offset_samples;

                if end_s - start_s >= min_dur_on_samples {
                    temp_segments.push(SpeakerSegment {
                        start: start_s,
                        end: end_s,
                        speaker_id: spk,
                    });
                }
            }

            // Merge close segments (min_duration_off)
            if temp_segments.len() > 1 {
                let mut filtered = vec![temp_segments[0].clone()];
                for seg in temp_segments.into_iter().skip(1) {
                    let last = filtered.last_mut().unwrap();
                    // saturating_sub: overlapping segments (gap<-0) always merge..
                    let gap = seg.start.saturating_sub(last.end);
                    if gap < min_dur_off_samples {
                        last.end = seg.end; // Merge
                    } else {
                        filtered.push(seg);
                    }
                }
                segments.extend(filtered);
            } else {
                segments.extend(temp_segments);
            }
        }

        // Sort by start time
        segments.sort_by_key(|s| s.start);
        segments
    }

    fn hann_window(window_length: usize) -> Vec<f32> {
        // Librosa uses periodic window (fftbins=True): divide by N, not N-1
        (0..window_length)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / window_length as f32).cos())
            .collect()
    }

    fn stft(audio: &[f32]) -> Result<Array2<f32>> {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(N_FFT);

        // Create Hann window of length win_length, then zero-pad to n_fft (centered)
        // This is exactly what librosa does: util.pad_center(fft_window, size=n_fft)
        let hann = Self::hann_window(WIN_LENGTH);
        let win_offset = (N_FFT - WIN_LENGTH) / 2;
        let mut fft_window = vec![0.0f32; N_FFT];
        fft_window[win_offset..(WIN_LENGTH + win_offset)].copy_from_slice(&hann[..WIN_LENGTH]);

        // Pad signal for center=True (like librosa/torch.stft)
        // Padding is n_fft // 2 on each side
        let pad_amount = N_FFT / 2;
        let mut padded_audio = vec![0.0; pad_amount];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(vec![0.0; pad_amount]);

        let num_frames = (padded_audio.len() - N_FFT) / HOP_LENGTH + 1;
        let freq_bins = N_FFT / 2 + 1;
        let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));

        let mut input = vec![0.0f32; N_FFT];
        let mut output = r2c.make_output_vec();
        let mut scratch = r2c.make_scratch_vec();

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            // Extract n_fft samples and multiply by zero-padded window
            for i in 0..N_FFT {
                input[i] = if start + i < padded_audio.len() {
                    padded_audio[start + i] * fft_window[i]
                } else {
                    0.0
                };
            }

            r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
                .map_err(|e| Error::Audio(format!("FFT failed: {e}")))?;

            for k in 0..freq_bins {
                // Power spectrum (magnitude^2) - NeMo uses mag_power=2.0
                spectrogram[[k, frame_idx]] = output[k].norm_sqr();
            }
        }

        Ok(spectrogram)
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        // 1. Add dither (small random noise to prevent log(0))
        // NeMo uses dither=1e-5, but for determinism we skip random noise
        // The log_zero_guard handles zero values

        // 2. Apply preemphasis (NeMo uses preemph=0.97)
        let preemphasized = crate::audio::apply_preemphasis(audio, PREEMPH);

        // 3. STFT
        let spectrogram = Self::stft(&preemphasized)?;

        // 4. Apply mel filterbank (with Slaney normalization)
        let mel_spec = self.mel_basis.dot(&spectrogram);

        // 5. Log with guard value (NeMo uses log_zero_guard_value = 2^-24)
        // NeMo uses normalize='NA' which means NO normalization
        let log_mel_spec = mel_spec.mapv(|x| (x + LOG_ZERO_GUARD).ln());

        // Transpose to (batch, time, features) - NeMo outputs (B, D, T), model expects (B, T, D)
        Ok(log_mel_spec.t().to_owned().insert_axis(Axis(0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1kHz sine at 16kHz sample rate, 1 second
        let audio = sine_wave(1000.0, SAMPLE_RATE, SAMPLE_RATE);
        let spec = Sortformer::stft(&audio).unwrap();

        // Expected bin: 1000 * N_FFT / SAMPLE_RATE = 1000 * 512 / 16000 = 32
        let expected_bin = 32;
        let freq_bins = N_FFT / 2 + 1;
        let num_frames = spec.shape()[1];

        let mut correct_frames = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct_frames += 1;
            }
        }

        let interior_frames = num_frames.saturating_sub(4);
        assert!(
            correct_frames > interior_frames / 2,
            "Expected bin {expected_bin} to dominate, but only {correct_frames}/{interior_frames}"
        );
    }

    #[test]
    fn stft_output_shape_is_correct() {
        let audio = vec![0.0f32; SAMPLE_RATE]; // 1 second
        let spec = Sortformer::stft(&audio).unwrap();

        let freq_bins = N_FFT / 2 + 1;
        assert_eq!(spec.shape()[0], freq_bins);
        assert!(spec.shape()[1] > 0);
    }

    /// One-hot labels: `owner[t]` is the slot active at frame t, `None` for a
    /// frame nobody owns.
    fn labels(owner: &[Option<usize>]) -> Array2<f32> {
        let mut p = Array2::zeros((owner.len(), NUM_SPEAKERS));
        for (t, o) in owner.iter().enumerate() {
            if let Some(s) = o {
                p[[t, *s]] = 1.0;
            }
        }
        p
    }

    fn cache(rows: &[[f32; 2]]) -> Array2<f32> {
        let mut c = Array2::zeros((rows.len(), 2));
        for (t, r) in rows.iter().enumerate() {
            c[[t, 0]] = r[0];
            c[[t, 1]] = r[1];
        }
        c
    }

    #[test]
    fn a_frame_two_slots_claim_belongs_to_neither_profile() {
        // A profile says what ONE speaker sounds like. An overlapped frame
        // describes two, so admitting it blurs exactly the distinction the
        // profile exists to measure.
        let mut preds = labels(&[Some(0), Some(1)]);
        preds[[0, 1]] = 0.9; // frame 0 now claimed by slots 0 AND 1

        assert!(
            slot_member_rows_from(preds.view(), 2, 0).is_empty(),
            "overlapped frame must not join slot 0"
        );
        assert_eq!(slot_member_rows_from(preds.view(), 2, 1), vec![1]);
    }

    #[test]
    fn a_slot_holding_two_voices_reports_a_looser_cluster() {
        // The signal this exists to provide: a slot fed from two distinct
        // directions in embedding space should look less cohesive than one
        // fed from a single direction.
        let split = cache(&[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]);
        let unified = cache(&[[1.0, 0.0], [0.99, 0.14], [1.0, 0.0], [0.99, 0.14]]);
        let preds = labels(&[Some(0); 4]);

        let mixed = &slot_profiles_from(split.view(), preds.view())[0];
        let clean = &slot_profiles_from(unified.view(), preds.view())[0];

        assert_eq!(mixed.frames, 4);
        assert!(
            mixed.mean_cosine < clean.mean_cosine,
            "two-voice slot ({}) should be looser than one-voice slot ({})",
            mixed.mean_cosine,
            clean.mean_cosine
        );
        assert!(mixed.min_cosine < clean.min_cosine);

        // An evenly split slot sits equidistant from both voices, so `min`
        // says nothing `mean` does not. `min` earns its place on LOPSIDED
        // splits — see the next test — not on this one.
        assert!((mixed.min_cosine - mixed.mean_cosine).abs() < 1e-5);
    }

    #[test]
    fn a_lopsided_slot_shows_up_in_min_cosine_before_mean_cosine() {
        // Three frames of one voice and one of another: the interloper barely
        // moves the mean but is the minimum by construction. This is the case
        // a mean-only summary would wave through.
        let lopsided = cache(&[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let p = &slot_profiles_from(lopsided.view(), labels(&[Some(0); 4]).view())[0];

        assert!(
            p.min_cosine < p.mean_cosine - 0.2,
            "min {} should sit well below mean {}",
            p.min_cosine,
            p.mean_cosine
        );
    }

    #[test]
    fn every_slot_is_reported_even_when_the_model_never_used_it() {
        // Callers index by slot id; omitting empty slots would silently
        // shift every slot after the gap.
        let profiles =
            slot_profiles_from(cache(&[[1.0, 0.0]]).view(), labels(&[Some(2)]).view());

        assert_eq!(profiles.len(), NUM_SPEAKERS);
        for (i, p) in profiles.iter().enumerate() {
            assert_eq!(p.speaker_id, i);
        }
        assert_eq!(profiles[2].frames, 1);
        assert_eq!(profiles.iter().filter(|p| p.frames > 0).count(), 1);
        // Placeholder centroids match the real ones' width.
        assert_eq!(profiles[0].centroid.len(), profiles[2].centroid.len());
    }

    #[test]
    fn a_labels_cache_length_mismatch_clamps_instead_of_panicking() {
        // Compression keeps the two arrays in step, so this should not
        // happen — but indexing one past the end of the other is a panic
        // inside a live pipeline, and the honest response is fewer frames.
        let preds = labels(&[Some(0); 8]);
        assert_eq!(slot_member_rows_from(preds.view(), 3, 0), vec![0, 1, 2]);
    }

    #[test]
    fn an_empty_result_still_has_the_embedding_width() {
        // A caller that asked for embeddings should never have to
        // distinguish "none this call" from "not enabled".
        assert_eq!(Sortformer::concat_rows(&[], EMB_DIM).dim(), (0, EMB_DIM));
        assert_eq!(Sortformer::concat_predictions(&[]).ncols(), NUM_SPEAKERS);
    }
}
