//! Multi-talker streaming ASR pipeline.
//!
//! Combines Sortformer speaker diarisation with the multitalker encoder
//! (speaker kernel injection) to produce per-speaker transcriptions from
//! mixed audio. Each active speaker gets an independent encoder cache and
//! decoder state.
//!
//! Architecture:
//! ```text
//! Audio -> [Mel] -> [Sortformer raw preds] -> per-speaker masks
//!                   -> [ASR Encoder(mel, cache_k, spk_k, bg_k)] -> [RNNT Decode] -> text_k
//! ```

use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_multitalker::{MultitalkerEncoderCache, MultitalkerModel};
use crate::nemotron::SentencePieceVocab;
use crate::sortformer::{Sortformer, NUM_SPEAKERS};
use crate::timestamps::{self, TimestampMode};
use crate::transcriber::Transcriber;
use ndarray::{s, Array1, Array2, Array3};
use std::path::Path;

// Reuse the same audio constants as Nemotron (same encoder architecture)
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

// Encoder arch (same as Nemotron 0.6B)
const NUM_ENCODER_LAYERS: usize = 24;
const HIDDEN_DIM: usize = 1024;
const LEFT_CONTEXT: usize = 70;
const CONV_CONTEXT: usize = 8;

// Decoder
const VOCAB_SIZE: usize = 1024;
const BLANK_ID: usize = 1024;
const DECODER_LSTM_DIM: usize = 640;
const MAX_SYMBOLS_PER_STEP: usize = 10;

// Pre-encode cache frames (fixed, independent of latency mode)
const PRE_ENCODE_CACHE: usize = 9;

// Each encoded frame spans 8 mel frames at 10ms hop = 80ms
const SECONDS_PER_ENCODED_FRAME: f32 = 0.08;

/// Activity threshold: a speaker is considered active if any frame in the
/// chunk exceeds this probability.
const SPEAKER_ACTIVITY_THRESHOLD: f32 = 0.3;

/// Word-level timestamp for a single word in a speaker's transcript.
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_secs: f32,
    pub end_secs: f32,
    /// Confidence score (min softmax probability across subword tokens). 0.0–1.0.
    pub confidence: f32,
}

/// Per-speaker state for the multi-instance architecture.
struct SpeakerInstance {
    encoder_cache: MultitalkerEncoderCache,
    state_1: Array3<f32>,
    state_2: Array3<f32>,
    last_token: i32,
    /// Each entry is (token_id, absolute_encoder_frame, confidence).
    accumulated_tokens: Vec<(usize, usize, f32)>,
    speaker_id: usize,
}

impl SpeakerInstance {
    fn new(speaker_id: usize) -> Self {
        Self {
            encoder_cache: MultitalkerEncoderCache::new(
                NUM_ENCODER_LAYERS,
                LEFT_CONTEXT,
                HIDDEN_DIM,
                CONV_CONTEXT,
            ),
            state_1: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            state_2: Array3::zeros((2, 1, DECODER_LSTM_DIM)),
            last_token: BLANK_ID as i32,
            accumulated_tokens: Vec::new(),
            speaker_id,
        }
    }
}

/// Per-speaker transcription output.
#[derive(Debug, Clone)]
pub struct SpeakerTranscript {
    pub speaker_id: usize,
    pub text: String,
    pub words: Vec<WordTimestamp>,
}

/// Per-frame model embeddings for a region of the stream.
///
/// Carries its own absolute frame index because it does **not** in general
/// start where `ChunkResult::speaker_activity` starts: activity is emitted per
/// ASR sub-chunk and may include provisional peek frames ahead of the
/// diarizer, while embeddings exist only for frames the diarizer has settled
/// authoritatively. Align on `first_frame`, never on position.
#[derive(Debug, Clone)]
pub struct FrameEmbeddings {
    /// Absolute 80ms-frame index of row 0.
    pub first_frame: usize,
    /// Columns per row; equals [`crate::sortformer::EMB_DIM`].
    pub dim: usize,
    /// Row-major `[frames, dim]`.
    pub data: Vec<f32>,
}

impl FrameEmbeddings {
    pub fn frames(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }

    /// Row `i`, i.e. the embedding of absolute frame `first_frame + i`.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        (i < self.frames()).then(|| &self.data[i * self.dim..(i + 1) * self.dim])
    }

    /// The embedding of an absolute frame index, if this block covers it.
    pub fn frame(&self, absolute_frame: usize) -> Option<&[f32]> {
        self.row(absolute_frame.checked_sub(self.first_frame)?)
    }
}

/// Result of processing one audio chunk, including transcripts and diarization.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// Per-speaker text deltas for this chunk.
    pub transcripts: Vec<SpeakerTranscript>,
    /// Per-frame speaker activity probabilities from Sortformer.
    /// Shape [num_frames, NUM_SPEAKERS], values in [0.0, 1.0].
    /// Frame rate: 80ms. Frame 0 = start of the region processed by this
    /// call (i.e., the first sub-chunk consumed); frames are contiguous
    /// and non-overlapping across sub-chunks.
    pub speaker_activity: Vec<Vec<f32>>,
    /// Absolute 80ms-frame index of `speaker_activity[0]`, so activity from
    /// successive calls lands on one timeline and can be lined up with
    /// `frame_embeddings`.
    pub first_activity_frame: usize,
    /// Duration in seconds of each activity frame (0.08s).
    pub frame_duration_secs: f32,
    /// The model's per-frame embeddings for frames the diarizer settled during
    /// this call. `None` unless
    /// [`set_emit_frame_embeddings(true)`](MultitalkerASR::set_emit_frame_embeddings).
    ///
    /// Covers a *different* frame range than `speaker_activity` — see
    /// [`FrameEmbeddings`].
    pub frame_embeddings: Option<FrameEmbeddings>,
}

/// Streaming latency mode controlling the encoder chunk size.
///
/// The multitalker encoder was trained with multi-latency masking, so it can
/// operate at different chunk sizes at inference time. Smaller chunks give
/// lower latency but reduce accuracy because fewer future frames are available
/// to the attention layers.
///
/// Each mode corresponds to an `att_context_size` configuration in the model:
/// the second value is the number of future encoded frames the first layer
/// group can attend to.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LatencyMode {
    /// `[70, 13]` -- 14 encoded frames, 112 mel frames, 1.12s latency.
    /// Highest accuracy. This is the default.
    #[default]
    Normal,
    /// `[70, 6]` -- 7 encoded frames, 56 mel frames, 0.56s latency.
    Low,
    /// `[70, 1]` -- 2 encoded frames, 16 mel frames, 0.16s latency.
    VeryLow,
    /// `[70, 0]` -- 1 encoded frame, 8 mel frames, 0.08s latency.
    /// Lowest accuracy.
    Ultra,
}

impl LatencyMode {
    /// Number of mel spectrogram frames per encoder chunk.
    pub const fn chunk_mel_frames(self) -> usize {
        match self {
            Self::Normal => 112,  // 14 * 8
            Self::Low => 56,      //  7 * 8
            Self::VeryLow => 16,  //  2 * 8
            Self::Ultra => 8,     //  1 * 8
        }
    }

    /// Number of encoded frames per chunk (after 8x subsampling).
    pub const fn encoded_frames(self) -> usize {
        match self {
            Self::Normal => 14,
            Self::Low => 7,
            Self::VeryLow => 2,
            Self::Ultra => 1,
        }
    }

    /// Approximate latency in seconds.
    pub const fn latency_secs(self) -> f32 {
        match self {
            Self::Normal => 1.12,
            Self::Low => 0.56,
            Self::VeryLow => 0.16,
            Self::Ultra => 0.08,
        }
    }
}

/// Runtime configuration for the multitalker pipeline.
///
/// These settings can be changed between calls to `transcribe_chunk()` via
/// the setter methods on [`MultitalkerASR`]. Changing `latency_mode` requires
/// calling [`MultitalkerASR::reset()`] first (the setter does this automatically).
#[derive(Debug, Clone)]
pub struct MultitalkerConfig {
    /// Maximum number of concurrent speakers to track (1..=4).
    /// The Sortformer model supports up to 4 speaker slots. Setting this
    /// lower reduces compute by skipping inactive slots.
    pub max_speakers: usize,

    /// Minimum speaker activity probability to consider a speaker active
    /// in a given chunk. Higher values require stronger evidence of speech
    /// before creating a speaker instance. Range: 0.0..=1.0.
    pub activity_threshold: f32,

    /// Streaming latency mode. Controls the encoder chunk size and
    /// therefore the latency-accuracy tradeoff.
    pub latency_mode: LatencyMode,

    /// Blank-logit penalty subtracted from the RNN-T blank before the
    /// greedy argmax (sherpa-onnx convention: `logits[blank] -= penalty`).
    /// Positive values recover borderline words where a token narrowly
    /// trails blank — trailing/short words are the usual casualties;
    /// negative values boost blank, suppressing spurious emissions.
    /// Genuine silence frames, where blank leads by a wide margin, are
    /// unaffected. Default 0.0 = off (bit-identical stock decode).
    pub blank_penalty: f32,

    /// Sleep inserted between ready-sub-chunk iterations inside one
    /// `transcribe_chunk()` call (between the ~1.12s encoder inferences,
    /// never before the first). Duty-cycles the encoder so a concurrent
    /// GPU consumer — a video call's encode pipeline — gets scheduling
    /// gaps instead of a solid multi-second inference pulse. Purely a
    /// timing change: the computation sequence and every output are
    /// identical with or without it. Default None = flat-out (unchanged
    /// behavior). A 30s caller in Normal mode does ~26 sub-chunks, so
    /// budget `pause × 25` of added wall time per call.
    pub inter_chunk_pause: Option<std::time::Duration>,

    /// When true, ASR sub-chunks not fully covered by COMPLETED diarizer
    /// strides are held for a later call instead of decoding against the
    /// provisional peek (whose zero-padded partial-stride predictions
    /// differ from the authoritative pass that replaces them). Every word
    /// is then decoded with settled speaker conditioning, at the cost of
    /// up to one stride (~10s) of extra latency for chunk-tail words.
    /// ⚠ With this on, a stream must be finished by feeding
    /// [`MultitalkerASR::settle_flush_subchunks`] sub-chunks of silence,
    /// or the held tail is never decoded. Default false (unchanged
    /// peek-and-decode behavior).
    pub hold_at_settled_frontier: bool,
}

impl Default for MultitalkerConfig {
    fn default() -> Self {
        Self {
            max_speakers: NUM_SPEAKERS,
            activity_threshold: SPEAKER_ACTIVITY_THRESHOLD,
            latency_mode: LatencyMode::default(),
            blank_penalty: 0.0,
            inter_chunk_pause: None,
            hold_at_settled_frontier: false,
        }
    }
}

impl MultitalkerConfig {
    /// The mel-frame chunk size for the current latency mode.
    pub fn chunk_size(&self) -> usize {
        self.latency_mode.chunk_mel_frames()
    }
}

/// Multi-talker streaming ASR combining Sortformer diarisation with
/// speaker-kernel-injected ASR encoding.
pub struct MultitalkerASR {
    model: MultitalkerModel,
    sortformer: Sortformer,
    vocab: SentencePieceVocab,
    speakers: Vec<SpeakerInstance>,
    config: MultitalkerConfig,
    mel_basis: Array2<f32>,
    audio_buffer: Vec<f32>,
    audio_processed: usize,
    chunk_idx: usize,
    /// Authoritative Sortformer predictions from completed native strides.
    /// Row i holds absolute 80ms frame `diar_pred_offset + i`. Consumed rows
    /// are trimmed after each call; ASR sub-chunks ahead of the last
    /// completed stride use a provisional peek instead (see
    /// `transcribe_chunk_inner`).
    diar_preds: Vec<[f32; NUM_SPEAKERS]>,
    /// Absolute 80ms-frame index of `diar_preds[0]`.
    diar_pred_offset: usize,
}

/// Assemble the diarization mask window for one ASR sub-chunk covering
/// absolute 80ms frames `[f_start, f_end)`.
///
/// `authoritative` row i holds absolute frame `auth_offset + i` (completed
/// Sortformer strides); `provisional` row j holds absolute frame
/// `auth_offset + authoritative.len() + j` (a state-safe peek over the
/// not-yet-strided tail). Frames covered by neither source are zero
/// (treated as silence).
fn assemble_diar_window(
    authoritative: &[[f32; NUM_SPEAKERS]],
    auth_offset: usize,
    provisional: Option<&Array2<f32>>,
    f_start: usize,
    f_end: usize,
) -> Array2<f32> {
    let n = f_end.saturating_sub(f_start);
    let covered = auth_offset + authoritative.len();
    let mut window = Array2::zeros((n, NUM_SPEAKERS));

    for i in 0..n {
        let f = f_start + i;
        if f >= auth_offset && f < covered {
            let row = &authoritative[f - auth_offset];
            for s in 0..NUM_SPEAKERS {
                window[[i, s]] = row[s];
            }
        } else if let Some(p) = provisional.filter(|p| f >= covered && f - covered < p.nrows()) {
            let pi = f - covered;
            for s in 0..NUM_SPEAKERS.min(p.ncols()) {
                window[[i, s]] = p[[pi, s]];
            }
        }
        // Frames covered by neither source (pre-trim or beyond the peeked
        // tail) stay zero = silence.
    }

    window
}

impl MultitalkerASR {
    /// Load the multitalker ASR pipeline.
    ///
    /// # Arguments
    /// * `asr_model_dir` - Directory containing encoder.onnx, decoder_joint.onnx, tokenizer.model
    /// * `sortformer_model_path` - Path to Sortformer ONNX model
    /// * `exec_config` - ONNX Runtime execution config (optional)
    pub fn from_pretrained<P: AsRef<Path>, Q: AsRef<Path>>(
        asr_model_dir: P,
        sortformer_model_path: Q,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let asr_dir = asr_model_dir.as_ref();
        let exec = exec_config.unwrap_or_default();

        let vocab = SentencePieceVocab::from_file(asr_dir.join("tokenizer.model"))?;

        let model = MultitalkerModel::from_pretrained(asr_dir, exec.clone())?;

        // CoreML never helps the Sortformer: its streaming state (spkcache/
        // fifo) grows call-to-call, so a static export is rejected at the
        // first inference ("Got: 0 Expected: 188") and a dynamic one cannot
        // be compiled for ANE/GPU — MLProgram rejects unbounded dimensions
        // outright, and the legacy format claimed nodes only to run them on
        // CPU with partitioning overhead. Pin the diarizer to the plain CPU
        // EP; the ASR encoder/decoder sessions above keep the requested
        // provider, which is where the static-shape CoreML win lives.
        #[cfg(feature = "coreml")]
        let exec = if exec.execution_provider == crate::execution::ExecutionProvider::CoreML {
            ExecutionConfig {
                execution_provider: crate::execution::ExecutionProvider::Cpu,
                ..exec
            }
        } else {
            exec
        };

        // The Sortformer's activation shapes vary run to run (streaming
        // state + whole-conversation diarize_full passes), which makes
        // ORT's BFC arena accumulate 128 MB extents without bound in a
        // long-lived process — see ModelConfig::cpu_arena. Plain malloc
        // for this session; the packed weights are unaffected.
        let sortformer_exec = ExecutionConfig {
            cpu_arena: false,
            ..exec
        };
        let sortformer = Sortformer::with_config(
            sortformer_model_path,
            Some(sortformer_exec),
            crate::sortformer::DiarizationConfig::default(),
        )?;

        let mel_basis = crate::audio::create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);

        Ok(Self {
            model,
            sortformer,
            vocab,
            speakers: Vec::new(),
            config: MultitalkerConfig::default(),
            mel_basis,
            audio_buffer: Vec::new(),
            audio_processed: 0,
            chunk_idx: 0,
            diar_preds: Vec::new(),
            diar_pred_offset: 0,
        })
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.speakers.clear();
        self.sortformer.reset_state();
        self.audio_buffer.clear();
        self.audio_processed = 0;
        self.chunk_idx = 0;
        self.diar_preds.clear();
        self.diar_pred_offset = 0;
    }

    /// Returns the current multitalker configuration.
    pub fn multitalker_config(&self) -> &MultitalkerConfig {
        &self.config
    }

    /// Set the maximum number of speakers to track (1..=4).
    ///
    /// Can be called between chunks to adjust mid-session. Existing speaker
    /// instances above the new limit will still produce output for any
    /// already-accumulated tokens, but won't receive new audio.
    pub fn set_max_speakers(&mut self, max_speakers: usize) {
        self.config.max_speakers = max_speakers.clamp(1, NUM_SPEAKERS);
    }

    /// Set the speaker activity threshold (0.0..=1.0).
    ///
    /// A speaker is considered active in a chunk if any frame's probability
    /// exceeds this value. Lower values are more sensitive (detect quieter
    /// speakers sooner), higher values require stronger evidence.
    pub fn set_activity_threshold(&mut self, threshold: f32) {
        self.config.activity_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Blank-logit penalty: positive recovers borderline words, negative
    /// suppresses spurious emissions. Default 0.0 = off. Takes effect on
    /// the next decoded sub-chunk; no state reset needed.
    pub fn set_blank_penalty(&mut self, penalty: f32) {
        self.config.blank_penalty = penalty;
    }

    /// Pause between ready-sub-chunk inferences within one call — see
    /// [`MultitalkerConfig::inter_chunk_pause`]. Takes effect on the next
    /// `transcribe_chunk()` call; no state reset needed. `None` restores
    /// flat-out processing.
    pub fn set_inter_chunk_pause(&mut self, pause: Option<std::time::Duration>) {
        self.config.inter_chunk_pause = pause;
    }

    /// Hold ASR decoding at the settled diarizer frontier — see
    /// [`MultitalkerConfig::hold_at_settled_frontier`]. Set before the
    /// first `transcribe_chunk()` call; flipping it mid-stream is safe
    /// (held sub-chunks simply decode on the next call) but pointless.
    pub fn set_hold_at_settled_frontier(&mut self, hold: bool) {
        self.config.hold_at_settled_frontier = hold;
    }

    /// Upper bound, in seconds, on audio waiting behind the settled
    /// frontier when the hold is on: one not-yet-completed diarizer
    /// stride plus one partial ASR sub-chunk. 0.0 when the hold is off.
    pub fn max_holdback_secs(&self) -> f64 {
        if !self.config.hold_at_settled_frontier {
            return 0.0;
        }
        (self.sortformer.chunk_len + self.config.latency_mode.encoded_frames()) as f64
            * f64::from(SECONDS_PER_ENCODED_FRAME)
    }

    /// Silence sub-chunks a caller must feed at end of stream so the
    /// diarizer's current stride completes from any phase and every held
    /// sub-chunk decodes settled. 0 when the hold is off.
    pub fn settle_flush_subchunks(&self) -> usize {
        if !self.config.hold_at_settled_frontier {
            return 0;
        }
        self.sortformer
            .chunk_len
            .div_ceil(self.config.latency_mode.encoded_frames())
    }

    /// Set the streaming latency mode.
    ///
    /// This changes the encoder chunk size, trading latency for accuracy.
    /// Because encoder caches are tied to the chunk size, this automatically
    /// calls [`reset()`](Self::reset) to clear all state.
    pub fn set_latency_mode(&mut self, mode: LatencyMode) {
        if self.config.latency_mode != mode {
            self.config.latency_mode = mode;
            self.reset();
        }
    }

    /// Return the diarizer's per-frame embeddings on [`ChunkResult`].
    ///
    /// Off by default; see
    /// [`Sortformer::set_emit_frame_embeddings`](crate::sortformer::Sortformer::set_emit_frame_embeddings)
    /// for what they are and what they cost.
    pub fn set_emit_frame_embeddings(&mut self, on: bool) {
        self.sortformer.set_emit_frame_embeddings(on);
    }

    /// What the diarizer currently believes each of its slots sounds like.
    ///
    /// A snapshot of live streaming state: it reflects everything fed so far
    /// and moves on the next call, so read it at the point you mean to
    /// describe (e.g. when a conversation ends) rather than treating it as a
    /// property of any one chunk.
    pub fn slot_profiles(&self) -> Vec<crate::sortformer::SlotProfile> {
        self.sortformer.slot_profiles()
    }

    /// Returns the number of audio samples the caller should provide per
    /// chunk for the current latency mode. This is `chunk_mel_frames * HOP_LENGTH`.
    pub fn chunk_audio_samples(&self) -> usize {
        self.config.chunk_size() * HOP_LENGTH
    }

    /// Run Sortformer diarization on a full audio buffer independently of the
    /// streaming ASR pipeline. Returns per-frame speaker activity at 80ms resolution.
    ///
    /// This is useful when you have the complete audio upfront (e.g., reprocessing)
    /// and want whole-buffer diarization in one pass, independent of the
    /// streaming feed's stride bookkeeping.
    ///
    /// **Does not affect ASR state.** The Sortformer's streaming state is saved
    /// before and restored after, so subsequent `transcribe_chunk` calls are unaffected.
    pub fn diarize_full(&mut self, audio_16k_mono: &[f32]) -> Result<ChunkResult> {
        if audio_16k_mono.is_empty() {
            return Ok(ChunkResult {
                transcripts: vec![],
                speaker_activity: vec![],
                first_activity_frame: 0,
                frame_duration_secs: SECONDS_PER_ENCODED_FRAME,
                frame_embeddings: None,
            });
        }

        // Save Sortformer state, run on full buffer, restore state.
        // Uses the current streaming state (speaker cache, silence profile) so
        // slot assignments match the streaming path — slot 0 stays the same person.
        // Do NOT reset_state() here — that would lose the slot→person mapping.
        let saved_state = self.sortformer.save_state();

        let result = self.sortformer.diarize_chunk_raw(audio_16k_mono);

        self.sortformer.restore_state(saved_state);

        let raw_preds = result?;
        let num_frames = raw_preds.num_valid_frames.min(raw_preds.predictions.nrows());
        let num_spk_cols = raw_preds.predictions.ncols();
        let speaker_activity: Vec<Vec<f32>> = (0..num_frames)
            .map(|t| (0..num_spk_cols).map(|s| raw_preds.predictions[[t, s]]).collect())
            .collect();

        // This pass re-diarises the buffer from frame 0 of the audio it was
        // given, so both frame ranges are relative to that buffer — not to the
        // streaming timeline the other methods report against.
        let frame_embeddings = raw_preds.embeddings.as_ref().map(|e| FrameEmbeddings {
            first_frame: 0,
            dim: e.ncols(),
            data: e.slice(s![..num_frames.min(e.nrows()), ..])
                .iter()
                .copied()
                .collect(),
        });

        Ok(ChunkResult {
            transcripts: vec![],
            speaker_activity,
            first_activity_frame: 0,
            frame_duration_secs: SECONDS_PER_ENCODED_FRAME,
            frame_embeddings,
        })
    }

    /// Get accumulated per-speaker transcripts.
    pub fn get_transcripts(&self) -> Vec<SpeakerTranscript> {
        self.speakers
            .iter()
            .map(|spk| {
                let valid_ids: Vec<usize> = spk
                    .accumulated_tokens
                    .iter()
                    .filter(|&&(t, _, _)| t < VOCAB_SIZE)
                    .map(|&(t, _, _)| t)
                    .collect();
                let words = self.tokens_to_words(&spk.accumulated_tokens);
                SpeakerTranscript {
                    speaker_id: spk.speaker_id,
                    text: self.vocab.decode(&valid_ids),
                    words,
                }
            })
            .collect()
    }

    /// Process audio in streaming mode.
    ///
    /// Accepts any length: all complete ASR sub-chunks (1.12s in Normal
    /// mode) contained in the buffered audio are processed in this call,
    /// and any partial remainder is buffered for the next call. Passing a
    /// large block (e.g. 30s) is significantly cheaper than the equivalent
    /// sequence of per-sub-chunk calls, because the diarizer runs at its
    /// native ~10s stride over the block instead of once per sub-chunk.
    ///
    /// Returns per-speaker text deltas for the processed region. Speakers
    /// are created automatically when first detected.
    pub fn transcribe_chunk(&mut self, audio_chunk: &[f32]) -> Result<Vec<SpeakerTranscript>> {
        let result = self.transcribe_chunk_inner(audio_chunk)?;
        Ok(result.transcripts)
    }

    /// Process one audio chunk, returning transcripts and Sortformer speaker activity.
    ///
    /// Same as [`transcribe_chunk`] but also returns per-frame speaker activity
    /// probabilities from Sortformer. Each frame is 80ms. Use this to determine
    /// which speaker is active at any point in the chunk.
    pub fn transcribe_chunk_with_activity(&mut self, audio_chunk: &[f32]) -> Result<ChunkResult> {
        self.transcribe_chunk_inner(audio_chunk)
    }

    fn transcribe_chunk_inner(&mut self, audio_chunk: &[f32]) -> Result<ChunkResult> {
        self.audio_buffer.extend_from_slice(audio_chunk);

        let t_call = std::time::Instant::now();

        // Feed the diarizer at its NATIVE stride (~10s windows), decoupled
        // from the ASR sub-chunk rate (~1.12s). Sortformer zero-pads short
        // inputs to a full stride window internally, so the old
        // per-sub-chunk `diarize_chunk_raw` paid a full-window inference per
        // 1.12s of audio (~9x waste). Authoritative predictions accumulate
        // in `diar_preds` as strides complete; sub-chunks ahead of the last
        // completed stride use one provisional peek per call (state-saved,
        // so the same audio is re-processed authoritatively later).
        let t = std::time::Instant::now();
        // Absolute index the diarizer's next settled frame will occupy —
        // taken BEFORE the append below, which is what makes the embeddings
        // block self-locating.
        let emb_first_frame = self.diar_pred_offset + self.diar_preds.len();
        let new_raw = self.sortformer.feed_raw(audio_chunk)?;
        let t_sortformer = t.elapsed();
        let frame_embeddings = new_raw.embeddings.as_ref().map(|e| FrameEmbeddings {
            first_frame: emb_first_frame,
            dim: e.ncols(),
            data: e.iter().copied().collect(),
        });
        for row in new_raw.predictions.rows() {
            let mut frame = [0.0f32; NUM_SPEAKERS];
            for s in 0..NUM_SPEAKERS.min(row.len()) {
                frame[s] = row[s];
            }
            self.diar_preds.push(frame);
        }

        let total_audio = self.audio_buffer.len();
        if total_audio < WIN_LENGTH {
            return Ok(ChunkResult {
                transcripts: vec![],
                speaker_activity: vec![],
                first_activity_frame: self.chunk_idx * self.config.latency_mode.encoded_frames(),
                frame_duration_secs: SECONDS_PER_ENCODED_FRAME,
                frame_embeddings,
            });
        }

        // Compute mel ONCE over the full buffer; every ready sub-chunk in
        // this call indexes into it.
        let t = std::time::Instant::now();
        let full_mel = self.compute_mel_spectrogram(&self.audio_buffer)?;
        let t_mel = t.elapsed();
        let total_mel_frames = full_mel.shape()[1];

        let chunk_size = self.config.chunk_size();
        let enc_frames = self.config.latency_mode.encoded_frames();
        let expected_size = PRE_ENCODE_CACHE + chunk_size;

        // Per-speaker token counts at call start: everything appended past
        // these marks is this call's delta. Token-level accumulation means
        // words straddling internal sub-chunk boundaries assemble correctly.
        let token_marks: Vec<(usize, usize)> = self
            .speakers
            .iter()
            .map(|s| (s.speaker_id, s.accumulated_tokens.len()))
            .collect();

        let mut provisional: Option<Array2<f32>> = None;
        let mut speaker_activity: Vec<Vec<f32>> = Vec::new();
        // Where the first sub-chunk of this call starts, before the loop
        // advances `chunk_idx`.
        let first_activity_frame = self.chunk_idx * enc_frames;

        // Per-call stage timing, printed when PARAKEET_STAGE_TIMING is set.
        // Answers "where does feed time actually go" (sortformer vs encoder
        // vs decode) without a profiler; ~ns overhead when unset.
        let stage_timing = std::env::var("PARAKEET_STAGE_TIMING").is_ok();
        let mut t_peek = std::time::Duration::ZERO;
        let mut t_encoder = std::time::Duration::ZERO;
        let mut t_decode = std::time::Duration::ZERO;

        // Process ALL ready ASR sub-chunks (a 30s caller does ~26 here; a
        // 1.12s streaming caller does one, exactly as before).
        let mut first_subchunk = true;
        loop {
            let processed_mel_frames = self.audio_processed / HOP_LENGTH;
            let available_new_frames = total_mel_frames.saturating_sub(processed_mel_frames);
            if available_new_frames < chunk_size {
                break;
            }

            // Absolute diarization frame range for this sub-chunk. Both
            // sides use 80ms frames (SUBSAMPLING mel frames), so the
            // mapping is exact for every latency mode.
            let f_start = self.chunk_idx * enc_frames;
            let f_end = f_start + enc_frames;
            let covered = self.diar_pred_offset + self.diar_preds.len();

            // Settled-frontier hold: a sub-chunk not fully covered by
            // COMPLETED diarizer strides waits for the next call instead
            // of decoding against the provisional peek. The provisional
            // predictions come from a zero-padded partial stride and
            // genuinely differ from the authoritative pass that follows —
            // but by then the words' channel routing and the emitted
            // activity were already final. Holding trades up to one
            // stride of latency (~10s) for every word being decoded with
            // settled speaker conditioning. The held audio stays in the
            // buffer (the trim never removes past `audio_processed`), as
            // do its settled-but-undecoded diar frames.
            if self.config.hold_at_settled_frontier && f_end > covered {
                break;
            }

            // Optional duty-cycling between sub-chunk inferences (never
            // before the first, never after the last — placed after both
            // break conditions so we never sleep for a sub-chunk we don't
            // process). See `MultitalkerConfig::inter_chunk_pause`.
            if !first_subchunk && let Some(pause) = self.config.inter_chunk_pause {
                std::thread::sleep(pause);
            }
            first_subchunk = false;

            if f_end > covered && provisional.is_none() {
                let t = std::time::Instant::now();
                provisional = Some(self.sortformer.peek_buffered_raw()?.predictions);
                t_peek += t.elapsed();
            }
            let window = assemble_diar_window(
                &self.diar_preds,
                self.diar_pred_offset,
                provisional.as_ref(),
                f_start,
                f_end,
            );

            // Determine active speakers in this sub-chunk
            let mut active_speakers = Vec::new();
            for spk_id in 0..self.config.max_speakers {
                if spk_id >= window.ncols() {
                    break;
                }
                let max_activity = (0..window.nrows())
                    .map(|t| window[[t, spk_id]])
                    .fold(0.0f32, f32::max);
                if max_activity > self.config.activity_threshold {
                    active_speakers.push(spk_id);
                }
            }

            // Ensure speaker instances exist
            for &spk_id in &active_speakers {
                if !self.speakers.iter().any(|s| s.speaker_id == spk_id) {
                    self.speakers.push(SpeakerInstance::new(spk_id));
                }
            }

            // Build encoder input chunk
            let is_first_chunk = self.chunk_idx == 0;
            let main_start = processed_mel_frames;
            let mel_chunk =
                self.build_mel_chunk(&full_mel, main_start, is_first_chunk, expected_size)?;
            let chunk_length = expected_size;
            let chunk_frame_offset = self.chunk_idx * enc_frames;

            // For each active speaker, run encoder with speaker-specific masks
            for &spk_id in &active_speakers {
                // Derive spk_targets and bg_spk_targets from the mask window
                let (spk_targets, bg_spk_targets) =
                    self.derive_speaker_targets(&window, spk_id, chunk_length)?;

                let spk_idx = self
                    .speakers
                    .iter()
                    .position(|s| s.speaker_id == spk_id)
                    .unwrap();

                // Run encoder with this speaker's targets and cache
                let t = std::time::Instant::now();
                let (encoded, enc_len, new_cache) = self.model.run_encoder(
                    &mel_chunk,
                    chunk_length as i64,
                    &self.speakers[spk_idx].encoder_cache,
                    &spk_targets,
                    &bg_spk_targets,
                )?;
                t_encoder += t.elapsed();
                self.speakers[spk_idx].encoder_cache = new_cache;

                // Decode tokens for this speaker
                let t = std::time::Instant::now();
                let tokens = self.decode_chunk_for_speaker(
                    spk_idx,
                    &encoded,
                    enc_len as usize,
                    chunk_frame_offset,
                )?;
                t_decode += t.elapsed();
                self.speakers[spk_idx]
                    .accumulated_tokens
                    .extend(tokens.iter().copied());
            }

            // Activity output: this sub-chunk's mask window (contiguous,
            // non-overlapping across sub-chunks; frame 0 = start of the
            // first sub-chunk processed in this call).
            for t in 0..window.nrows() {
                speaker_activity.push((0..window.ncols()).map(|s| window[[t, s]]).collect());
            }

            // Advance processed position
            self.audio_processed += chunk_size * HOP_LENGTH;
            self.chunk_idx += 1;
        }

        // Trim audio buffer
        let keep_samples = (PRE_ENCODE_CACHE + chunk_size) * HOP_LENGTH + WIN_LENGTH;
        if self.audio_buffer.len() > keep_samples * 2 {
            let remove = self.audio_buffer.len() - keep_samples;
            let actual_remove = remove.min(self.audio_processed);
            self.audio_buffer.drain(0..actual_remove);
            self.audio_processed -= actual_remove;
        }

        // Trim consumed diarization frames (all sub-chunks below chunk_idx
        // are done; provisional frames beyond `covered` were never stored).
        let consumed = self.chunk_idx * enc_frames;
        if consumed > self.diar_pred_offset {
            let drop = (consumed - self.diar_pred_offset).min(self.diar_preds.len());
            self.diar_preds.drain(..drop);
            self.diar_pred_offset += drop;
        }

        // Build per-speaker deltas for this call from tokens accumulated
        // past the call-start marks.
        let mut results = Vec::new();
        for spk in &self.speakers {
            let mark = token_marks
                .iter()
                .find(|(id, _)| *id == spk.speaker_id)
                .map(|(_, n)| *n)
                .unwrap_or(0);
            let new_tokens = &spk.accumulated_tokens[mark..];
            if new_tokens.is_empty() {
                continue;
            }

            let mut text = String::new();
            for &(t, _, _) in new_tokens {
                if t < VOCAB_SIZE {
                    text.push_str(&self.vocab.decode_single(t));
                }
            }

            if !text.is_empty() {
                let words = self.tokens_to_words(new_tokens);
                results.push(SpeakerTranscript {
                    speaker_id: spk.speaker_id,
                    text,
                    words,
                });
            }
        }

        if stage_timing {
            eprintln!(
                "STAGE_TIMING call={:.0}ms sortformer={:.0}ms peek={:.0}ms mel={:.0}ms encoder={:.0}ms decode={:.0}ms",
                t_call.elapsed().as_secs_f64() * 1e3,
                t_sortformer.as_secs_f64() * 1e3,
                t_peek.as_secs_f64() * 1e3,
                t_mel.as_secs_f64() * 1e3,
                t_encoder.as_secs_f64() * 1e3,
                t_decode.as_secs_f64() * 1e3,
            );
        }

        Ok(ChunkResult {
            transcripts: results,
            speaker_activity,
            first_activity_frame,
            frame_duration_secs: SECONDS_PER_ENCODED_FRAME,
            frame_embeddings,
        })
    }

    /// Non-streaming transcription of an audio file.
    pub fn transcribe_file_multitalker<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
    ) -> Result<Vec<SpeakerTranscript>> {
        let (audio, spec) = crate::audio::load_audio(audio_path)?;

        if spec.sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, spec.sample_rate
            )));
        }

        let audio = if spec.channels > 1 {
            audio
                .chunks(spec.channels as usize)
                .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            audio
        };

        self.transcribe_audio_multitalker(&audio)
    }

    /// Non-streaming transcription of raw audio samples.
    pub fn transcribe_audio_multitalker(
        &mut self,
        audio: &[f32],
    ) -> Result<Vec<SpeakerTranscript>> {
        self.reset();

        let audio_chunk_size = self.chunk_audio_samples();
        for chunk in audio.chunks(audio_chunk_size) {
            let chunk_vec = if chunk.len() < audio_chunk_size {
                let mut p = chunk.to_vec();
                p.resize(audio_chunk_size, 0.0);
                p
            } else {
                chunk.to_vec()
            };
            self.transcribe_chunk(&chunk_vec)?;
        }

        // Flush with silence
        let flush_chunk = vec![0.0f32; audio_chunk_size];
        for _ in 0..3 {
            self.transcribe_chunk(&flush_chunk)?;
        }

        Ok(self.get_transcripts())
    }

    /// Derive per-speaker target masks from raw Sortformer predictions.
    ///
    /// For the target speaker k:
    /// - `spk_targets[t] = raw_preds[t, k]`
    /// - `bg_spk_targets[t] = max(raw_preds[t, j]) for j != k`
    ///
    /// The masks are resized/interpolated to match the encoder's time dimension.
    fn derive_speaker_targets(
        &self,
        diar_preds: &Array2<f32>,
        speaker_id: usize,
        encoder_time: usize,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let diar_frames = diar_preds.nrows();

        let mut spk_vals = Vec::with_capacity(encoder_time);
        let mut bg_vals = Vec::with_capacity(encoder_time);

        for enc_t in 0..encoder_time {
            // Map encoder time to diarisation time (nearest-neighbour)
            let diar_t = if diar_frames > 0 && encoder_time > 0 {
                (enc_t * diar_frames / encoder_time).min(diar_frames - 1)
            } else {
                0
            };

            if diar_t < diar_frames && speaker_id < diar_preds.ncols() {
                let spk_val = diar_preds[[diar_t, speaker_id]];
                let bg_val = (0..diar_preds.ncols())
                    .filter(|&j| j != speaker_id)
                    .map(|j| diar_preds[[diar_t, j]])
                    .fold(0.0f32, f32::max);
                spk_vals.push(spk_val);
                bg_vals.push(bg_val);
            } else {
                // No diarisation data: assume single speaker
                spk_vals.push(1.0);
                bg_vals.push(0.0);
            }
        }

        let spk_targets = Array2::from_shape_vec((1, encoder_time), spk_vals)
            .map_err(|e| Error::Model(format!("spk_targets shape mismatch: {e}")))?;
        let bg_spk_targets = Array2::from_shape_vec((1, encoder_time), bg_vals)
            .map_err(|e| Error::Model(format!("bg_spk_targets shape mismatch: {e}")))?;

        Ok((spk_targets, bg_spk_targets))
    }

    fn build_mel_chunk(
        &self,
        full_mel: &Array2<f32>,
        main_start: usize,
        is_first_chunk: bool,
        expected_size: usize,
    ) -> Result<Array3<f32>> {
        let total_mel_frames = full_mel.shape()[1];
        let chunk_size = self.config.chunk_size();
        let mut chunk_data = vec![0.0f32; N_MELS * expected_size];

        if is_first_chunk {
            for f in 0..chunk_size.min(total_mel_frames) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] = full_mel[[m, f]];
                }
            }
        } else {
            let cache_start = main_start.saturating_sub(PRE_ENCODE_CACHE);
            let cache_frames = main_start - cache_start;
            let cache_offset = PRE_ENCODE_CACHE - cache_frames;

            for f in 0..cache_frames {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + cache_offset + f] =
                        full_mel[[m, cache_start + f]];
                }
            }

            for f in 0..chunk_size.min(total_mel_frames - main_start) {
                for m in 0..N_MELS {
                    chunk_data[m * expected_size + PRE_ENCODE_CACHE + f] =
                        full_mel[[m, main_start + f]];
                }
            }
        }

        Array3::from_shape_vec((1, N_MELS, expected_size), chunk_data)
            .map_err(|e| Error::Model(format!("Failed to create mel chunk: {e}")))
    }

    fn decode_chunk_for_speaker(
        &mut self,
        spk_idx: usize,
        encoder_out: &Array3<f32>,
        enc_frames: usize,
        chunk_frame_offset: usize,
    ) -> Result<Vec<(usize, usize, f32)>> {
        let mut tokens = Vec::new();
        let hidden_dim = encoder_out.shape()[1];

        for t in 0..enc_frames {
            let frame = encoder_out.slice(s![0, .., t]).to_owned();
            let frame = frame
                .to_shape((1, 1, hidden_dim))
                .map_err(|e| Error::Model(format!("Failed to reshape frame: {e}")))?
                .to_owned();

            let absolute_frame = chunk_frame_offset + t;

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let (mut logits, new_state_1, new_state_2) = self.model.run_decoder(
                    &frame,
                    self.speakers[spk_idx].last_token,
                    &self.speakers[spk_idx].state_1,
                    &self.speakers[spk_idx].state_2,
                )?;

                apply_blank_penalty(&mut logits, self.config.blank_penalty);
                let (max_idx, max_val) = crate::tensor_utils::argmax_f32(logits.iter().copied());

                if max_idx == BLANK_ID {
                    break;
                }

                // Compute softmax probability for the chosen token.
                let log_sum_exp = {
                    let max_for_stability = max_val;
                    let sum: f32 = logits.iter().map(|&v| (v - max_for_stability).exp()).sum();
                    max_for_stability + sum.ln()
                };
                let confidence = (max_val - log_sum_exp).exp();

                tokens.push((max_idx, absolute_frame, confidence));
                self.speakers[spk_idx].last_token = max_idx as i32;
                self.speakers[spk_idx].state_1 = new_state_1;
                self.speakers[spk_idx].state_2 = new_state_2;
            }
        }

        Ok(tokens)
    }

    /// Convert (token_id, absolute_frame) pairs into word-level timestamps.
    ///
    /// Token end time = next token's start (spans the full inter-token gap),
    /// with a 1-frame fallback for the last token. This gives more accurate
    /// word boundaries than a fixed 80ms per token.
    fn tokens_to_words(&self, tokens: &[(usize, usize, f32)]) -> Vec<WordTimestamp> {
        let filtered: Vec<(usize, usize, f32)> = tokens
            .iter()
            .filter(|(id, _, _)| *id < VOCAB_SIZE)
            .copied()
            .collect();
        let timed: Vec<TimedToken> = filtered
            .iter()
            .enumerate()
            .map(|(i, &(id, frame, conf))| {
                let start = frame as f32 * SECONDS_PER_ENCODED_FRAME;
                let end = if i + 1 < filtered.len() {
                    filtered[i + 1].1 as f32 * SECONDS_PER_ENCODED_FRAME
                } else {
                    (frame + 1) as f32 * SECONDS_PER_ENCODED_FRAME
                };
                TimedToken {
                    text: self.vocab.decode_single(id),
                    start,
                    end,
                    confidence: conf,
                }
            })
            .collect();

        timestamps::group_by_words(&timed)
            .into_iter()
            .map(|t| WordTimestamp {
                word: t.text,
                start_secs: t.start,
                end_secs: t.end,
                confidence: t.confidence,
            })
            .collect()
    }

    /// Compute mel spectrogram using shared audio utilities.
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>> {
        if audio.is_empty() {
            return Ok(Array2::zeros((N_MELS, 0)));
        }

        let preemph = crate::audio::apply_preemphasis(audio, PREEMPH);
        let spec = crate::audio::stft(&preemph, N_FFT, HOP_LENGTH, WIN_LENGTH)?;
        let mel = self.mel_basis.dot(&spec);

        Ok(mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln()))
    }
}

/// Subtract `penalty` from the blank logit — index [`BLANK_ID`], the last
/// of the `vocab_size + 1` joint outputs. No-op at 0.0. Applied before
/// both the argmax and the confidence softmax, so an emitted token's
/// confidence reflects the distribution the decode actually ran on.
fn apply_blank_penalty(logits: &mut Array1<f32>, penalty: f32) {
    if penalty == 0.0 {
        return;
    }
    if let Some(blank_logit) = logits.get_mut(BLANK_ID) {
        *blank_logit -= penalty;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth_rows(vals: &[f32]) -> Vec<[f32; NUM_SPEAKERS]> {
        vals.iter().map(|&v| [v; NUM_SPEAKERS]).collect()
    }

    // Joint output: VOCAB_SIZE tokenizer tokens (ids 0..=1023) + blank
    // appended last at index 1024 = BLANK_ID = VOCAB_SIZE.
    const LOGIT_WIDTH: usize = VOCAB_SIZE + 1;

    // The penalty must hit blank (1024), not the last tokenizer token (1023).
    #[test]
    fn penalty_targets_blank_not_last_tokenizer_token() {
        let mut v = Array1::from_elem(LOGIT_WIDTH, -20.0f32);
        v[VOCAB_SIZE - 1] = 5.0;
        v[BLANK_ID] = 5.0;
        apply_blank_penalty(&mut v, 3.0);
        assert_eq!(v[VOCAB_SIZE - 1], 5.0);
        assert_eq!(v[BLANK_ID], 2.0);
    }

    // The 0.0 default must leave the logits bit-identical (feature off =
    // stock decode).
    #[test]
    fn zero_penalty_is_a_no_op() {
        let mut v = Array1::from_elem(LOGIT_WIDTH, -20.0f32);
        v[300] = 6.5;
        v[BLANK_ID] = 5.0;
        let original = v.clone();
        apply_blank_penalty(&mut v, 0.0);
        assert_eq!(v, original);
    }

    // A penalized blank that still leads must still decode as blank —
    // the knob shifts borderline frames only.
    #[test]
    fn wide_margin_blank_still_wins_after_penalty() {
        let mut v = Array1::from_elem(LOGIT_WIDTH, -20.0f32);
        v[BLANK_ID] = 10.0;
        apply_blank_penalty(&mut v, 3.0);
        let (max_idx, _) = crate::tensor_utils::argmax_f32(v.iter().copied());
        assert_eq!(max_idx, BLANK_ID);
    }

    fn prov(vals: &[f32]) -> Array2<f32> {
        let n = vals.len();
        Array2::from_shape_fn((n, NUM_SPEAKERS), |(i, _)| vals[i])
    }

    #[test]
    fn window_fully_authoritative() {
        let auth = auth_rows(&[0.1, 0.2, 0.3, 0.4]);
        let w = assemble_diar_window(&auth, 10, None, 11, 13);
        assert_eq!(w.nrows(), 2);
        assert_eq!(w[[0, 0]], 0.2);
        assert_eq!(w[[1, 3]], 0.3);
    }

    #[test]
    fn window_spans_authoritative_and_provisional() {
        let auth = auth_rows(&[0.1, 0.2]); // frames 0..2
        let p = prov(&[0.7, 0.8]); // frames 2..4
        let w = assemble_diar_window(&auth, 0, Some(&p), 1, 4);
        assert_eq!(w.nrows(), 3);
        assert_eq!(w[[0, 0]], 0.2); // frame 1: authoritative
        assert_eq!(w[[1, 0]], 0.7); // frame 2: provisional
        assert_eq!(w[[2, 0]], 0.8); // frame 3: provisional
    }

    #[test]
    fn window_uncovered_frames_are_zero() {
        // No provisional and range beyond authoritative coverage → zeros
        // (treated as silence, never a panic).
        let auth = auth_rows(&[0.5]);
        let w = assemble_diar_window(&auth, 0, None, 0, 3);
        assert_eq!(w[[0, 0]], 0.5);
        assert_eq!(w[[1, 0]], 0.0);
        assert_eq!(w[[2, 0]], 0.0);

        // Provisional shorter than needed → trailing zeros.
        let p = prov(&[0.9]);
        let w = assemble_diar_window(&auth, 0, Some(&p), 0, 3);
        assert_eq!(w[[1, 0]], 0.9);
        assert_eq!(w[[2, 0]], 0.0);
    }

    #[test]
    fn window_empty_range() {
        let auth = auth_rows(&[0.5]);
        let w = assemble_diar_window(&auth, 0, None, 3, 3);
        assert_eq!(w.nrows(), 0);
    }

    #[test]
    fn an_embedding_block_is_addressed_absolutely_not_positionally() {
        // Embeddings and activity start at different frames in the same
        // call, so a caller lining them up by position silently reads the
        // wrong speaker's audio. `first_frame` is what makes that impossible.
        let e = FrameEmbeddings {
            first_frame: 40,
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        assert_eq!(e.frames(), 3);
        assert_eq!(e.row(0), Some(&[1.0, 2.0][..]));
        assert_eq!(e.frame(40), Some(&[1.0, 2.0][..]));
        assert_eq!(e.frame(42), Some(&[5.0, 6.0][..]));
        // Outside the block in either direction, not a wrapped or clamped row.
        assert_eq!(e.frame(39), None);
        assert_eq!(e.frame(43), None);
        assert_eq!(e.row(3), None);
    }

    #[test]
    fn an_empty_embedding_block_reports_no_frames_rather_than_dividing_by_zero() {
        let e = FrameEmbeddings { first_frame: 0, dim: 0, data: vec![] };
        assert_eq!(e.frames(), 0);
        assert_eq!(e.row(0), None);
    }
}

/// Implement the Transcriber trait for single-speaker fallback.
/// Runs with spk_targets=1.0 and bg_spk_targets=0.0 (no diarisation),
/// treating the multitalker encoder as a standard streaming ASR encoder.
impl Transcriber for MultitalkerASR {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        _mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        if sample_rate != SAMPLE_RATE as u32 {
            return Err(Error::Audio(format!(
                "Expected {} Hz, got {} Hz",
                SAMPLE_RATE, sample_rate
            )));
        }

        let audio = if channels > 1 {
            audio
                .chunks(channels as usize)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            audio
        };

        // Single-speaker mode: run encoder with full speaker activity
        self.reset();

        let mel = self.compute_mel_spectrogram(&audio)?;
        let total_frames = mel.shape()[1];

        if total_frames == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                tokens: Vec::new(),
            });
        }

        // Create a single speaker instance
        self.speakers.push(SpeakerInstance::new(0));

        let chunk_size = self.config.chunk_size();
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;

        while buffer_idx < total_frames {
            let expected_size = PRE_ENCODE_CACHE + chunk_size;

            let is_first = chunk_idx == 0;
            let mel_chunk = self.build_mel_chunk(&mel, buffer_idx, is_first, expected_size)?;
            // Use expected_size consistently (matches transcribe_chunk path)
            let chunk_length = expected_size;

            // Single-speaker: full activity, no background
            let spk_targets = Array2::from_elem((1, chunk_length), 1.0f32);
            let bg_spk_targets = Array2::from_elem((1, chunk_length), 0.0f32);

            let (encoded, enc_len, new_cache) = self.model.run_encoder(
                &mel_chunk,
                chunk_length as i64,
                &self.speakers[0].encoder_cache,
                &spk_targets,
                &bg_spk_targets,
            )?;
            self.speakers[0].encoder_cache = new_cache;

            let chunk_frame_offset =
                chunk_idx * self.config.latency_mode.encoded_frames();
            let tokens =
                self.decode_chunk_for_speaker(0, &encoded, enc_len as usize, chunk_frame_offset)?;
            self.speakers[0].accumulated_tokens.extend(tokens.iter().copied());

            buffer_idx += chunk_size;
            chunk_idx += 1;
        }

        let valid_ids: Vec<usize> = self.speakers[0]
            .accumulated_tokens
            .iter()
            .filter(|&&(t, _, _)| t < VOCAB_SIZE)
            .map(|&(t, _, _)| t)
            .collect();

        let text = self.vocab.decode(&valid_ids);

        Ok(TranscriptionResult {
            text,
            tokens: Vec::new(),
        })
    }
}
