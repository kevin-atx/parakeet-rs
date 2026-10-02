//! The streaming diarizer the multitalker pipeline drives: Streaming
//! Sortformer v2 (4 speakers) or Nemotron 3 Diarization (8), chosen by the
//! ONNX file itself — a Nemotron export declares `model_family`, a v2 export
//! declares none — so a caller passes a path and nothing else.
//!
//! An enum rather than a trait object: two concrete engines, matched in one
//! place, with no dynamic dispatch on the per-stride path.

use crate::error::Result;
use crate::execution::ModelConfig;
use crate::nemotron_diar::{self, NemotronDiarizer, NemotronState};
use crate::sortformer::{DiarizationConfig, RawDiarizationPredictions, SlotProfile, Sortformer, SortformerState, NUM_SPEAKERS};
use std::path::Path;

pub enum Diarizer {
    SortformerV2(Box<Sortformer>),
    Nemotron3(Box<NemotronDiarizer>),
}

/// Saved streaming state of either diarizer, for save/restore around an
/// independent pass. Restoring a state into the other kind is a no-op.
pub enum DiarizerState {
    SortformerV2(Box<SortformerState>),
    Nemotron3(Box<NemotronState>),
}

impl Diarizer {
    /// Load whichever diarizer `model_path` holds, with one session.
    pub fn load<P: AsRef<Path>>(model_path: P, execution_config: Option<ModelConfig>) -> Result<Self> {
        let session = execution_config.unwrap_or_default().build_session(model_path.as_ref())?;
        if nemotron_diar::model_family(&session).as_deref() == Some(nemotron_diar::MODEL_FAMILY) {
            Ok(Self::Nemotron3(Box::new(NemotronDiarizer::from_session(session)?)))
        } else {
            Ok(Self::SortformerV2(Box::new(Sortformer::from_session(session, DiarizationConfig::default())?)))
        }
    }

    /// Stable name for logs and provenance.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SortformerV2(_) => "sortformer_v2",
            Self::Nemotron3(_) => "nemotron3",
        }
    }

    pub fn num_speakers(&self) -> usize {
        match self {
            Self::SortformerV2(_) => NUM_SPEAKERS,
            Self::Nemotron3(d) => d.num_speakers(),
        }
    }

    /// Native stride, in 80 ms frames.
    pub fn chunk_len(&self) -> usize {
        match self {
            Self::SortformerV2(d) => d.chunk_len,
            Self::Nemotron3(d) => d.cfg.chunk_len,
        }
    }

    pub fn right_context(&self) -> usize {
        match self {
            Self::SortformerV2(d) => d.right_context,
            Self::Nemotron3(d) => d.cfg.right_context,
        }
    }

    pub fn feed_raw(&mut self, audio: &[f32]) -> Result<RawDiarizationPredictions> {
        match self {
            Self::SortformerV2(d) => d.feed_raw(audio),
            Self::Nemotron3(d) => d.feed_raw(audio),
        }
    }

    pub fn peek_buffered_raw(&mut self) -> Result<RawDiarizationPredictions> {
        match self {
            Self::SortformerV2(d) => d.peek_buffered_raw(),
            Self::Nemotron3(d) => d.peek_buffered_raw(),
        }
    }

    pub fn diarize_chunk_raw(&mut self, audio: &[f32]) -> Result<RawDiarizationPredictions> {
        match self {
            Self::SortformerV2(d) => d.diarize_chunk_raw(audio),
            Self::Nemotron3(d) => d.diarize_chunk_raw(audio),
        }
    }

    pub fn save_state(&self) -> DiarizerState {
        match self {
            Self::SortformerV2(d) => DiarizerState::SortformerV2(Box::new(d.save_state())),
            Self::Nemotron3(d) => DiarizerState::Nemotron3(Box::new(d.save_state())),
        }
    }

    pub fn restore_state(&mut self, state: DiarizerState) {
        match (self, state) {
            (Self::SortformerV2(d), DiarizerState::SortformerV2(s)) => d.restore_state(*s),
            (Self::Nemotron3(d), DiarizerState::Nemotron3(s)) => d.restore_state(*s),
            _ => {}
        }
    }

    pub fn reset_state(&mut self) {
        match self {
            Self::SortformerV2(d) => d.reset_state(),
            Self::Nemotron3(d) => d.reset_state(),
        }
    }

    pub fn slot_profiles(&self) -> Vec<SlotProfile> {
        match self {
            Self::SortformerV2(d) => d.slot_profiles(),
            Self::Nemotron3(d) => d.slot_profiles(),
        }
    }

    pub fn set_emit_frame_embeddings(&mut self, on: bool) {
        match self {
            Self::SortformerV2(d) => d.set_emit_frame_embeddings(on),
            Self::Nemotron3(d) => d.set_emit_frame_embeddings(on),
        }
    }
}
