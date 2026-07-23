use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::tensor_utils::{
    extract_1d_i64, extract_3d_f32, extract_4d_f32, extract_flat_f32, extract_scalar_i64,
};
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::path::Path;

/// Encoder cache for the multitalker model.
///
/// Unlike `NemotronEncoderCache` which uses `[n_layers, batch, ...]` ordering,
/// the multitalker ONNX encoder expects `[batch, n_layers, ...]` because the
/// export wrapper calls `forward_for_export()` which transposes (0,1) internally.
#[derive(Clone)]
pub(crate) struct MultitalkerEncoderCache {
    /// [1, n_layers, left_context, d_model] - batch-first cache
    pub(crate) cache_last_channel: Array4<f32>,
    /// [1, n_layers, d_model, conv_context] - batch-first cache
    pub(crate) cache_last_time: Array4<f32>,
    /// [1] - current cache length
    pub(crate) cache_last_channel_len: Array1<i64>,
}

impl MultitalkerEncoderCache {
    pub(crate) fn new(
        num_layers: usize,
        left_context: usize,
        hidden_dim: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            // batch-first: [1, n_layers, left_context, hidden_dim]
            cache_last_channel: Array4::zeros((1, num_layers, left_context, hidden_dim)),
            // batch-first: [1, n_layers, hidden_dim, conv_context]
            cache_last_time: Array4::zeros((1, num_layers, hidden_dim, conv_context)),
            cache_last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

/// Multitalker ONNX wrapper.
/// Encoder accepts additional spk_targets and bg_spk_targets inputs for speaker
/// kernel injection. Decoder is identical to Nemotron's RNNT decoder.
pub(crate) struct MultitalkerModel {
    encoder: Session,
    decoder_joint: Session,
}

impl MultitalkerModel {
    pub(crate) fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        #[cfg(feature = "coreml")]
        let is_coreml =
            exec_config.execution_provider == crate::execution::ExecutionProvider::CoreML;
        #[cfg(not(feature = "coreml"))]
        let is_coreml = false;

        // Encoder file preference is EP-aware. The int8 export is fastest on
        // the CPU EP, but its DynamicQuantizeLinear/MatMulInteger clusters are
        // unsupported by CoreML — the graph shatters into ~300 partitions and
        // runs SLOWER than CPU. The fp16 export (produced by
        // scripts/make_multitalker_coreml_fp16.py: dequantized weights,
        // Where-masking rewritten to arithmetic, no-op Slice removed) compiles
        // to ~4 CoreML partitions and beats int8-CPU while freeing the CPU.
        let encoder_path = {
            let fp16 = model_dir.join("encoder.fp16.onnx");
            let int8 = model_dir.join("encoder.int8.onnx");
            let fp32 = model_dir.join("encoder.onnx");
            let order = if is_coreml {
                [&fp16, &int8, &fp32]
            } else {
                [&int8, &fp32, &fp16]
            };
            match order.iter().find(|p| p.exists()) {
                Some(p) => (*p).clone(),
                None => {
                    return Err(Error::Config(format!(
                        "Missing encoder.fp16.onnx, encoder.int8.onnx or encoder.onnx in {}",
                        model_dir.display()
                    )))
                }
            }
        };

        let decoder_path = {
            let int8 = model_dir.join("decoder_joint.int8.onnx");
            let fp32 = model_dir.join("decoder_joint.onnx");
            if int8.exists() {
                int8
            } else if fp32.exists() {
                fp32
            } else {
                return Err(Error::Config(format!(
                    "Missing decoder_joint.onnx or decoder_joint.int8.onnx in {}",
                    model_dir.display()
                )));
            }
        };

        // Under CoreML the DECODER stays on the CPU EP by default: it's a
        // tiny per-token LSTM step (DynamicQuantizeLSTM, unsupported by
        // CoreML anyway) called up to 10x per encoded frame, so per-dispatch
        // EP overhead swamps any compute win. CoreML accelerates the encoder,
        // which is where ~all the FLOPs live.
        //
        // PARAKEET_COREML_ONLY=encoder|decoder|both overrides the split for
        // diagnostics — e.g. the 2026-07-22 bisect that isolated the Apple
        // transpose+identity-slice miscompilation to the encoder session.
        let cpu_config = ExecutionConfig {
            execution_provider: crate::execution::ExecutionProvider::Cpu,
            ..exec_config.clone()
        };
        let (enc_config, dec_config) = match std::env::var("PARAKEET_COREML_ONLY").as_deref() {
            Ok("encoder") => (exec_config.clone(), cpu_config),
            Ok("decoder") => (cpu_config, exec_config.clone()),
            Ok("both") => (exec_config.clone(), exec_config),
            _ if is_coreml => (exec_config.clone(), cpu_config),
            _ => (exec_config.clone(), exec_config),
        };

        let encoder = enc_config.build_session(&encoder_path)?;
        let decoder_joint = dec_config.build_session(&decoder_path)?;

        Ok(Self {
            encoder,
            decoder_joint,
        })
    }

    /// Run encoder with cache-aware streaming and speaker target injection.
    ///
    /// Compared to NemotronModel::run_encoder(), this adds two extra inputs:
    /// - `spk_targets`: per-frame target speaker activity [1, T_enc]
    /// - `bg_spk_targets`: per-frame background speaker activity [1, T_enc]
    ///
    /// Cache format is batch-first: [1, n_layers, ...] (unlike Nemotron which
    /// uses [n_layers, 1, ...]).
    pub(crate) fn run_encoder(
        &mut self,
        features: &Array3<f32>,
        length: i64,
        cache: &MultitalkerEncoderCache,
        spk_targets: &Array2<f32>,
        bg_spk_targets: &Array2<f32>,
    ) -> Result<(Array3<f32>, i64, MultitalkerEncoderCache)> {
        let length_arr = Array1::from_vec(vec![length]);

        let outputs = self.encoder.run(ort::inputs![
            "processed_signal" => ort::value::Value::from_array(features.clone())?,
            "processed_signal_length" => ort::value::Value::from_array(length_arr)?,
            "cache_last_channel" => ort::value::Value::from_array(cache.cache_last_channel.clone())?,
            "cache_last_time" => ort::value::Value::from_array(cache.cache_last_time.clone())?,
            "cache_last_channel_len" => ort::value::Value::from_array(cache.cache_last_channel_len.clone())?,
            "spk_targets" => ort::value::Value::from_array(spk_targets.clone())?,
            "bg_spk_targets" => ort::value::Value::from_array(bg_spk_targets.clone())?
        ])?;

        let encoder_out = extract_3d_f32(&outputs["encoded"], "encoder output")?;
        let encoded_len = extract_scalar_i64(&outputs["encoded_len"], "encoded_len")?;

        let new_cache = MultitalkerEncoderCache {
            cache_last_channel: extract_4d_f32(
                &outputs["cache_last_channel_next"],
                "cache_last_channel",
            )?,
            cache_last_time: extract_4d_f32(&outputs["cache_last_time_next"], "cache_last_time")?,
            cache_last_channel_len: extract_1d_i64(
                &outputs["cache_last_channel_len_next"],
                "cache_len",
            )?,
        };

        Ok((encoder_out, encoded_len, new_cache))
    }

    /// Run RNNT decoder step.
    ///
    /// The ONNX layout differs from the standard NeMo export (model_nemotron.rs):
    /// encoder_outputs is [B, T, D] (not [B, D, T]), there is no target_length
    /// input, and states are named states_1/states_2. This matches the custom
    /// DecoderJointExport wrapper used in export_multitalker.py.
    ///
    /// Returns: (logits [vocab_size+1], new_state_1, new_state_2)
    pub(crate) fn run_decoder(
        &mut self,
        encoder_frame: &Array3<f32>,
        target_token: i32,
        state_1: &Array3<f32>,
        state_2: &Array3<f32>,
    ) -> Result<(Array1<f32>, Array3<f32>, Array3<f32>)> {
        let targets = Array2::from_shape_vec((1, 1), vec![target_token as i64])
            .map_err(|e| Error::Model(format!("Failed to create targets: {e}")))?;

        let outputs = self.decoder_joint.run(ort::inputs![
            "encoder_outputs" => ort::value::Value::from_array(encoder_frame.clone())?,
            "targets" => ort::value::Value::from_array(targets)?,
            "input_states_1" => ort::value::Value::from_array(state_1.clone())?,
            "input_states_2" => ort::value::Value::from_array(state_2.clone())?
        ])?;

        let logits = extract_flat_f32(&outputs["outputs"], "logits")?;
        let new_state_1 = extract_3d_f32(&outputs["states_1"], "state_1")?;
        let new_state_2 = extract_3d_f32(&outputs["states_2"], "state_2")?;

        Ok((logits, new_state_1, new_state_2))
    }
}
