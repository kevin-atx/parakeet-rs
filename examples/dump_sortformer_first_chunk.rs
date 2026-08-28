// Parity instrument, phase b: run the PRODUCT diarization path (parakeet-rs
// mel front end + ONNX session, CPU EP, fresh empty caches) over the first
// 10.0s of a wav and dump the raw per-frame speaker probabilities, for
// comparison against NeMo's forward_for_export on identical audio
// (recogment/scripts/nemo_parity/dump_nemo_first_chunk.py).
//
// 10.0s = 160,000 samples = 1,000 mel frames = exactly one feed for the
// default streaming geometry (chunk_len 124 + right_context 1, subsampling 8),
// so the dump is one ONNX call with no padding and no cache carry-over.
//
// Usage:
//   cargo run --release --example dump_sortformer_first_chunk --features sortformer -- \
//       <model.onnx> <audio.wav> <out_prefix>

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::sortformer::Sortformer;
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: dump_sortformer_first_chunk <model.onnx> <audio.wav> <out_prefix> [num_samples]");
        std::process::exit(2);
    }
    let (model_path, wav_path, out_prefix) = (&args[1], &args[2], &args[3]);
    // Default 160,000 (one chunk). Phase c passes a multiple of 158,720
    // (124 frames x 1280) to stream whole chunks through the cache policy.
    let num_samples: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(160_000);

    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!("expected 16kHz mono, got {}Hz {}ch", spec.sample_rate, spec.channels).into());
    }
    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    samples.truncate(num_samples);
    if samples.len() < num_samples {
        return Err(format!("need >= {num_samples} samples, wav holds {}", samples.len()).into());
    }

    let mut sf = Sortformer::new(model_path)?; // default config = CPU EP
    let raw = sf.diarize_chunk_raw(&samples)?;
    let (frames, spks) = (raw.predictions.nrows(), raw.predictions.ncols());

    let mut bin = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}.f32"))?);
    for t in 0..frames {
        for s in 0..spks {
            bin.write_all(&raw.predictions[[t, s]].to_le_bytes())?;
        }
    }
    bin.flush()?;
    std::fs::write(
        format!("{out_prefix}.meta.json"),
        format!(
            "{{\"source\": \"parakeet-rs\", \"frames\": {frames}, \"n_mels\": {spks}, \"layout\": \"time_major_f32le\", \"kind\": \"first_chunk_preds\"}}\n"
        ),
    )?;
    println!("{frames} frames x {spks} speakers");
    Ok(())
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("requires --features sortformer");
    std::process::exit(2);
}
