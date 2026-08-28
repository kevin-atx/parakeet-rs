// Parity instrument: dump the Sortformer mel front end's output for a wav so it
// can be compared bit-for-bit against NeMo's AudioToMelSpectrogramPreprocessor
// on identical audio (see recogment/scripts/nemo_parity/).
//
// Usage:
//   cargo run --release --example dump_sortformer_mel --features sortformer -- \
//       <audio.wav> <out_prefix> [max_secs]
//
// Writes:
//   <out_prefix>.f32        raw little-endian f32, row-major [time][mel]
//   <out_prefix>.meta.json  shape + front-end constants for the comparator
//
// The wav must be 16 kHz mono PCM (the model's native format) — the instrument
// refuses anything else rather than resample, because resampler choice would
// become part of what's being measured.

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::sortformer::Sortformer;
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dump_sortformer_mel <audio.wav> <out_prefix> [max_secs]");
        std::process::exit(2);
    }
    let wav_path = &args[1];
    let out_prefix = &args[2];
    let max_secs: Option<f64> = args.get(3).map(|s| s.parse()).transpose()?;

    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16_000 || spec.channels != 1 {
        return Err(format!(
            "expected 16kHz mono, got {}Hz {}ch — resample outside the instrument",
            spec.sample_rate, spec.channels
        )
        .into());
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
    if let Some(secs) = max_secs {
        let n = ((secs * 16_000.0) as usize).min(samples.len());
        samples.truncate(n);
    }

    let mel = Sortformer::mel_features_standalone(&samples)?; // (1, T, D)
    let (frames, n_mels) = (mel.shape()[1], mel.shape()[2]);

    let mut bin = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}.f32"))?);
    for t in 0..frames {
        for d in 0..n_mels {
            bin.write_all(&mel[[0, t, d]].to_le_bytes())?;
        }
    }
    bin.flush()?;

    // Also dump the filterbank so the comparator can isolate stage 4 (mel
    // weights) from stages 1-3 (preemphasis/STFT).
    let fb = Sortformer::mel_filterbank_standalone();
    let mut fb_out = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}.fb.f32"))?);
    for m in 0..fb.shape()[0] {
        for k in 0..fb.shape()[1] {
            fb_out.write_all(&fb[[m, k]].to_le_bytes())?;
        }
    }
    fb_out.flush()?;

    let meta = format!(
        concat!(
            "{{\"source\": \"parakeet-rs\", \"frames\": {}, \"n_mels\": {}, ",
            "\"sample_rate\": 16000, \"num_samples\": {}, \"layout\": \"time_major_f32le\"}}\n"
        ),
        frames,
        n_mels,
        samples.len()
    );
    std::fs::write(format!("{out_prefix}.meta.json"), meta)?;

    println!("{frames} frames x {n_mels} mels from {} samples", samples.len());
    Ok(())
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("requires --features sortformer");
    std::process::exit(2);
}
