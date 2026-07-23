/*
EP A/B on REAL audio — does this (model dir × execution provider) combination
actually TRANSCRIBE?

coreml_smoke proves a provider can EXECUTE the graph (per-chunk Ok), which is
necessary but nowhere near sufficient: on 2026-07-22 the MLProgram-compiled
static ASR executed every chunk cleanly and emitted ZERO words on real speech.
This example is the content-level check: feed a real 16 kHz WAV through the
same entry point the daemon uses and report the transcript + word count.

Usage:
  cargo run --release --example ep_ab --features multitalker,coreml -- \
    <audio.wav> <asr_model_dir> <sortformer.onnx> <ep>

  <ep> is one of: cpu | coreml-all | coreml-ane | coreml-gpu | coreml-cpuonly

The last line is machine-parseable: `WORDS: <n>`. Exit code 0 always (the
caller judges the count); non-zero only on hard errors.
*/

#[cfg(all(feature = "multitalker", feature = "coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{CoreMLComputeUnits, ExecutionConfig, ExecutionProvider, MultitalkerASR};
    use std::env;

    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "Usage: {} <audio.wav> <asr_model_dir> <sortformer.onnx> <cpu|coreml-all|coreml-ane|coreml-gpu|coreml-cpuonly>",
            args[0]
        );
        std::process::exit(2);
    }
    let audio_path = &args[1];
    let asr_dir = &args[2];
    let sortformer = &args[3];
    let ep = args[4].as_str();

    let config = match ep {
        "cpu" => None,
        "coreml-all" | "coreml-ane" | "coreml-gpu" | "coreml-cpuonly" => {
            let units = match ep {
                "coreml-all" => CoreMLComputeUnits::All,
                "coreml-ane" => CoreMLComputeUnits::CpuAndNeuralEngine,
                "coreml-gpu" => CoreMLComputeUnits::CpuAndGpu,
                _ => CoreMLComputeUnits::CpuOnly,
            };
            Some(
                ExecutionConfig::new()
                    .with_execution_provider(ExecutionProvider::CoreML)
                    .with_coreml_compute_units(units),
            )
        }
        other => {
            eprintln!("unknown ep: {other}");
            std::process::exit(2);
        }
    };
    println!("EP: {ep}");
    println!("ASR dir: {asr_dir}");

    // Load 16 kHz mono WAV
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 {
        return Err(format!("Expected 16kHz, got {}Hz", spec.sample_rate).into());
    }
    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if spec.channels > 1 {
        audio = audio
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect();
    }
    let duration = audio.len() as f32 / 16000.0;

    let t0 = std::time::Instant::now();
    let mut model = MultitalkerASR::from_pretrained(asr_dir, sortformer, config)?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Same entry point as the daemon: transcribe_chunk_with_activity.
    let chunk_samples = model.chunk_audio_samples();
    let t1 = std::time::Instant::now();
    let mut chunks = 0usize;
    for chunk in audio.chunks(chunk_samples) {
        let chunk_vec = if chunk.len() < chunk_samples {
            let mut p = chunk.to_vec();
            p.resize(chunk_samples, 0.0);
            p
        } else {
            chunk.to_vec()
        };
        model.transcribe_chunk_with_activity(&chunk_vec)?;
        chunks += 1;
    }
    // Flush with silence so trailing tokens decode.
    let flush = vec![0.0f32; chunk_samples];
    for _ in 0..3 {
        model.transcribe_chunk_with_activity(&flush)?;
    }
    let elapsed = t1.elapsed().as_secs_f32();

    let mut total_words = 0usize;
    for t in model.get_transcripts() {
        println!("--- speaker {} ({} words) ---", t.speaker_id, t.words.len());
        println!("{}", t.text);
        total_words += t.words.len();
    }
    println!(
        "\naudio {duration:.1}s, {chunks} chunks, inference {elapsed:.1}s ({:.2}x realtime)",
        duration / elapsed
    );
    println!("WORDS: {total_words}");
    Ok(())
}

#[cfg(not(all(feature = "multitalker", feature = "coreml")))]
fn main() {
    eprintln!("rebuild with --features multitalker,coreml");
    std::process::exit(2);
}
