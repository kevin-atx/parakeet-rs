/*
CoreML smoke test — does the CoreML EP actually EXECUTE this model?

Loading a model under CoreML proves nothing: the EP compiles happily and then
fails per-inference. On macOS 26.5 / M1 Pro the legacy NeuralNetwork format
failed EVERY chunk with:

    Where node '/encoder/layers.10/self_attn/Where_1'
    Status Message: GetElementType is not implemented

...while reporting itself as loaded, so ~11,900 chunks "transcribed" to
nothing. This runs a handful of chunks through and reports how many actually
came back, which is the only question that matters.

Input is synthetic audio: the failure is an operator-conversion gap in the
encoder, so it triggers on any input that reaches inference. No fixtures.

Usage:
  cargo run --release --example coreml_smoke \
    --features multitalker,coreml -- <asr_model_dir> <sortformer.onnx> [cpu]

Pass `cpu` as the third argument to run the same check on the CPU EP, which is
the control: it should report 0 failures.
*/

#[cfg(all(feature = "multitalker", feature = "coreml"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use parakeet_rs::{
        CoreMLComputeUnits, ExecutionConfig, ExecutionProvider, LatencyMode, MultitalkerASR,
    };
    use std::env;

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <asr_model_dir> <sortformer.onnx> [cpu]", args[0]);
        std::process::exit(2);
    }
    let asr_dir = &args[1];
    let sortformer = &args[2];
    let use_cpu = args.get(3).map(|s| s == "cpu").unwrap_or(false);

    let config = if use_cpu {
        println!("EP: CPU (control)");
        None
    } else {
        println!("EP: CoreML / All compute units");
        Some(
            ExecutionConfig::new()
                .with_execution_provider(ExecutionProvider::CoreML)
                .with_coreml_compute_units(CoreMLComputeUnits::All),
        )
    };

    let t0 = std::time::Instant::now();
    let mut model = MultitalkerASR::from_pretrained(asr_dir, sortformer, config)?;
    println!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Static-shape exports bake in ONE chunk size. If it doesn't match the
    // latency mode's chunking the graph rejects the input before any compute,
    // so sweep the modes and report which (if any) the export actually fits.
    let mode = match std::env::var("SMOKE_LATENCY").as_deref() {
        Ok("low") => LatencyMode::Low,
        Ok("very-low") => LatencyMode::VeryLow,
        Ok("ultra") => LatencyMode::Ultra,
        _ => LatencyMode::Normal,
    };
    model.set_latency_mode(mode);
    println!("latency mode   : {mode:?}");

    // Deterministic non-silent input: a 220 Hz tone with a little shaped noise,
    // enough to drive the encoder. Content is irrelevant — reaching inference
    // is the whole point.
    let chunk_samples = model.chunk_audio_samples();
    let mut phase = 0.0f32;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next_chunk = || {
        (0..chunk_samples)
            .map(|_| {
                phase += 2.0 * std::f32::consts::PI * 220.0 / 16000.0;
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let noise = ((seed >> 40) as f32 / 16_777_216.0) - 0.5;
                phase.sin() * 0.25 + noise * 0.02
            })
            .collect::<Vec<f32>>()
    };

    // The daemon calls `transcribe_chunk_with_activity`, not `transcribe_chunk`.
    // Using the other entry point tripped a shape gate (`chunk` got 1000,
    // expected 992) on BOTH EPs, i.e. before the encoder — which would have
    // made this test look like a CoreML verdict when it was a harness bug.
    const CHUNKS: usize = 8;
    let (mut ok, mut failed) = (0usize, 0usize);
    let mut first_error: Option<String> = None;

    let t1 = std::time::Instant::now();
    for i in 0..CHUNKS {
        match model.transcribe_chunk_with_activity(&next_chunk()) {
            Ok(_) => ok += 1,
            Err(e) => {
                failed += 1;
                if first_error.is_none() {
                    first_error = Some(format!("{e}"));
                }
                if i == 0 {
                    eprintln!("first chunk failed: {e}");
                }
            }
        }
    }
    let elapsed = t1.elapsed().as_secs_f32();

    println!("\nchunks OK      : {ok}/{CHUNKS}");
    println!("chunks FAILED  : {failed}/{CHUNKS}");
    println!(
        "time           : {elapsed:.1}s total, {:.2}s per chunk",
        elapsed / CHUNKS as f32
    );
    if let Some(e) = &first_error {
        println!("first error    : {e}");
    }

    if failed > 0 {
        println!("\nVERDICT: this EP cannot execute the model.");
        std::process::exit(1);
    }
    println!("\nVERDICT: this EP executes the model.");
    Ok(())
}

#[cfg(not(all(feature = "multitalker", feature = "coreml")))]
fn main() {
    eprintln!("rebuild with --features multitalker,coreml");
    std::process::exit(2);
}
