// Parity instrument: run the spkcache-compression frame-selection chain on a
// captured preds matrix and print the chosen indices, for comparison against
// NeMo's _compress_spkcache selection on identical input
// (recogment/scripts/nemo_parity/capture_nemo_compress.py).
//
// Usage:
//   cargo run --release --example dump_compress_selection --features sortformer -- \
//       <model.onnx> <preds.f32> <n_frames>
//
// preds.f32: raw little-endian f32, row-major [n_frames][4].
// Output: one line per selected slot: "<frame_index> <disabled 0|1>".

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::Array2;
    use parakeet_rs::sortformer::Sortformer;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: dump_compress_selection <model.onnx> <preds.f32> <n_frames>");
        std::process::exit(2);
    }
    let n_frames: usize = args[3].parse()?;
    let bytes = std::fs::read(&args[2])?;
    if bytes.len() != n_frames * 4 * 4 {
        return Err(format!("expected {} bytes, got {}", n_frames * 16, bytes.len()).into());
    }
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let preds = Array2::from_shape_vec((n_frames, 4), vals)?;

    let sf = Sortformer::new(&args[1])?;
    let (indices, disabled) = sf.debug_compress_selection(&preds);
    for (i, d) in indices.iter().zip(disabled.iter()) {
        println!("{i} {}", u8::from(*d));
    }
    Ok(())
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("requires --features sortformer");
    std::process::exit(2);
}
