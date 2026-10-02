/*
Dump raw streaming-diarizer predictions for a 16 kHz mono f32le file, fed the
way the multitalker pipeline feeds it: `feed_raw` in 30 s pieces at the
model's native stride. Output: f32le rows of the model's speaker count (4 for Sortformer v2, 8 for
Nemotron 3), one per 80 ms frame from sample 0. Used by recogment's
scripts/quality/diar_bakeoff to compare diarizers on the same audio.

Usage:
cargo run --release --example dump_diar_raw --features sortformer -- MODEL.onnx IN.f32 OUT.f32
*/

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("dump_diar_raw requires the 'sortformer' feature");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        use parakeet_rs::nemotron_diar::NemotronDiarizer;
        use parakeet_rs::sortformer::{DiarizationConfig, RawDiarizationPredictions, Sortformer};
        use std::io::Write;

        // Either diarizer, chosen by the model's own metadata: a Nemotron
        // export declares `model_family`, a Sortformer v2 export does not.
        enum Any {
            V2(Sortformer),
            Nemotron(NemotronDiarizer),
        }
        impl Any {
            fn feed_raw(&mut self, a: &[f32]) -> parakeet_rs::Result<RawDiarizationPredictions> {
                match self {
                    Any::V2(d) => d.feed_raw(a),
                    Any::Nemotron(d) => d.feed_raw(a),
                }
            }
        }

        let args: Vec<String> = std::env::args().collect();
        if args.len() != 4 {
            return Err("usage: dump_diar_raw MODEL.onnx IN.f32 OUT.f32".into());
        }
        let bytes = std::fs::read(&args[2])?;
        let mut audio: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Trailing silence so the last real stride completes.
        audio.extend(std::iter::repeat_n(0.0, 16_000 * 11));

        let mut sf = match NemotronDiarizer::new(&args[1], None) {
            Ok(n) => {
                eprintln!(
                    "nemotron: {} speakers, chunk_len={} right_context={} fifo={} update={}",
                    n.num_speakers(), n.cfg.chunk_len, n.cfg.right_context, n.cfg.fifo_len, n.cfg.spkcache_update_period
                );
                Any::Nemotron(n)
            }
            Err(_) => {
                // The post-processing config is unused: feed_raw returns raw sigmoids.
                let v2 = Sortformer::with_config(&args[1], None, DiarizationConfig::callhome())?;
                eprintln!("sortformer v2: chunk_len={} right_context={}", v2.chunk_len, v2.right_context);
                Any::V2(v2)
            }
        };
        let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
        let t = std::time::Instant::now();
        let mut frames = 0usize;
        for piece in audio.chunks(16_000 * 30) {
            let p = sf.feed_raw(piece)?;
            for row in p.predictions.rows() {
                for v in row.iter() {
                    out.write_all(&v.to_le_bytes())?;
                }
                frames += 1;
            }
        }
        out.flush()?;
        eprintln!("{frames} frames in {:.1}s", t.elapsed().as_secs_f64());
        Ok(())
    }
}
