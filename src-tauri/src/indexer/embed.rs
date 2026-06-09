use ort::value::TensorRef;
use bytemuck::cast_slice;
use crate::time_block;
use ort::session::Session;

pub fn run_batch(session: &mut Session, flat: &[f32], n: usize) -> Vec<u8> {
    time_block!("neural network inference", {
        let tensor = TensorRef::<f32>::from_array_view((
            [n, 3usize, 224usize, 224usize],
            flat,
        )).unwrap();

        let batch_result: Result<Vec<u8>, _> = session.run(ort::inputs![tensor])
            .map(|outputs| {
                let (_shape, data) = outputs[0].try_extract_tensor::<f32>().unwrap();
                // Normalize each embedding in the batch before casting to bytes
                let stride = data.len() / n;
                let mut normalized: Vec<f32> = Vec::with_capacity(data.len());
                for chunk in data.chunks(stride) {
                    let mut chunk_vec = chunk.to_vec();
                    let norm = chunk_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in chunk_vec.iter_mut() { *v /= norm; }
                    }
                    normalized.extend_from_slice(&chunk_vec);
                }
                cast_slice(normalized.as_slice()).to_vec()
            });

        match batch_result {
            Ok(bytes) => bytes,
            Err(_e) => {
                // If session.run fails, fall back to per-image runs on the same session (no DirectML provider used now)
                let stride = flat.len() / n;
                let mut all_bytes: Vec<u8> = Vec::new();

                for i in 0..n {
                    let chunk = &flat[i * stride..(i + 1) * stride];
                    let t = TensorRef::<f32>::from_array_view((
                        [1usize, 3usize, 224usize, 224usize],
                        chunk,
                    )).unwrap();
                    let out = session.run(ort::inputs![t]).unwrap();
                    let (_shape, data) = out[0].try_extract_tensor::<f32>().unwrap();
                    let mut emb = data.to_vec();
                    let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in emb.iter_mut() { *v /= norm; }
                    }
                    all_bytes.extend_from_slice(cast_slice(emb.as_slice()));
                }
                all_bytes
            }
        }
    })
}
