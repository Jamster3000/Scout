use fast_image_resize::{images::Image, PixelType, Resizer, ResizeOptions, ResizeAlg, FilterType as FirFilter};
use std::sync::LazyLock;
use crate::time_block;

const IMAGE_SIZE: u32 = 224;
const JPEG_QUALITY: u8 = 60;

static THUMB_RESIZE_OPTS: LazyLock<ResizeOptions> = LazyLock::new(|| {
    ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FirFilter::Box))
});

static EMBED_RESIZE_OPTS: LazyLock<ResizeOptions> = LazyLock::new(|| {
    ResizeOptions::new().resize_alg(ResizeAlg::Nearest)
});

thread_local! {
    pub static RESIZER: std::cell::RefCell<Resizer> = std::cell::RefCell::new(Resizer::new());
    pub static COMPRESSOR: std::cell::RefCell<turbojpeg::Compressor> = std::cell::RefCell::new(
        turbojpeg::Compressor::new().expect("failed to create turbojpeg compressor")
    );
    static CLIP_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(
        vec![0f32; 3 * 224 * 224]
    );
}

#[derive(Clone, Copy)]
pub struct NormConfig {
    pub mean: [f32; 3],
    pub std:  [f32; 3],
}

impl NormConfig {
    pub fn clip() -> Self {
        Self {
            mean: [0.48145466, 0.4578275,  0.40821073],
            std:  [0.26862954, 0.26130258, 0.27577711],
        }
    }

    pub fn mobileclip() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std:  [1.0, 1.0, 1.0],
        }
    }

    pub fn from_model_family(family: &str) -> Self {
        if family.to_lowercase().contains("mobileclip") {
            Self::mobileclip()
        } else {
            Self::clip()
        }
    }
}

pub fn generate_thumbnail(path: &str, size: u32) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;

    let img = if data.starts_with(&[0xFF, 0xD8]) {
        time_block!("decode JPEG", {
            let mut decompressor = turbojpeg::Decompressor::new().ok()?;
            let header = decompressor.read_header(&data).ok()?;
            let w = header.width;
            let h = header.height;
            let mut buf = vec![0u8; w * h * 3];
            let img = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: w,
                pitch: w * 3,
                height: h,
                format: turbojpeg::PixelFormat::RGB,
            };
            decompressor.decompress(&data, img).ok()?;
            image::RgbImage::from_raw(w as u32, h as u32, buf)?
        })
    } else {
        time_block!("decode other format (generic)", {
            image::load_from_memory(&data).ok()?.into_rgb8()
        })
    };

    let (w, h) = img.dimensions();
    encode_thumbnail(img.into_raw(), w, h, size)
}

pub fn resize_and_embed(
    rgb: image::RgbImage,
    w: u32,
    h: u32,
    thumbnail_size: u32,
    filename: &str,
    norm: NormConfig,
) -> Option<(Vec<f32>, Vec<u8>)> {
    time_block!("resizer pipeline total", {
        RESIZER.with(|r| {
            let mut resizer = r.borrow_mut();

            let src_img = time_block!("prepare src_img", {
                let pre_scale_limit = (thumbnail_size.max(IMAGE_SIZE) as f32 * 1.5).round() as u32;

                if w > pre_scale_limit || h > pre_scale_limit {
                    let scale = (pre_scale_limit as f32 / w.max(h) as f32).min(1.0);
                    let pre_w = ((w as f32 * scale) as u32).max(1);
                    let pre_h = ((h as f32 * scale) as u32).max(1);

                    let src = match Image::from_vec_u8(w, h, rgb.into_raw(), PixelType::U8x3) {
                        Ok(img) => img,
                        Err(e) => {
                            eprintln!("[error] failed to create source image buffer for {:?}: {}", filename, e);
                            return None;
                        }
                    };
                    let mut pre = Image::new(pre_w, pre_h, PixelType::U8x3);
                    match resizer.resize(&src, &mut pre, &*EMBED_RESIZE_OPTS) {
                        Ok(_) => pre,
                        Err(e) => {
                            eprintln!("[error] failed to pre-resize image {:?}: {}", filename, e);
                            return None;
                        }
                    }
                } else {
                    match Image::from_vec_u8(w, h, rgb.into_raw(), PixelType::U8x3) {
                        Ok(img) => img,
                        Err(e) => {
                            eprintln!("[error] failed to create source image buffer for {:?}: {}", filename, e);
                            return None;
                        }
                    }
                }
            });

            let mut thumb_dst = Image::new(thumbnail_size, thumbnail_size, PixelType::U8x3);
            time_block!("resize thumbnail", {
                if let Err(e) = resizer.resize(&src_img, &mut thumb_dst, &*THUMB_RESIZE_OPTS) {
                    eprintln!("[error] failed to resize thumbnail for {:?}: {}", filename, e);
                    return None;
                }
            });

            let embed_raw: Vec<u8> = if thumbnail_size == IMAGE_SIZE {
                thumb_dst.into_vec()
            } else {
                let mut dst = Image::new(IMAGE_SIZE, IMAGE_SIZE, PixelType::U8x3);
                time_block!("resize embedding", {
                    if let Err(e) = resizer.resize(&src_img, &mut dst, &*EMBED_RESIZE_OPTS) {
                        eprintln!("[error] failed to resize embedding image for {:?}: {}", filename, e);
                        return None;
                    }
                });
                dst.into_vec()
            };

            let chw = normalize_for_model(&embed_raw, norm);

            let thumb_buf = time_block!("encode thumbnail", {
                encode_thumbnail(embed_raw, thumbnail_size, thumbnail_size, thumbnail_size)?
            });

            Some((chw, thumb_buf))
        })
    })
}

pub fn encode_thumbnail(raw_pixels: Vec<u8>, w: u32, h: u32, _size: u32) -> Option<Vec<u8>> {
    let width = w as usize;
    let height = h as usize;
    let img = turbojpeg::Image {
        pixels: raw_pixels.as_slice(),
        width,
        pitch: width * 3,
        height,
        format: turbojpeg::PixelFormat::RGB,
    };
    COMPRESSOR.with(|c| {
        let mut compressor = c.borrow_mut();
        compressor.set_quality(JPEG_QUALITY as i32).ok()?;
        compressor.compress_to_vec(img).ok()
    })
}

fn normalize_for_model(raw: &[u8], norm: NormConfig) -> Vec<f32> {
    const PLANAR_STRIDE: usize = 224 * 224;

    // Precompute: pixel_out = pixel_u8 * (1 / (255 * std)) + (-mean / std)
    let div_factors = [
        1.0 / (255.0 * norm.std[0]),
        1.0 / (255.0 * norm.std[1]),
        1.0 / (255.0 * norm.std[2]),
    ];
    let offsets = [
        -norm.mean[0] / norm.std[0],
        -norm.mean[1] / norm.std[1],
        -norm.mean[2] / norm.std[2],
    ];

    CLIP_BUF.with(|buf| {
        let mut chw = buf.borrow_mut();
        let (r_slice, rest) = chw.split_at_mut(PLANAR_STRIDE);
        let (g_slice, b_slice) = rest.split_at_mut(PLANAR_STRIDE);

        for (i, px) in raw.chunks_exact(3).enumerate() {
            r_slice[i] = (px[0] as f32).mul_add(div_factors[0], offsets[0]);
            g_slice[i] = (px[1] as f32).mul_add(div_factors[1], offsets[1]);
            b_slice[i] = (px[2] as f32).mul_add(div_factors[2], offsets[2]);
        }

        chw.clone()
    })
}

pub fn resize_for_embedding_only(
    rgb: image::RgbImage,
    w: u32,
    h: u32,
    filename: &str,
    norm: NormConfig,
) -> Option<Vec<f32>> {
    RESIZER.with(|r| {
        let mut resizer = r.borrow_mut();

        let pre_scale_limit = (IMAGE_SIZE as f32 * 1.5).round() as u32;

        let src_img = if w > pre_scale_limit || h > pre_scale_limit {
            let scale = (pre_scale_limit as f32 / w.max(h) as f32).min(1.0);
            let pre_w = ((w as f32 * scale) as u32).max(1);
            let pre_h = ((h as f32 * scale) as u32).max(1);

            let src = match Image::from_vec_u8(w, h, rgb.into_raw(), PixelType::U8x3) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("[error] failed to create source image buffer for {:?}: {}", filename, e);
                    return None;
                }
            };
            let mut pre = Image::new(pre_w, pre_h, PixelType::U8x3);
            match resizer.resize(&src, &mut pre, &*EMBED_RESIZE_OPTS) {
                Ok(_) => pre,
                Err(e) => {
                    eprintln!("[error] failed to pre-resize image {:?}: {}", filename, e);
                    return None;
                }
            }
        } else {
            match Image::from_vec_u8(w, h, rgb.into_raw(), PixelType::U8x3) {
                Ok(img) => img,
                Err(e) => {
                    eprintln!("[error] failed to create source image buffer for {:?}: {}", filename, e);
                    return None;
                }
            }
        };

        let mut dst = Image::new(IMAGE_SIZE, IMAGE_SIZE, PixelType::U8x3);
        if let Err(e) = resizer.resize(&src_img, &mut dst, &*EMBED_RESIZE_OPTS) {
            eprintln!("[error] failed to resize for embedding {:?}: {}", filename, e);
            return None;
        }

        Some(normalize_for_model(&dst.into_vec(), norm))
    })
}
 
