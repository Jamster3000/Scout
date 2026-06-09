use std::path::Path;
use image::{ImageDecoder, ImageReader, DynamicImage};
use crate::{time_block};
use super::resize::{self, NormConfig};

pub fn load_image(
    path: &Path,
    thumbnail_size: u32,
    norm: NormConfig,
) -> Option<(Vec<f32>, Vec<u8>, f32, Option<Vec<u8>>)> {
    match load_image_inner(path, thumbnail_size, norm) {
        Some(v) => Some(v),
        None => {
            let filename = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_else(|| "unknown".into());
            eprintln!("[skip] {} - check error messages above for details", filename);
            None
        }
    }
}

fn load_image_inner(
    path: &Path,
    thumbnail_size: u32,
    norm: NormConfig,
) -> Option<(Vec<f32>, Vec<u8>, f32, Option<Vec<u8>>)> {
    let filename = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_else(|| "unknown".into());

    let data = std::fs::read(path).ok()?;

    let rgb = if data.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg(&data, &filename)?
    } else {
        decode_other(&data, &filename)?
    };

    let (w, h) = rgb.dimensions();
    let aspect_ratio = w as f32 / h as f32;

    let result = resize::resize_and_embed(rgb, w, h, thumbnail_size, &filename, norm);
    result.map(|(chw, thumb_buf)| (chw, thumb_buf, aspect_ratio, None))
}

fn decode_jpeg(data: &[u8], filename: &str) -> Option<image::RgbImage> {
    time_block!("decode JPEG (turbojpeg)", {
        // Read EXIF orientation cheaply from raw bytes without a full decoder
        let orientation = exif::Reader::new()
            .read_raw(data.to_vec())
            .ok()
            .and_then(|exif| {
                exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
                    .and_then(|f| f.value.get_uint(0))
                    .map(|v| match v {
                        2 => image::metadata::Orientation::FlipHorizontal,
                        3 => image::metadata::Orientation::Rotate180,
                        4 => image::metadata::Orientation::FlipVertical,
                        5 => image::metadata::Orientation::Rotate90FlipH,
                        6 => image::metadata::Orientation::Rotate270,
                        7 => image::metadata::Orientation::Rotate270FlipH,
                        8 => image::metadata::Orientation::Rotate90,
                        _ => image::metadata::Orientation::NoTransforms,
                    })
            })
            .unwrap_or(image::metadata::Orientation::NoTransforms);

        let turbojpeg_result = (|| -> Option<image::RgbImage> {
            let mut decompressor = turbojpeg::Decompressor::new().ok()?;
            let header = decompressor.read_header(data).ok()?;
            let max_dim = header.width.max(header.height);
            let scaling = if max_dim >= 1344 {
                turbojpeg::ScalingFactor::ONE_QUARTER
            } else if max_dim >= 672 {
                turbojpeg::ScalingFactor::ONE_HALF
            } else {
                turbojpeg::ScalingFactor::ONE
            };
            decompressor.set_scaling_factor(scaling).ok()?;
            let scaled = header.scaled(scaling);
            let mut buf = vec![0u8; scaled.width * scaled.height * 3];
            let img = turbojpeg::Image {
                pixels: buf.as_mut_slice(),
                width: scaled.width,
                pitch: scaled.width * 3,
                height: scaled.height,
                format: turbojpeg::PixelFormat::RGB,
            };
            decompressor.decompress(data, img).ok()?;
            image::RgbImage::from_raw(scaled.width as u32, scaled.height as u32, buf)
        })();

        match turbojpeg_result {
            Some(decoded) => {
                let mut dyn_img = image::DynamicImage::ImageRgb8(decoded);
                dyn_img.apply_orientation(orientation);
                Some(dyn_img.into_rgb8())
            }
            None => {
                match image::load_from_memory(data) {
                    Ok(mut dyn_img) => {
                        dyn_img.apply_orientation(orientation);
                        Some(dyn_img.into_rgb8())
                    }
                    Err(e) => {
                        eprintln!("[error] all JPEG decoders failed for {:?}: {}", filename, e);
                        None
                    }
                }
            }
        }
    })
}

fn decode_other(data: &[u8], filename: &str) -> Option<image::RgbImage> {
    time_block!("decode other format", {
        let reader = match ImageReader::new(std::io::Cursor::new(data)).with_guessed_format() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[error] failed to determine image format for {:?}: {}", filename, e);
                return None;
            }
        };

        let mut decoder = match reader.into_decoder() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[error] failed to create decoder for {:?}: {}", filename, e);
                return None;
            }
        };

        let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);

        let mut dyn_img = match DynamicImage::from_decoder(decoder) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("[error] failed to decode image content for {:?}: {}", filename, e);
                return None;
            }
        };

        dyn_img.apply_orientation(orientation);
        Some(dyn_img.into_rgb8())
    })
}

pub fn load_image_for_embedding(
    path: &Path,
    norm: NormConfig,
) -> Option<Vec<f32>> {
    let filename = path.file_name().map(|n| n.to_string_lossy()).unwrap_or_else(|| "unknown".into());
    let data = std::fs::read(path).ok()?;

    let rgb = if data.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg(&data, &filename)?
    } else {
        decode_other(&data, &filename)?
    };

    let (w, h) = rgb.dimensions();
    resize::resize_for_embedding_only(rgb, w, h, &filename, norm)
}
    