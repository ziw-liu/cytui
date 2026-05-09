use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use imageproc::contours::find_contours;
use ndarray::Array1;
use ndarray_stats::{interpolate::Nearest, Quantile1dExt};
use noisy_float::types::n64;

/// Convert track_id into a distinct, deterministic color.
pub fn color_for_id(id: u32) -> Rgb<u8> {
    // Simple HSL-to-RGB. Hue is hashed from id, saturation and lightness fixed.
    let hue = ((id.wrapping_mul(137) ^ id.wrapping_mul(269)) % 360) as f32;
    hsl_to_rgb(hue, 0.85, 0.55)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Rgb<u8> {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hh = h / 60.0;
    let x = c * (1.0 - (hh % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r1, g1, b1) = match hh.floor() as i32 {
        0 => (c, x, 0.0f32),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Rgb([
        ((r1 + m).clamp(0.0, 1.0) * 255.0) as u8,
        ((g1 + m).clamp(0.0, 1.0) * 255.0) as u8,
        ((b1 + m).clamp(0.0, 1.0) * 255.0) as u8,
    ])
}

/// Load a 16-bit grayscale CTC image and normalize it to RgbImage using
/// quantile-based clipping for contrast enhancement.
pub fn load_image(path: &Path, low_q: f64, high_q: f64) -> Result<RgbImage> {
    let dyn_img = image::open(path)?;
    let gray = dyn_img.to_luma16();
    let (w, h) = gray.dimensions();

    // Compute quantiles via ndarray-stats (O(n) selection, no full sort).
    let pixels = gray.as_raw().to_owned();
    let mut arr = Array1::from(pixels);
    let low_val = arr.quantile_mut(n64(low_q), &Nearest)?;
    let high_val = arr.quantile_mut(n64(high_q), &Nearest)?;

    let range = (high_val.saturating_sub(low_val)).max(1) as f64;

    let mut rgb = RgbImage::new(w, h);
    for (x, y, p) in gray.enumerate_pixels() {
        let v = ((p.0[0].saturating_sub(low_val)) as f64 / range * 255.0).clamp(0.0, 255.0) as u8;
        rgb.put_pixel(x, y, Rgb([v, v, v]));
    }
    Ok(rgb)
}

/// Load a 16-bit label mask and return a Vec of its raw u16 values.
pub fn load_labels(path: &Path) -> Result<(Vec<u16>, u32, u32)> {
    let dyn_img = image::open(path)?;
    let gray = dyn_img.to_luma16();
    let (w, h) = gray.dimensions();
    let pixels: Vec<u16> = gray.pixels().map(|p| p.0[0]).collect();
    Ok((pixels, w, h))
}

/// Overlay colored contours for each unique label onto the base image.
pub fn overlay_contours(
    base: &mut RgbImage,
    labels: &[u16],
    width: u32,
    height: u32,
) {
    // Collect unique non-zero labels
    let mut unique = HashSet::new();
    for &v in labels {
        if v != 0 {
            unique.insert(v);
        }
    }

    for label in unique {
        let id = label as u32;
        let color = color_for_id(id);

        // Build binary mask for this label
        let mut mask: GrayImage = ImageBuffer::new(width, height);
        for (idx, &val) in labels.iter().enumerate() {
            if val == label {
                let x = (idx % width as usize) as u32;
                let y = (idx / width as usize) as u32;
                mask.put_pixel(x, y, Luma([255u8]));
            }
        }

        let contours: Vec<imageproc::contours::Contour<u32>> = find_contours(&mask);
        for contour in contours {
            let pts = &contour.points;
            if pts.len() < 2 {
                continue;
            }
            // Draw the contour as a closed polyline
            for i in 0..pts.len() {
                let p1 = &pts[i];
                let p2 = &pts[(i + 1) % pts.len()];
                draw_line_safe(base, p1.x, p1.y, p2.x, p2.y, color);
            }
        }
    }
}

fn draw_line_safe(
    img: &mut RgbImage,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    color: Rgb<u8>,
) {
    // Bresenham-ish line drawing with bounds checking
    let (w, h) = img.dimensions();
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0 as i32;
    let mut y = y0 as i32;

    loop {
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 as i32 && y == y1 as i32 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Compose the final overlay image for a single frame.
pub fn compose_frame(
    image_path: &Path,
    label_path: &Path,
    low_q: f64,
    high_q: f64,
) -> Result<RgbImage> {
    let mut base = load_image(image_path, low_q, high_q)?;
    let (labels, w, h) = load_labels(label_path)?;
    overlay_contours(&mut base, &labels, w, h);
    Ok(base)
}
