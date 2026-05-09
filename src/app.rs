use std::collections::HashMap;

use anyhow::Result;
use image::RgbImage;

use crate::ctc::Dataset;
use crate::overlay::{compose_frame, compute_centroids, load_labels};

pub struct App {
    pub dataset: Dataset,
    pub frame_idx: usize,
    pub num_frames: usize,
    pub current_image: Option<RgbImage>,
    pub low: f64,
    pub high: f64,
    pub tail_length: usize,
    pub render_error: Option<String>,
    /// Pre-computed centroids per frame: map from track_id to (x, y).
    pub centroids: Vec<HashMap<u32, (f64, f64)>>,
}

impl App {
    pub fn new(dataset: Dataset, low: f64, high: f64, tail_length: usize) -> Result<Self> {
        let num_frames = dataset.num_frames();
        if num_frames == 0 {
            anyhow::bail!("No frames found in dataset");
        }

        // Pre-compute centroids for all frames.
        let mut centroids = Vec::with_capacity(num_frames);
        for idx in 0..num_frames {
            if let Some((_img_path, lbl_path)) = dataset.frame_paths(idx) {
                let (labels, w, h) = load_labels(lbl_path)?;
                centroids.push(compute_centroids(&labels, w, h));
            } else {
                centroids.push(HashMap::new());
            }
        }

        let mut app = App {
            dataset,
            frame_idx: 0,
            num_frames,
            current_image: None,
            low,
            high,
            tail_length,
            render_error: None,
            centroids,
        };
        app.load_frame()?;
        Ok(app)
    }

    pub fn next_frame(&mut self) -> Result<()> {
        if self.frame_idx + 1 < self.num_frames {
            self.frame_idx += 1;
            self.load_frame()?;
        }
        Ok(())
    }

    pub fn prev_frame(&mut self) -> Result<()> {
        if self.frame_idx > 0 {
            self.frame_idx -= 1;
            self.load_frame()?;
        }
        Ok(())
    }

    fn load_frame(&mut self) -> Result<()> {
        if let Some((img_path, lbl_path)) = self.dataset.frame_paths(self.frame_idx) {
            self.current_image = Some(compose_frame(
                img_path,
                lbl_path,
                self.frame_idx as u32,
                &self.dataset.tracks,
                &self.centroids,
                self.low,
                self.high,
                self.tail_length,
            )?);
        }
        Ok(())
    }
}
