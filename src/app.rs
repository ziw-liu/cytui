use anyhow::Result;
use image::RgbImage;

use crate::ctc::Dataset;
use crate::overlay::compose_frame;

pub struct App {
    pub dataset: Dataset,
    pub frame_idx: usize,
    pub num_frames: usize,
    pub current_image: Option<RgbImage>,
    pub low: f64,
    pub high: f64,
    pub render_error: Option<String>,
}

impl App {
    pub fn new(dataset: Dataset, low: f64, high: f64) -> Result<Self> {
        let num_frames = dataset.num_frames();
        if num_frames == 0 {
            anyhow::bail!("No frames found in dataset");
        }
        let mut app = App {
            dataset,
            frame_idx: 0,
            num_frames,
            current_image: None,
            low,
            high,
            render_error: None,
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
                self.low,
                self.high,
            )?);
        }
        Ok(())
    }
}
