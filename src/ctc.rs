use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A single track from a CTC man_track.txt or res_track.txt file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Track {
    pub id: u32,
    pub start_frame: u32,
    pub end_frame: u32,
    pub parent_id: u32,
}

/// Container for all CTC dataset metadata for one sequence.
pub struct Dataset {
    /// Sorted list of raw image frame paths, e.g. t000.tif, t001.tif, ...
    pub image_paths: Vec<PathBuf>,
    /// Sorted list of label mask paths (may be GT or RES).
    pub label_paths: Vec<PathBuf>,
    /// Map from track_id to Track.
    pub tracks: HashMap<u32, Track>,
    /// Parent directory name, for display.
    pub name: String,
}

impl Dataset {
    /// Load a CTC sequence from an image directory and an annotation directory.
    ///
    /// `image_dir`: directory containing `t000.tif`, `t001.tif`, etc.
    /// `annot_dir`: directory containing `man_track.txt` and `man_track*.tif` (GT)
    ///              or `res_track.txt` and `mask*.tif` (RES).
    pub fn load(image_dir: &Path, annot_dir: &Path) -> Result<Self> {
        let name = image_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // --- discover images ---
        let mut image_paths: Vec<PathBuf> = fs::read_dir(image_dir)
            .with_context(|| format!("reading image directory: {}", image_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
                    .unwrap_or(false)
            })
            .collect();

        image_paths.sort();

        // --- discover labels ---
        let mut label_paths: Vec<PathBuf> = fs::read_dir(annot_dir)
            .with_context(|| format!("reading annotation directory: {}", annot_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
                    .unwrap_or(false)
            })
            .collect();

        label_paths.sort();

        // --- load tracks ---
        let mut tracks = HashMap::new();
        let track_file_gt = annot_dir.join("man_track.txt");
        let track_file_res = annot_dir.join("res_track.txt");
        let track_file = if track_file_gt.exists() {
            track_file_gt
        } else if track_file_res.exists() {
            track_file_res
        } else {
            anyhow::bail!(
                "No track file found in {} (expected man_track.txt or res_track.txt)",
                annot_dir.display()
            );
        };

        let contents = fs::read_to_string(&track_file)
            .with_context(|| format!("reading track file: {}", track_file.display()))?;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 4 {
                continue;
            }
            let id = parts[0].parse::<u32>().context("track id")?;
            let start_frame = parts[1].parse::<u32>().context("start frame")?;
            let end_frame = parts[2].parse::<u32>().context("end frame")?;
            let parent_id = parts[3].parse::<u32>().context("parent id")?;
            tracks.insert(
                id,
                Track {
                    id,
                    start_frame,
                    end_frame,
                    parent_id,
                },
            );
        }

        Ok(Dataset {
            image_paths,
            label_paths,
            tracks,
            name,
        })
    }

    /// Return the image and label paths for a given frame index.
    pub fn frame_paths(&self, idx: usize) -> Option<(&Path, &Path)> {
        let img = self.image_paths.get(idx)?;
        let lbl = self.label_paths.get(idx)?;
        Some((img, lbl))
    }

    pub fn num_frames(&self) -> usize {
        self.image_paths.len().min(self.label_paths.len())
    }

    pub fn is_valid(&self) -> bool {
        !self.image_paths.is_empty() && !self.label_paths.is_empty()
    }
}
