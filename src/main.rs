use std::io;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use viuer::Config as ViuerConfig;

mod app;
mod ctc;
mod overlay;
mod ui;

use app::App;

#[derive(Parser, Debug)]
#[command(name = "cytui")]
#[command(about = "CTC Dataset TUI Viewer")]
struct Cli {
    /// Path to the image sequence directory (e.g. train/01)
    #[arg(short, long, value_name = "DIR")]
    images: PathBuf,

    /// Path to the annotation directory (e.g. train/01_GT/TRA or a RES folder)
    #[arg(short, long, value_name = "DIR")]
    tracks: PathBuf,

    /// Dump the composed first frame to a PNG file and exit (for testing)
    #[arg(long, value_name = "FILE")]
    dump_png: Option<PathBuf>,

    /// Lower quantile for contrast clipping [0.0, 1.0]
    #[arg(long, default_value_t = 0.001, value_name = "Q")]
    low: f64,

    /// Upper quantile for contrast clipping [0.0, 1.0]
    #[arg(long, default_value_t = 0.999, value_name = "Q")]
    high: f64,
    /// Number of tail segments to draw for tracking history
    #[arg(long, default_value_t = 3, value_name = "N")]
    tail_length: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.low < 0.0 || cli.low >= 1.0 {
        anyhow::bail!("--low must be in [0.0, 1.0)");
    }
    if cli.high <= cli.low || cli.high > 1.0 {
        anyhow::bail!("--high must be in (low, 1.0]");
    }

    // Load dataset
    let dataset = ctc::Dataset::load(&cli.images, &cli.tracks)?;
    if !dataset.is_valid() {
        anyhow::bail!("Dataset is empty or invalid");
    }

    // If --dump-png is provided, render the first frame and exit
    if let Some(out_path) = cli.dump_png {
        let (img_path, lbl_path) = dataset
            .frame_paths(0)
            .expect("dataset should have at least one frame");
        let composed = overlay::compose_frame(
            img_path,
            lbl_path,
            0,
            &dataset.tracks,
            &[],
            cli.low,
            cli.high,
            cli.tail_length,
        )?;
        composed.save(&out_path)?;
        println!("Saved composed frame to {}", out_path.display());
        return Ok(());
    }

    let mut app = App::new(dataset, cli.low, cli.high, cli.tail_length)?;

    // Setup terminal
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(
        &mut stdout,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = run_app(&mut terminal, &mut app);

    // Restore terminal
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    res
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<()> {
    loop {
        let mut viuer_err: Option<String> = None;

        // Draw UI and capture image area
        terminal.draw(|frame| {
            let image_area = ui::draw_ui(frame, app);

            // Render image via viuer within the allocated block
            if let Some(ref img) = app.current_image {
                let dyn_img = image::DynamicImage::ImageRgb8(img.clone());
                let conf = ViuerConfig {
                    x: image_area.x as u16,
                    y: image_area.y as i16,
                    width: Some(image_area.width as u32),
                    height: Some(image_area.height as u32),
                    ..Default::default()
                };
                if let Err(e) = viuer::print(&dyn_img, &conf) {
                    viuer_err = Some(format!("viuer: {e}"));
                }
            }
        })?;

        app.render_error = viuer_err;

        // Handle events
        if event::poll(std::time::Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => break,
                        KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => {
                            app.next_frame()?;
                        }
                        KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => {
                            app.prev_frame()?;
                        }
                        KeyCode::Char('j') | KeyCode::Char('J') => {
                            app.next_frame()?;
                        }
                        KeyCode::Char('k') | KeyCode::Char('K') => {
                            app.prev_frame()?;
                        }
                        _ => {}
                    }
                }
    }

    Ok(())
}
