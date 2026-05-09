use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;

/// Draw the full UI.
pub fn draw_ui(frame: &mut Frame, app: &App) -> Rect {
    let area = frame.area();

    // Vertical split: image area + status bar at the bottom
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),   // image area
            Constraint::Length(1), // status bar
        ])
        .split(area);

    draw_status_bar(frame, app, chunks[1]);

    chunks[0]
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![
        Span::styled(
            format!("  {}  ", app.dataset.name),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "Frame {}/{} | Tracks: {} | {} | Contrast [{:.3}, {:.3}]",
            app.frame_idx + 1,
            app.num_frames,
            app.dataset.tracks.len(),
            annotation_type(&app.dataset),
            app.low,
            app.high,
        )),
        Span::raw("  |  "),
        Span::raw("[\u{2190}/\u{2192}] or [J/K] Navigate"),
        Span::raw("  |  "),
        Span::raw("[Q] Quit  "),
    ];

    if let Some(ref err) = app.render_error {
        spans.push(Span::raw("  |  "));
        spans.push(Span::styled(
            format!("Render error: {err}"),
            Style::default().fg(Color::Red),
        ));
    }

    let status = Paragraph::new(Line::from(spans)).style(Style::default().fg(Color::Gray));
    frame.render_widget(status, area);
}

fn annotation_type(dataset: &crate::ctc::Dataset) -> &'static str {
    if dataset
        .label_paths
        .first()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|n| n.starts_with("man_"))
        .unwrap_or(false)
    {
        "GT"
    } else {
        "RES"
    }
}
