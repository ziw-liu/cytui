use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;

/// Draw the full UI.
pub fn draw_ui(frame: &mut Frame, app: &App) -> Rect {
    let area = frame.area();

    // Main vertical split: header, image, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header bar
            Constraint::Min(10),   // image area
            Constraint::Length(1), // footer bar
        ])
        .split(area);

    draw_header(frame, app, chunks[0]);
    let image_area = chunks[1];
    draw_footer(frame, app, chunks[2]);

    image_area
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let header_text = Line::from(vec![
        Span::styled(
            format!("  {}  ", app.dataset.name),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            "Frame {}/{} | Tracks: {} | Annotation: {} | Contrast [{:.3}, {:.3}]",
            app.frame_idx + 1,
            app.num_frames,
            app.dataset.tracks.len(),
            annotation_type(&app.dataset),
            app.low,
            app.high,
        )),
    ]);
    let header = Paragraph::new(header_text);
    frame.render_widget(header, area);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![
        Span::raw("  [←/→] or [J/K] Navigate"),
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

    let footer = Paragraph::new(Line::from(spans)).style(Style::default().fg(Color::Gray));
    frame.render_widget(footer, area);
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
