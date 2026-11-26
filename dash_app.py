"""Dash interface for the simulated urban irrigation diagnosis prototype."""

from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from html import escape as escape_html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update
from flask import Response

from diagnosis import DEFAULT_MODEL_PATH, load_model, predict_records
from schedule_aware_threshold import predict_thresholds


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "outputs" / "labeled_fault_scenarios.csv"
ALL_SCHEDULE_CV_PATH = ROOT / "outputs" / "all_schedules_fault_cv_metrics.json"
MAP_PATH = ROOT / "assets" / "uow_dubai_context.png"
ZONES = ["Zone A", "Zone B", "Zone C"]
ZONE_KEYS = {zone: zone.replace(" ", "") for zone in ZONES}
PAGES = ["Study area", "Pipeline view", "Watering data", "Method comparison", "Live diagnosis"]
PAGE_ICONS = {"Study area": "⌖", "Pipeline view": "⌁", "Watering data": "◉", "Method comparison": "▥", "Live diagnosis": "⌕"}
COLORS = {
    "blue": "#005f73", "blue2": "#2a7f87", "cyan": "#49a0a5", "ink": "#17343a",
    "muted": "#697b79", "border": "#dadbcb", "rose": "#d95d70", "gold": "#b9ad79",
    "white": "#ffffff", "light_bg": "#ffffff", "dark_bg": "#101e22", "dark_card": "#17292d",
}

app = Dash(__name__, title="Irrigation Twin", assets_folder="assets", suppress_callback_exceptions=True)
server = app.server


@server.get("/testing-report")
def serve_testing_report():
    """Render the current validation report as a readable web page."""
    report_path = ROOT / "outputs" / "TESTING_VALIDATION_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    rendered_report = _render_report_markdown(report)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#005f73">
  <title>Testing, Validation and Iteration Report · Irrigation Twin</title>
  <link rel="stylesheet" href="/assets/report.css">
</head>
<body>
  <header class="report-toolbar">
    <a class="report-brand" href="/">Irrigation Twin</a>
    <span class="toolbar-title">Testing, validation and iteration</span>
    <div class="toolbar-actions">
      <button type="button" onclick="window.print()">Print / Save as PDF</button>
      <a href="/">Back to dashboard</a>
    </div>
  </header>
  <main class="report-document">{rendered_report}</main>
  <footer class="report-footer">Irrigation Twin · Validation evidence for the simulated diagnosis prototype</footer>
</body>
</html>"""
    return Response(page, mimetype="text/html")


def _inline_report_markdown(text: str) -> str:
    """Convert the small inline Markdown subset used by the report safely."""
    safe = escape_html(text, quote=True)
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", safe)
    return safe


def _render_report_markdown(markdown: str) -> str:
    """Render report headings, paragraphs, lists, and tables without dependencies."""
    lines = markdown.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{_inline_report_markdown(' '.join(line.strip() for line in paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            items = "".join(f"<li>{_inline_report_markdown(item)}</li>" for item in list_items)
            blocks.append(f"<ul>{items}</ul>")
            list_items.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush_paragraph()
            flush_list()
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{_inline_report_markdown(heading.group(2))}</h{level}>")
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            flush_paragraph()
            flush_list()
            rows = [line]
            i += 2  # header plus Markdown separator
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            parsed = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
            header = "".join(f"<th>{_inline_report_markdown(cell)}</th>" for cell in parsed[0])
            body = "".join("<tr>" + "".join(f"<td>{_inline_report_markdown(cell)}</td>" for cell in row) + "</tr>" for row in parsed[1:])
            blocks.append(f"<div class=\"report-table-wrap\"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>")
            continue
        if line.startswith("- "):
            flush_paragraph()
            list_items.append(line[2:].strip())
            i += 1
            continue
        flush_list()
        paragraph.append(line)
        i += 1

    flush_paragraph()
    flush_list()
    return "\n".join(blocks)


@lru_cache(maxsize=1)
def dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("The labeled scenario CSV is missing from outputs/.")
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def comparison() -> tuple[pd.DataFrame, dict]:
    # WNTR/EPANET is only required on the evaluation page, so defer its imports
    # to keep ordinary dashboard startup lighter.
    from evaluate_methods import evaluate_methods

    return evaluate_methods(dataset(), load_model(DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def all_schedule_fault_cv() -> dict:
    if not ALL_SCHEDULE_CV_PATH.exists():
        raise FileNotFoundError("The all-schedule grouped evaluation metrics are missing from outputs/.")
    return json.loads(ALL_SCHEDULE_CV_PATH.read_text(encoding="utf-8"))


def card(children, class_name="card", **kwargs):
    return html.Div(children, className=class_name, **kwargs)


def section_heading(kicker: str, title: str, subtitle: str):
    return html.Div([
        html.Div(kicker.upper(), className="eyebrow"),
        html.H1(title),
        html.P(subtitle, className="page-subtitle"),
    ], className="page-heading")


def dropdown(label: str, element_id: str, options, value=None, clearable=False, disabled=False):
    return html.Div([
        html.Label(label, htmlFor=element_id, className="field-label"),
        dcc.Dropdown(id=element_id, options=[{"label": x, "value": x} for x in options],
                     value=value if value is not None else (options[0] if options else None),
                     clearable=clearable, disabled=disabled, className="dash-dropdown"),
    ], className="field")


def metric_card(label, value, detail, tone="blue"):
    return card([
        html.Div(className=f"metric-mark {tone}"),
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
        html.Div(detail, className="metric-detail"),
    ], "metric-card")


def data_table(frame: pd.DataFrame, table_id: str, height="460px"):
    if frame.empty:
        return html.Div("No rows match these filters.", className="empty-state")
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c.replace("_", " "), "id": c} for c in frame.columns],
        data=frame.to_dict("records"), page_size=12, sort_action="native", filter_action="native",
        fixed_rows={"headers": True}, style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
        style_header={"backgroundColor": "var(--card-soft)", "color": "var(--muted)", "fontWeight": 700,
                      "border": "1px solid var(--border)", "padding": "12px", "textAlign": "left"},
        style_cell={"backgroundColor": "var(--card)", "color": "var(--text)", "border": "1px solid var(--border)",
                    "padding": "11px 12px", "fontFamily": "inherit", "fontSize": "13px", "textAlign": "left",
                    "minWidth": "100px", "maxWidth": "300px", "whiteSpace": "normal"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "var(--stripe)"}],
    )


def map_component(row: pd.Series, highlighted_zone: str | None = None):
    """Display geographic context with an illustrative SVG network overlay."""
    if not MAP_PATH.exists():
        return html.Div("Map context image is not available", className="empty-state")
    positions = {"Source": (310, 382), "Manifold": (430, 382), "Zone A": (550, 228),
                 "Zone B": (640, 382), "Zone C": (550, 523)}
    routes = []
    for start, end in [("Source", "Manifold"), ("Manifold", "Zone A"), ("Manifold", "Zone B"), ("Manifold", "Zone C")]:
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        routes.append(f'<path d="M{x1},{y1} L{x2},{y2}"/>')
    nodes = []
    for name, (x, y) in positions.items():
        fill = COLORS["blue"] if name == "Source" else (COLORS["blue2"] if name == "Manifold" else (COLORS["rose"] if name == highlighted_zone else COLORS["cyan"]))
        radius = 13 if name in ("Source", "Manifold") else 15
        nodes.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="white" stroke-width="3"><title>{name} · simulated point</title></circle>')
    labels = [
        f'<g transform="translate(218 424)"><rect width="178" height="31" rx="7" fill="#123a40" stroke="{COLORS["white"]}"/><text fill="{COLORS["white"]}" x="89" y="20" text-anchor="middle">SIMULATED SOURCE</text></g>',
        f'<g transform="translate(383 427)"><rect width="208" height="31" rx="7" fill="#123a40" stroke="{COLORS["white"]}"/><text fill="{COLORS["white"]}" x="104" y="20" text-anchor="middle">DISTRIBUTION MANIFOLD</text></g>',
    ]
    for zone in ZONES:
        key = ZONE_KEYS[zone]
        x, y = positions[zone]
        labels.append(f'<g transform="translate({x-42} {y-58})"><rect width="84" height="29" rx="7" fill="#123a40" stroke="{COLORS["white"]}"/><text fill="{COLORS["white"]}" x="42" y="19" text-anchor="middle">{zone}</text></g>')
        labels.append(f'<g transform="translate({x+25} {y-12})"><rect width="135" height="48" rx="7" fill="#123a40" stroke="{COLORS["white"]}"/><text fill="{COLORS["white"]}" x="10" y="19">{row[f"Flow_{key}_L_min"]:.1f} L/min</text><text fill="{COLORS["white"]}" x="10" y="37">{row[f"Pressure_{key}_kPa"]:.1f} kPa</text></g>')
    overlay = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 670" preserveAspectRatio="none">'
        '<g fill="none" stroke="#173b40" stroke-width="13" stroke-linecap="round" stroke-linejoin="round">' + "".join(routes) + '</g>'
        f'<g fill="none" stroke="{COLORS["cyan"]}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round">' + "".join(routes) + '</g>'
        + "".join(nodes) + '<g font-family="Figtree, Segoe UI, Arial, sans-serif" font-size="13" font-weight="700">'
        + "".join(labels) + '</g></svg>'
    )
    svg_uri = "data:image/svg+xml;base64," + base64.b64encode(overlay.encode("utf-8")).decode("ascii")
    return html.Div([
        html.Img(src="/assets/uow_dubai_context.png", className="map-image", alt="Dubai study area geographic context"),
        html.Img(src=svg_uri, className="map-overlay", alt="Illustrative simulated pipe network, zones and readings"),
        html.Div("SIMULATED NETWORK · NOT REAL CAMPUS PIPES", className="map-disclaimer"),
    ], className="map-stage")


def study_area_component():
    """Geographic context and a schematic boundary, without the simulated pipe layer."""
    return html.Div([
        html.Div([
            html.Img(src="/assets/uow_dubai_context.png", className="study-context-image",
                     alt="Geographic context around UOW Dubai; no simulated pipe layer"),
            html.Div("GEOGRAPHIC CONTEXT · NO SIMULATED PIPELINES", className="map-disclaimer"),
        ], className="study-context-map"),
        html.Div([
            html.Span("STUDY BOUNDARY", className="finding-kicker"),
            html.Div([
                html.Div("UOW Dubai", className="boundary-center-label"),
                html.Div("3 km", className="boundary-radius-label"),
            ], className="radius-diagram"),
            html.H3("Illustrative 3 km radius"),
            html.P("The circle represents the study boundary around the UOW Dubai reference point. It is schematic and not to scale."),
            html.P("Simulated pipes, zones and readings are shown separately in Pipeline view."),
        ], className="boundary-panel"),
    ], className="study-area-layout")


def bar_figure(frame, x, y, color, barmode="group", horizontal=False):
    if horizontal:
        fig = px.bar(frame, x=x, y=y, color=color, barmode=barmode, orientation="h",
                     color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], COLORS["blue2"], COLORS["muted"]], text_auto=".0%")
    else:
        fig = px.bar(frame, x=x, y=y, color=color, barmode=barmode,
                     color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], COLORS["blue2"], COLORS["muted"]], text_auto=".0%")
    fig.update_layout(height=330, margin={"l": 15, "r": 15, "t": 18, "b": 15,
                                      "pad": 4}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Figtree, Segoe UI, sans-serif", "color": "var(--plot-text)"},
                      legend={"orientation": "h", "y": -0.22}, hovermode="x unified")
    if horizontal:
        fig.update_xaxes(tickformat=".0%", range=[0, 1.08], gridcolor="var(--grid)", zeroline=False)
        fig.update_yaxes(gridcolor="var(--grid)", zeroline=False)
    else:
        fig.update_yaxes(tickformat=".0%", range=[0, 1.08], gridcolor="var(--grid)", zeroline=False)
        fig.update_xaxes(gridcolor="var(--grid)", zeroline=False)
    return fig


def confusion_figure(frame, prediction_column, title):
    labels = ["normal", "leak", "clog"]
    predicted = labels + (["unknown_fault"] if (frame[prediction_column] == "unknown_fault").any() else [])
    matrix = pd.crosstab(frame["Condition"], frame[prediction_column]).reindex(index=labels, columns=predicted, fill_value=0)
    fig = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=[p.title() for p in predicted], y=[x.title() for x in labels],
                               colorscale=[[0, COLORS["white"]], [1, COLORS["blue"]]], showscale=False,
                               text=matrix.to_numpy(), texttemplate="%{text}", hovertemplate="Actual %{y}<br>Predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=title, height=315, margin={"l": 12, "r": 12, "t": 40, "b": 20},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Figtree, Segoe UI, sans-serif", "color": "var(--plot-text)"})
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return fig


def render_page(page: str):
    data = dataset()
    if page == "Study area":
        return html.Div([
            section_heading("Urban water · geographic context", "Study area", "Illustrative 3 km study boundary around the UOW Dubai reference point"),
            card([html.Div([html.Span("i", className="notice-icon"), html.Div([
                html.Strong("Geographic context only"),
                html.P("This view shows the study area and its schematic boundary. It does not display pipe routes, irrigation zones or sensor readings."),
            ])], className="notice-content")], "notice-card"),
            card(study_area_component(), "card study-area-card"),
            html.P("The 3 km boundary is a study assumption for context; it does not describe real irrigation infrastructure.", className="footnote"),
        ], className="page-view")
    if page == "Pipeline view":
        schedules = sorted(data.Schedule.unique())
        default_schedule = schedules[0]
        default_row = data.loc[
            (data.Schedule == default_schedule)
            & (data.Condition == "normal")
            & (data.Hour == 0)
        ].iloc[0]
        return html.Div([
            section_heading("Network model", "Pipeline overview", "Explore simulated pressure and flow under schedules and fault scenarios"),
            card(html.Div([html.Span("i", className="notice-icon"), html.P("Pipe geometry and locations are illustrative, not surveyed infrastructure.")], className="notice-content"), "notice-card"),
            html.Div([
                dropdown("Schedule", "pipe-schedule", schedules, default_schedule),
                dropdown("Condition", "pipe-condition", ["normal", "leak", "clog"], "normal"),
                dropdown("Severity", "pipe-severity", ["none", "mild", "moderate", "severe"], "none"),
                dropdown("Fault zone", "pipe-zone", ["none", *ZONES], "none"),
                dropdown("Hour", "pipe-hour", list(range(6)), 0),
            ], className="control-row pipeline-controls"),
            html.Div([
                html.Span("Choose the scenario settings, then run to refresh the map and readings.", className="pipeline-run-hint"),
                html.Button(["Run scenario", html.Span("→", className="button-arrow")], id="run-pipeline", n_clicks=0, className="button primary"),
            ], className="pipeline-run-row"),
            html.Div(id="pipeline-summary", className="scenario-summary"),
            card(html.Div(id="pipeline-map", children=map_component(default_row)), "map-card"),
        ], className="page-view")
    if page == "Watering data":
        schedules = ["All", *sorted(data.Schedule.unique())]
        conditions = ["All", *sorted(data.Condition.unique())]
        locations = ["All", *sorted(data.Fault_Location.unique())]
        return html.Div([
            section_heading("Synthetic dataset", "Watering data", "Filter and inspect the labeled WNTR scenario-hour records"),
            html.Div(id="dataset-summary", className="metric-grid three"),
            html.Div([dropdown("Schedule", "data-schedule", schedules), dropdown("Condition", "data-condition", conditions), dropdown("Fault location", "data-location", locations)], className="control-row"),
            card([html.Div([html.H3("Scenario records"), html.Button("Download CSV", id="download-csv", className="button secondary")], className="card-heading-row"),
                  html.Div(id="dataset-table")], "card table-card"),
            dcc.Download(id="download-data"),
            html.P("Synthetic data source: WNTR/EPANET model. See DATA_DICTIONARY.md for field meanings and assumptions.", className="footnote"),
        ], className="page-view")
    if page == "Method comparison":
        predictions, metrics = comparison()
        cv_metrics = all_schedule_fault_cv()
        rf, baseline = metrics["random_forest"], metrics["schedule_aware_threshold"]
        cv_rf, cv_threshold = cv_metrics["random_forest"], cv_metrics["schedule_aware_threshold"]
        fault_record_count = int((predictions["Condition"] != "normal").sum())
        random_forest_alerted_faults = fault_record_count - round(rf["missed_fault_rate"] * fault_record_count)
        normal_record_count = int((predictions["Condition"] == "normal").sum())
        rf_false_warning_mask = (predictions["Condition"] == "normal") & (predictions["Predicted_Condition"] != "normal")
        rf_false_warning_count = int(rf_false_warning_mask.sum())
        rf_false_warning_hours = ", ".join(
            str(int(hour)) for hour in sorted(predictions.loc[rf_false_warning_mask, "Hour"].unique())
        )
        night_missed_rf = round(rf["missed_fault_rate"] * fault_record_count)
        night_missed_threshold = round(baseline["missed_fault_rate"] * fault_record_count)
        night_false_rf = round(rf["false_alert_rate"] * normal_record_count)
        night_false_threshold = round(baseline["false_alert_rate"] * normal_record_count)
        summary = [
            metric_card(
                "Macro F1 · night / all-schedule faults",
                f"{rf['macro_f1']:.1%} / {cv_rf['macro_f1_leak_clog']:.1%}",
                f"Random Forest · Threshold {baseline['macro_f1']:.1%} / {cv_threshold['macro_f1_leak_clog']:.1%}",
                "blue",
            ),
            metric_card(
                "Fault location · night / all-schedule faults",
                f"{rf['fault_location_accuracy']:.1%} / {cv_metrics['fault_location_accuracy']['random_forest']:.1%}",
                f"Random Forest · Threshold {baseline['fault_location_accuracy']:.1%} / {cv_metrics['fault_location_accuracy']['schedule_aware_threshold']:.1%}",
                "cyan",
            ),
            metric_card(
                "Missed faults · night / all-schedule",
                f"{rf['missed_fault_rate']:.1%} / {cv_rf['missed_fault_rate']:.1%}",
                f"RF {night_missed_rf}/{fault_record_count} · {cv_rf['missed_fault_count']}/{cv_metrics['record_count']} | Threshold {night_missed_threshold}/{fault_record_count} · {cv_threshold['missed_fault_count']}/{cv_metrics['record_count']}",
                "rose",
            ),
            metric_card(
                "False warnings · night normals only",
                f"{rf['false_alert_rate']:.1%}",
                f"RF {night_false_rf}/{normal_record_count} · Threshold {night_false_threshold}/{normal_record_count}*; all-schedule CV has no normal test rows",
                "gold",
            ),
        ]
        performance = pd.DataFrame([
            {"Measure": label, "Method": method, "Score": scores[key]}
            for label, key in [("Condition accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("Fault location", "fault_location_accuracy"), ("Missed faults", "missed_fault_rate"), ("False warnings*", "false_alert_rate")]
            for method, scores in [("Random Forest", rf), ("Schedule-aware threshold", baseline)]
        ])
        classscores = pd.DataFrame([
            {"Condition": name.title(), "Score": scores[name][metric], "Series": f"{method} {metric.title()}"}
            for method, result in [("Random Forest", rf), ("Threshold", baseline)]
            for scores in [result["per_class"]]
            for name in ["normal", "leak", "clog"] for metric in ["precision", "recall"]
        ])
        return html.Div([
            section_heading("Model evaluation", "Method comparison", "Random Forest is the primary method; the schedule-aware threshold is the benchmark"),
            card(html.Div([
                html.Span("Overall result", className="finding-kicker"),
                html.H3("Random Forest led on overall fault classification"),
                html.P(f"On this held-out test, Random Forest scored {rf['accuracy']:.1%} accuracy and {rf['macro_f1']:.1%} macro F1, compared with {baseline['accuracy']:.1%} and {baseline['macro_f1']:.1%} for the schedule-aware threshold. It flagged {random_forest_alerted_faults}/{fault_record_count} simulated fault records."),
                html.P(f"False-warning detail: Random Forest flagged {rf_false_warning_count} of {normal_record_count} normal night records (Hours {rf_false_warning_hours}). The full night schedule was held out from training, so these cases test the model on an unseen schedule; the six normal records make the 50% rate sensitive to individual cases. This is a small synthetic test result, not an estimate for real irrigation operations."),
                html.P(f"Trade-off: the threshold baseline had higher fault-location accuracy ({baseline['fault_location_accuracy']:.1%} vs {rf['fault_location_accuracy']:.1%}). Its 0/6 false warnings are not an independent estimate because those normal readings also contribute to the schedule/hour reference."),
                html.A("Read testing, validation and iteration report ↗", href="/testing-report", target="_blank", className="report-link"),
            ], className="finding-copy"), "finding-card"),
            html.H3("Night / all-schedule results"),
            html.Div(summary, className="metric-grid four"),
            html.Div([card([html.H3("Diagnosis quality and alert risk · night holdout"), dcc.Graph(figure=bar_figure(performance, "Score", "Measure", "Method", horizontal=True), config={"displayModeBar": False})]),
                      card([html.H3("Precision and recall by condition · night holdout"), dcc.Graph(figure=bar_figure(classscores, "Score", "Condition", "Series"), config={"displayModeBar": False})])], className="chart-grid two"),
            html.Div([card([html.H3("Classification outcomes · night holdout"), html.Div([dcc.Graph(figure=confusion_figure(predictions, "Predicted_Condition", "Random Forest"), config={"displayModeBar": False}),
                      dcc.Graph(figure=confusion_figure(predictions, "Threshold_Condition", "Schedule-aware threshold"), config={"displayModeBar": False})], className="confusion-pair")])], className="chart-grid"),
            html.P("*Night-only false-warning figures are based on six normal records. The threshold baseline uses these same normal data to establish expected schedule/hour values, so its 0/6 is not an independent false-warning estimate. The all-schedule fault evaluation contains no normal test records.", className="footnote"),
            html.P(metrics["response_time_note"], className="footnote"),
            card([html.Div([html.H3("Held-out predictions"), html.Div([dropdown("Condition", "pred-condition", ["All", *sorted(predictions.Condition.unique())]), dropdown("Severity", "pred-severity", ["All", *sorted(predictions.Severity.unique())]), dropdown("Location", "pred-location", ["All", *sorted(predictions.Fault_Location.unique())])], className="control-row compact")], className="predictions-heading"), html.Div(id="prediction-table")], "card table-card"),
        ], className="page-view")
    if page == "Live diagnosis":
        scenarios = sorted(data.Scenario_ID.unique())
        default_scenario = "NORMAL_NIGHT" if "NORMAL_NIGHT" in scenarios else scenarios[0]
        return html.Div([
            section_heading("Scenario analysis", "Live diagnosis", "Run the saved Random Forest and threshold baseline on a selected simulated case"),
            html.Div([dropdown("Scenario", "diagnosis-scenario", scenarios, default_scenario), dropdown("Simulation hour", "diagnosis-hour", list(range(6)), 0), html.Button(["Run diagnosis", html.Span("→", className="button-arrow")], id="run-diagnosis", className="button primary")], className="control-row diagnosis-controls"),
            html.Div(id="diagnosis-alert", className="alert-host"),
            html.Div(id="diagnosis-readings", className="metric-grid three"),
            html.Div(id="diagnosis-results", className="diagnosis-results"),
        ], className="page-view")
    return html.Div("Page not found", className="empty-state")


def nav_item(page: str):
    return html.Button([html.Span(PAGE_ICONS[page], className="nav-icon"), html.Span(page)],
                       id={"type": "nav", "page": page}, className="nav-item")


app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="active-page", data="Study area"),
    dcc.Store(id="theme-mode", data="light", storage_type="local"),
    dcc.Store(id="notification-state", data={"open": False, "message": "No new diagnosis alerts."}),
    dcc.Interval(id="clock-tick", interval=30_000, n_intervals=0),
    html.Aside([
        html.Div([html.Div("IR", className="brand-mark"), html.Div([html.Span("IRRIGATION TWIN", className="brand-title"), html.Small("FAULT DIAGNOSIS WORKSPACE")])], className="brand-row"),
        html.Div([html.Div("WORKSPACE", className="nav-group-title"), *[nav_item(p) for p in PAGES[:2]],
                  html.Div("DATA & INSIGHTS", className="nav-group-title second"), *[nav_item(p) for p in PAGES[2:]]], className="sidebar-nav"),
        html.Div([html.Span("●", className="connection-dot"), html.Span("SIMULATION DATA"), html.Small("No live sensors connected")], className="sidebar-status"),
        html.P("Network paths and readings are generated from documented simulation assumptions.", className="sidebar-footnote"),
    ], className="sidebar"),
    html.Div([
        html.Header([
            html.Div([html.Span("Workspace", className="crumb-muted"), html.Span("/", className="crumb-slash"), html.Strong(id="breadcrumb", children="Study area")], className="breadcrumbs"),
            html.Div(id="dubai-clock", className="clock-text"),
            html.Button([html.Span("☾", id="theme-icon"), html.Span("Dark mode", id="theme-label")], id="theme-toggle", className="theme-toggle"),
            html.Div([html.Button("♧", id="notifications-toggle", className="icon-button"), html.Div(id="notification-panel", className="notification-panel hidden")], className="notification-wrap"),
        ], className="topbar"),
        html.Main(id="page-content", className="main-content"),
    ], className="workspace"),
], id="app-shell", className="app-shell light-theme")


@callback(Output("active-page", "data"),
          Input({"type": "nav", "page": "Study area"}, "n_clicks"),
          Input({"type": "nav", "page": "Pipeline view"}, "n_clicks"),
          Input({"type": "nav", "page": "Watering data"}, "n_clicks"),
          Input({"type": "nav", "page": "Method comparison"}, "n_clicks"),
          Input({"type": "nav", "page": "Live diagnosis"}, "n_clicks"),
          prevent_initial_call=True)
def navigate(*clicks):
    from dash import ctx
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        return no_update
    return ctx.triggered_id["page"]


@callback(
    Output({"type": "nav", "page": "Study area"}, "className"),
    Output({"type": "nav", "page": "Pipeline view"}, "className"),
    Output({"type": "nav", "page": "Watering data"}, "className"),
    Output({"type": "nav", "page": "Method comparison"}, "className"),
    Output({"type": "nav", "page": "Live diagnosis"}, "className"),
    Input("active-page", "data"),
)
def mark_active_navigation(page):
    return tuple("nav-item active" if name == page else "nav-item" for name in PAGES)


@callback(Output("page-content", "children"), Output("breadcrumb", "children"), Input("active-page", "data"))
def change_page(page):
    try:
        return render_page(page), page
    except Exception as exc:
        return html.Div([section_heading("Workspace", page, "The page could not be prepared"), card(str(exc), "notice-card error-card")]), page


@callback(Output("app-shell", "className"), Output("theme-icon", "children"), Output("theme-label", "children"),
          Input("theme-mode", "data"))
def set_theme(mode):
    dark = mode == "dark"
    return ("app-shell dark-theme" if dark else "app-shell light-theme", "☼" if dark else "☾", "Light mode" if dark else "Dark mode")


@callback(Output("theme-mode", "data"), Input("theme-toggle", "n_clicks"), State("theme-mode", "data"), prevent_initial_call=True)
def toggle_theme(n, mode):
    return "dark" if mode != "dark" else "light"


@callback(Output("dubai-clock", "children"), Input("clock-tick", "n_intervals"))
def update_clock(_):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=4)))
    return now.strftime("%A %d %b · %H:%M · Dubai")


@callback(Output("notification-state", "data"), Input("notifications-toggle", "n_clicks"), State("notification-state", "data"), prevent_initial_call=True)
def toggle_notifications(_, state):
    state = state or {"open": False, "message": "No new diagnosis alerts."}
    return {**state, "open": not state.get("open", False)}


@callback(Output("notification-panel", "children"), Output("notification-panel", "className"), Output("notification-panel", "style"), Input("notification-state", "data"))
def draw_notification(state):
    state = state or {"open": False, "message": "No new diagnosis alerts."}
    content = [html.Strong("Notifications"), html.P(state.get("message", "No new diagnosis alerts."))]
    return content, "notification-panel", {"display": "block" if state.get("open") else "none"}


@callback(Output("pipeline-hour", "options"), Output("pipeline-hour", "value"),
          Output("pipeline-map", "children"), Output("pipeline-summary", "children"),
          Input("run-pipeline", "n_clicks"), State("pipe-schedule", "value"), State("pipe-condition", "value"),
          State("pipe-severity", "value"), State("pipe-zone", "value"), State("pipe-hour", "value"),
          prevent_initial_call=True)
def update_pipeline(_n_clicks, schedule, condition, severity, zone, hour):
    data = dataset()
    frame = data.loc[(data.Schedule == schedule) & (data.Condition == condition)]
    if condition == "normal":
        frame = frame.loc[frame.Severity == "none"]
    else:
        frame = frame.loc[(frame.Severity == severity) & (frame.Fault_Location == zone)]
    hours = sorted(frame.Hour.unique())
    if not hours:
        return [], None, html.Div("No scenario matches the selected filters.", className="empty-state"), html.Div()
    selected_hour = hour if hour in hours else hours[0]
    row = frame.loc[frame.Hour == selected_hour].iloc[0]
    label = f"{schedule.title()} schedule · {condition.title()} · {severity.title()} · {zone if zone != 'none' else 'No fault'} · Hour {selected_hour}"
    return ([{"label": f"Hour {int(h)}", "value": int(h)} for h in hours], selected_hour,
            map_component(row, zone if condition != "normal" else None), html.Span(label, className="scenario-chip"))


@callback(Output("dataset-summary", "children"), Output("dataset-table", "children"),
          Input("data-schedule", "value"), Input("data-condition", "value"), Input("data-location", "value"))
def update_dataset(schedule, condition, location):
    frame = dataset().copy()
    if schedule != "All": frame = frame.loc[frame.Schedule == schedule]
    if condition != "All": frame = frame.loc[frame.Condition == condition]
    if location != "All": frame = frame.loc[frame.Fault_Location == location]
    summary = [metric_card("Scenario-hour records", str(len(frame)), "Filtered rows", "blue"),
               metric_card("Scenario configurations", str(frame.Scenario_ID.nunique()), "Unique scenario IDs", "cyan"),
               metric_card("Schedules represented", str(frame.Schedule.nunique()), "Watering profiles", "gold")]
    columns = ["Scenario_ID", "Schedule", "Hour", "Condition", "Severity", "Fault_Location",
               "Pressure_ZoneA_kPa", "Pressure_ZoneB_kPa", "Pressure_ZoneC_kPa",
               "Flow_ZoneA_L_min", "Flow_ZoneB_L_min", "Flow_ZoneC_L_min"]
    return summary, data_table(frame[columns], "dataset-records")


@callback(Output("download-data", "data"), Input("download-csv", "n_clicks"),
          State("data-schedule", "value"), State("data-condition", "value"), State("data-location", "value"), prevent_initial_call=True)
def download_filtered(_, schedule, condition, location):
    frame = dataset().copy()
    if schedule != "All": frame = frame.loc[frame.Schedule == schedule]
    if condition != "All": frame = frame.loc[frame.Condition == condition]
    if location != "All": frame = frame.loc[frame.Fault_Location == location]
    return dcc.send_data_frame(frame.to_csv, "labeled_fault_scenarios_filtered.csv", index=False)


@callback(Output("prediction-table", "children"), Input("pred-condition", "value"), Input("pred-severity", "value"), Input("pred-location", "value"))
def update_predictions(condition, severity, location):
    predictions, _ = comparison()
    frame = predictions.copy()
    if condition != "All": frame = frame.loc[frame.Condition == condition]
    if severity != "All": frame = frame.loc[frame.Severity == severity]
    if location != "All": frame = frame.loc[frame.Fault_Location == location]
    columns = ["Scenario_ID", "Hour", "Condition", "Severity", "Fault_Location", "Predicted_Condition",
               "Predicted_Fault_Location", "Predicted_Severity", "Threshold_Alert", "Threshold_Condition", "Threshold_Location"]
    return data_table(frame[columns], "heldout-predictions", "520px")


@callback(Output("diagnosis-hour", "options"), Output("diagnosis-hour", "value"), Input("diagnosis-scenario", "value"), State("diagnosis-hour", "value"))
def update_diagnosis_hours(scenario, current):
    hours = sorted(dataset().loc[dataset().Scenario_ID == scenario, "Hour"].unique())
    value = current if current in hours else hours[0]
    return [{"label": f"Hour {int(h)}", "value": int(h)} for h in hours], value


@callback(Output("diagnosis-readings", "children"), Input("diagnosis-scenario", "value"), Input("diagnosis-hour", "value"))
def show_sensor_readings(scenario, hour):
    rows = dataset().loc[(dataset().Scenario_ID == scenario) & (dataset().Hour == hour)]
    if rows.empty: return []
    row = rows.iloc[0]
    return [metric_card(f"{zone} pressure", f"{row[f'Pressure_{key}_kPa']:.1f} kPa", f"{row[f'Flow_{key}_L_min']:.1f} L/min flow", "blue" if i == 0 else "cyan")
            for i, (zone, key) in enumerate(ZONE_KEYS.items())]


@callback(Output("diagnosis-results", "children"), Output("diagnosis-alert", "children"),
          Output("notification-state", "data", allow_duplicate=True), Input("run-diagnosis", "n_clicks"),
          State("diagnosis-scenario", "value"), State("diagnosis-hour", "value"), State("notification-state", "data"),
          prevent_initial_call=True)
def run_diagnosis(_, scenario, hour, prior_notification):
    data = dataset()
    selected = data.loc[(data.Scenario_ID == scenario) & (data.Hour == hour)].copy()
    if selected.empty: return no_update, no_update, no_update
    rf = predict_records(selected, load_model())
    threshold = predict_thresholds(selected, baseline_data=data)
    row = selected.iloc[0]
    rf_condition = str(rf.Predicted_Condition.iloc[0])
    rf_location = str(rf.Predicted_Fault_Location.iloc[0])
    rf_severity = str(rf.Predicted_Severity.iloc[0])
    threshold_condition = str(threshold.Threshold_Condition.iloc[0])
    threshold_location = str(threshold.Threshold_Location.iloc[0])
    actual_condition = str(row.Condition)
    actual_severity = str(row.Severity)
    actual_location = str(row.Fault_Location)
    condition_agrees = rf_condition == threshold_condition
    location_agrees = rf_location == threshold_location
    methods_agree = condition_agrees and location_agrees
    differences = []
    if not condition_agrees:
        differences.append("condition")
    if not location_agrees:
        differences.append("suspected zone")
    message = f"{scenario} · hour {hour}. Random Forest: {rf_condition} at {rf_location}; threshold: {threshold_condition} at {threshold_location}."
    rf_summary = f"{rf_condition.replace('_', ' ')} at {rf_location if rf_location != 'none' else 'no fault zone'}"
    threshold_summary = f"{threshold_condition.replace('_', ' ')} at {threshold_location if threshold_location != 'none' else 'no fault zone'}"
    banner_title = "Methods agree" if methods_agree else "Methods disagree"
    if methods_agree:
        banner_message = f"Both methods predicted {rf_summary}. Compare their shared result with the known simulation label below."
    else:
        banner_message = f"The methods differ on {' and '.join(differences)}. Random Forest: {rf_summary}; threshold: {threshold_summary}."
    banner = html.Div([html.Span("●", className="alert-dot"), html.Div([html.Strong(banner_title), html.P(banner_message)])], className="result-banner agree" if methods_agree else "result-banner disagreement")
    flow_delta, pressure_delta = [], []
    baseline = data.loc[(data.Condition == "normal") & (data.Schedule == row.Schedule) & (data.Hour == hour)]
    if not baseline.empty:
        base = baseline.iloc[0]
        for zone, key in ZONE_KEYS.items():
            flow_delta.append({"Zone": zone, "Observed − expected": row[f"Flow_{key}_L_min"] - base[f"Flow_{key}_L_min"], "Metric": "Flow (L/min)"})
            pressure_delta.append({"Zone": zone, "Observed − expected": row[f"Pressure_{key}_kPa"] - base[f"Pressure_{key}_kPa"], "Metric": "Pressure (kPa)"})
    deviation = pd.DataFrame(flow_delta + pressure_delta)
    chart = px.bar(deviation, x="Zone", y="Observed − expected", color="Metric", barmode="group",
                   color_discrete_map={"Flow (L/min)": COLORS["blue"], "Pressure (kPa)": COLORS["cyan"]}, text_auto=".2f")
    chart.update_layout(height=340, margin={"l": 15, "r": 15, "t": 15, "b": 15}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Figtree, Segoe UI, sans-serif", "color": "var(--plot-text)"}, legend={"orientation": "h", "y": -0.2})
    chart.update_yaxes(title="Change from matching normal baseline", gridcolor="var(--grid)", zerolinecolor="var(--muted)")
    chart.update_xaxes(gridcolor="var(--grid)")
    def prediction_card(title, method_class, condition, location, severity):
        condition_matches = condition == actual_condition
        location_matches = location == actual_location
        severity_is_applicable = actual_severity in {"mild", "moderate", "severe"}
        severity_matches = severity == actual_severity
        severity_label = severity.replace("_", " ").title()
        if method_class == "rf" and severity == "not_applicable" and severity_is_applicable:
            severity_label = "Not returned (normal predicted)"
        elif method_class != "rf":
            severity_label = "Not estimated"
        checks = [
            html.Span("Condition matches" if condition_matches else "Condition differs", className="prediction-chip match" if condition_matches else "prediction-chip mismatch"),
            html.Span("Zone matches" if location_matches else "Zone differs", className="prediction-chip match" if location_matches else "prediction-chip mismatch"),
        ]
        if method_class == "rf" and severity_is_applicable:
            severity_returned = severity != "not_applicable"
            checks.append(html.Span(
                ("Severity matches" if severity_matches else "Severity differs") if severity_returned else "Severity not returned",
                className="prediction-chip match" if severity_returned and severity_matches else "prediction-chip mismatch",
            ))
        return card([
            html.Span("PRIMARY MODEL" if method_class == "rf" else "RULE-BASED BENCHMARK", className="result-kicker"),
            html.H3(title),
            html.Div(condition.replace("_", " ").title(), className="result-condition"),
            html.Div(checks, className="prediction-checks"),
            html.Div([html.Span("Severity estimate"), html.Strong(severity_label)], className="result-location"),
            html.Div([html.Span("Suspected zone"), html.Strong(location)], className="result-location"),
        ], f"result-method {method_class}")

    results = html.Div([
        html.Div([html.H2("Diagnosis results"), html.P(f"Simulated result · {scenario} · hour {hour}", className="page-subtitle")], className="result-heading"),
        card([
            html.Div("KNOWN SIMULATION LABEL", className="truth-kicker"),
            html.Div([
                html.Div([html.Span("Condition"), html.Strong(actual_condition.title())], className="truth-value"),
                html.Div([html.Span("Severity"), html.Strong(actual_severity.title())], className="truth-value"),
                html.Div([html.Span("Affected zone"), html.Strong(actual_location)], className="truth-value"),
            ], className="truth-fields"),
            html.P("This is the recorded answer for the selected synthetic scenario; the methods do not receive this label as an input.", className="truth-caption"),
        ], "truth-card"),
        html.Div([prediction_card("Random Forest", "rf", rf_condition, rf_location, rf_severity),
                  prediction_card("Schedule-aware threshold", "threshold", threshold_condition, threshold_location, "not_estimated")], className="diagnosis-methods"),
        card([html.H3("Sensor changes from the matching schedule baseline"), html.P("Observed minus expected values for the selected simulation hour.", className="card-subtitle"), dcc.Graph(figure=chart, config={"displayModeBar": False})], "card"),
    ], className="result-stack")
    notification = {"open": True, "message": message}
    return results, banner, notification


if __name__ == "__main__":
    app.run(debug=False, dev_tools_ui=False, dev_tools_props_check=False, dev_tools_hot_reload=False, host="0.0.0.0", port=8050)
