"""
Real-Time Sentiment Dashboard
Streams live social media sentiment with Plotly Dash + WebSocket updates.
"""

import json
import redis
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
from datetime import datetime, timedelta
from collections import deque

app = Dash(__name__, title="Sentiment Dashboard")
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
}

app.layout = html.Div([
    html.Div([
        html.H1("Real-Time Sentiment Monitor", style={"margin": 0}),
        html.P(id="last-updated", style={"color": "#6b7280", "margin": "4px 0 0"}),
    ], style={"padding": "24px 24px 0", "borderBottom": "1px solid #e5e7eb"}),

    html.Div([
        # Metric cards row
        html.Div([
            html.Div([
                html.P("Total Posts", style={"color": "#6b7280", "margin": 0, "fontSize": "13px"}),
                html.H2(id="total-count", children="—", style={"margin": "4px 0 0"}),
            ], className="metric-card"),
            html.Div([
                html.P("Positive", style={"color": "#6b7280", "margin": 0, "fontSize": "13px"}),
                html.H2(id="positive-pct", children="—", style={"margin": "4px 0 0", "color": "#22c55e"}),
            ], className="metric-card"),
            html.Div([
                html.P("Negative", style={"color": "#6b7280", "margin": 0, "fontSize": "13px"}),
                html.H2(id="negative-pct", children="—", style={"margin": "4px 0 0", "color": "#ef4444"}),
            ], className="metric-card"),
            html.Div([
                html.P("Avg Confidence", style={"color": "#6b7280", "margin": 0, "fontSize": "13px"}),
                html.H2(id="avg-confidence", children="—", style={"margin": "4px 0 0"}),
            ], className="metric-card"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px", "padding": "24px"}),

        # Charts row
        html.Div([
            dcc.Graph(id="sentiment-timeseries", style={"flex": 2}),
            dcc.Graph(id="sentiment-donut", style={"flex": 1}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 24px"}),

        # Live feed
        html.Div([
            html.H3("Live Feed", style={"margin": "0 0 12px"}),
            html.Div(id="live-feed"),
        ], style={"padding": "24px"}),
    ]),

    dcc.Interval(id="interval", interval=2000, n_intervals=0),
], style={"fontFamily": "system-ui, sans-serif", "maxWidth": "1400px", "margin": "0 auto"})


def get_recent_data(window_minutes: int = 60) -> pd.DataFrame:
    """Fetch recent sentiment records from Redis."""
    keys = r.keys("sentiment:*")
    records = []
    for key in keys[-500:]:
        raw = r.get(key)
        if raw:
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    return df[df["timestamp"] >= cutoff]


@callback(
    Output("total-count", "children"),
    Output("positive-pct", "children"),
    Output("negative-pct", "children"),
    Output("avg-confidence", "children"),
    Output("sentiment-timeseries", "figure"),
    Output("sentiment-donut", "figure"),
    Output("live-feed", "children"),
    Output("last-updated", "children"),
    Input("interval", "n_intervals"),
)
def update_dashboard(n):
    df = get_recent_data()

    if df.empty:
        empty_fig = go.Figure()
        return "0", "—", "—", "—", empty_fig, empty_fig, [], "No data yet"

    total = len(df)
    dist = df["label"].value_counts(normalize=True) * 100
    pos_pct = f"{dist.get('positive', 0):.1f}%"
    neg_pct = f"{dist.get('negative', 0):.1f}%"
    avg_conf = f"{df['score'].mean():.2%}"

    # Time series
    df_ts = df.set_index("timestamp").resample("5min")["label"].value_counts().unstack(fill_value=0)
    ts_fig = go.Figure()
    for label, color in SENTIMENT_COLORS.items():
        if label in df_ts.columns:
            ts_fig.add_trace(go.Scatter(
                x=df_ts.index, y=df_ts[label], name=label.capitalize(),
                line={"color": color, "width": 2}, fill="tonexty" if label != "positive" else None
            ))
    ts_fig.update_layout(title="Sentiment Over Time", xaxis_title="Time",
                          yaxis_title="Count", plot_bgcolor="white", height=300)

    # Donut
    counts = df["label"].value_counts()
    donut_fig = go.Figure(go.Pie(
        labels=counts.index.str.capitalize().tolist(),
        values=counts.values.tolist(),
        hole=0.6,
        marker_colors=[SENTIMENT_COLORS.get(l, "#ccc") for l in counts.index],
    ))
    donut_fig.update_layout(title="Distribution", height=300)

    # Live feed (last 10 posts)
    feed_items = []
    for _, row in df.sort_values("timestamp", ascending=False).head(10).iterrows():
        color = SENTIMENT_COLORS.get(row.get("label", "neutral"), "#94a3b8")
        feed_items.append(html.Div([
            html.Span(row.get("label", "").upper(),
                      style={"background": color, "color": "white", "padding": "2px 8px",
                             "borderRadius": "4px", "fontSize": "11px", "fontWeight": "600",
                             "marginRight": "8px"}),
            html.Span(row.get("text", ""), style={"color": "#374151", "fontSize": "14px"}),
            html.Span(f" {row['score']:.0%}", style={"color": "#9ca3af", "fontSize": "12px"}),
        ], style={"padding": "8px 0", "borderBottom": "1px solid #f3f4f6"}))

    updated = f"Last updated: {datetime.utcnow().strftime('%H:%M:%S UTC')}"
    return total, pos_pct, neg_pct, avg_conf, ts_fig, donut_fig, feed_items, updated


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
