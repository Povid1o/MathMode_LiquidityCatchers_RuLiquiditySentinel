"""Reusable Plotly chart builders for the dashboard."""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dashboard.config import COLORS, PLOTLY_TEMPLATE, MAD_STRESS_THRESHOLD


def _base_layout(**kwargs) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=20, t=88, b=40),
        hovermode="x unified",
        title_y=0.98,
        title_yanchor="top",
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0),
        **kwargs,
    )


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: list[str],
    labels: dict | None = None,
    title: str = "",
    yaxis_title: str = "",
    height: int = 380,
) -> go.Figure:
    """Generic multi-line time-series chart."""
    labels = labels or {}
    fig = go.Figure()
    for col in y:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[col],
            name=labels.get(col, col),
            mode="lines",
            connectgaps=False,
        ))
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        height=height,
        **_base_layout(),
    )
    return fig


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None = None,
    color_scale: list | None = None,
    title: str = "",
    yaxis_title: str = "",
    height: int = 320,
) -> go.Figure:
    """Bar chart, optionally colored by a numeric column."""
    if color_col and color_col in df.columns:
        fig = px.bar(
            df, x=x, y=y,
            color=color_col,
            color_continuous_scale=color_scale or ["#2ca02c", "#bcbd22", "#d62728"],
            title=title,
            height=height,
            template=PLOTLY_TEMPLATE,
        )
    else:
        fig = px.bar(df, x=x, y=y, title=title, height=height, template=PLOTLY_TEMPLATE)
    fig.update_layout(yaxis_title=yaxis_title, **_base_layout())
    return fig


def mad_score_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    height: int = 300,
) -> go.Figure:
    """Bar chart for MAD-score with color by absolute value."""
    vals = df[y].fillna(0)
    colors = [
        COLORS["danger"] if abs(v) >= MAD_STRESS_THRESHOLD else
        COLORS["warn"] if abs(v) >= 1.5 else
        COLORS["success"]
        for v in vals
    ]
    fig = go.Figure(go.Bar(
        x=df[x], y=vals,
        marker_color=colors,
        name=y,
    ))
    fig.add_hline(y=MAD_STRESS_THRESHOLD, line_dash="dot", line_color=COLORS["danger"], opacity=0.6)
    fig.add_hline(y=-MAD_STRESS_THRESHOLD, line_dash="dot", line_color=COLORS["danger"], opacity=0.6)
    fig.update_layout(
        title=title, height=height, yaxis_title="MAD score",
        **_base_layout(),
    )
    return fig


def signal_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    height: int = 320,
    threshold: float = MAD_STRESS_THRESHOLD,
) -> go.Figure:
    """Line chart with stress threshold bands."""
    fig = go.Figure()
    fig.add_hrect(y0=threshold, y1=df[y].max() * 1.1 if df[y].max() > threshold else threshold + 1,
                  fillcolor=COLORS["stress_high"], line_width=0)
    fig.add_hrect(y0=df[y].min() * 1.1 if df[y].min() < -threshold else -threshold - 1,
                  y1=-threshold,
                  fillcolor=COLORS["stress_high"], line_width=0)
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        mode="lines",
        name=y,
        line=dict(color=COLORS["primary"]),
        connectgaps=False,
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["danger"], opacity=0.7)
    fig.add_hline(y=-threshold, line_dash="dash", line_color=COLORS["danger"], opacity=0.7)
    fig.update_layout(title=title, height=height, yaxis_title="Сигнал", **_base_layout())
    return fig


def event_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    flag_col: str | None = None,
    flag_label: str = "Flag",
    title: str = "",
    yaxis_title: str = "",
    height: int = 340,
) -> go.Figure:
    """Scatter for sparse event data with optional flag highlighting."""
    fig = go.Figure()
    if flag_col and flag_col in df.columns:
        normal = df[df[flag_col] != 1]
        flagged = df[df[flag_col] == 1]
        fig.add_trace(go.Scatter(
            x=normal[x], y=normal[y],
            mode="markers",
            marker=dict(color=COLORS["primary"], size=7),
            name="Норма",
        ))
        fig.add_trace(go.Scatter(
            x=flagged[x], y=flagged[y],
            mode="markers",
            marker=dict(color=COLORS["danger"], size=10, symbol="x"),
            name=flag_label,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="markers",
            marker=dict(color=COLORS["primary"], size=7),
            name=y,
        ))
    fig.update_layout(
        title=title, height=height,
        yaxis_title=yaxis_title,
        **_base_layout(),
    )
    return fig


def dual_axis_chart(
    df: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    y1_label: str = "",
    y2_label: str = "",
    title: str = "",
    height: int = 380,
) -> go.Figure:
    """Two-line chart with secondary Y axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y1],
        name=y1_label or y1,
        line=dict(color=COLORS["primary"]),
        connectgaps=False,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y2],
        name=y2_label or y2,
        line=dict(color=COLORS["secondary"]),
        connectgaps=False,
    ), secondary_y=True)
    fig.update_layout(
        title=title, height=height,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, t=88, b=40),
        hovermode="x unified",
        title_y=0.98,
        title_yanchor="top",
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0),
    )
    return fig


def flag_timeline(
    df: pd.DataFrame,
    x: str,
    flags: dict[str, str],
    title: str = "",
    height: int = 250,
) -> go.Figure:
    """Stem-plot style timeline for binary flag columns.
    flags = {col_name: display_label}
    """
    fig = go.Figure()
    palette = [COLORS["danger"], COLORS["warn"], COLORS["primary"], COLORS["success"]]
    for i, (col, label) in enumerate(flags.items()):
        if col not in df.columns:
            continue
        active = df[df[col] == 1]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=active[x],
            y=[i + 1] * len(active),
            mode="markers",
            marker=dict(symbol="line-ns", size=12, color=color, line=dict(width=2, color=color)),
            name=label,
        ))
    fig.update_layout(
        title=title,
        height=height,
        yaxis=dict(tickvals=list(range(1, len(flags) + 1)), ticktext=list(flags.values()), showgrid=False),
        **_base_layout(),
    )
    return fig
