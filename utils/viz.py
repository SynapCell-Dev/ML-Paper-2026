"""Shared Plotly visualization helpers for EEG DeepDive."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.style import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_CARD,
    BG_PRIMARY,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

# Consistent Plotly layout defaults
_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG_PRIMARY,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)

MODEL_COLORS = [
    ACCENT_CYAN,
    ACCENT_PURPLE,
    ACCENT_GREEN,
    ACCENT_AMBER,
    ACCENT_RED,
    "#ec4899",
    "#6366f1",
    "#14b8a6",
    "#f97316",
    "#a855f7",
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a 6-char hex color to an rgba() string Plotly accepts."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _apply_layout(fig: go.Figure, **overrides) -> go.Figure:
    layout = {**_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**layout)
    return fig


def plot_eeg_signal(
    signal: np.ndarray,
    fs: int = 256,
    title: str = "EEG Signal",
    color: str = ACCENT_CYAN,
    y_label: str = "Amplitude (µV)",
) -> go.Figure:
    """Plot a single-channel EEG time series."""
    t = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", line=dict(color=color, width=1.2), name="EEG"))
    _apply_layout(fig, title=title, xaxis_title="Time (s)", yaxis_title=y_label)
    return fig


def plot_multichannel_eeg(
    signals: np.ndarray,
    fs: int = 256,
    channel_names: list[str] | None = None,
    colors: list[str] | None = None,
    title: str = "Multi-channel EEG",
    spacing: float = 0.0,
) -> go.Figure:
    """Plot stacked multi-channel EEG (signals shape: n_channels x n_samples)."""
    n_ch = signals.shape[0]
    if channel_names is None:
        channel_names = [f"Ch {i+1}" for i in range(n_ch)]
    if colors is None:
        colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_ch)]

    t = np.arange(signals.shape[1]) / fs

    if spacing == 0.0:
        spacing = np.max(np.abs(signals)) * 2.5

    fig = go.Figure()
    for i in range(n_ch):
        offset = -i * spacing
        fig.add_trace(
            go.Scatter(
                x=t,
                y=signals[i] + offset,
                mode="lines",
                line=dict(color=colors[i], width=1.2),
                name=channel_names[i],
            )
        )

    yticks = [-i * spacing for i in range(n_ch)]
    _apply_layout(
        fig,
        title=title,
        xaxis_title="Time (s)",
        yaxis=dict(
            tickvals=yticks,
            ticktext=channel_names,
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        height=120 * n_ch + 100,
    )
    return fig


def plot_spectrogram(
    Zxx: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    title: str = "Spectrogram",
    colorscale: str = "Viridis",
    max_freq: float | None = None,
) -> go.Figure:
    """Plot STFT magnitude as a heatmap."""
    if max_freq is not None:
        idx = freqs <= max_freq
        Zxx = Zxx[idx, :]
        freqs = freqs[idx]

    fig = go.Figure(
        go.Heatmap(
            z=Zxx,
            x=times,
            y=freqs,
            colorscale=colorscale,
            colorbar=dict(title="Magnitude"),
        )
    )
    _apply_layout(fig, title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
    return fig


def plot_attention_heatmap(
    matrix: np.ndarray,
    labels_x: list[str] | None = None,
    labels_y: list[str] | None = None,
    title: str = "Attention Weights",
) -> go.Figure:
    """Attention scores visualization."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels_x,
            y=labels_y,
            colorscale="Teal",
            colorbar=dict(title="Weight"),
        )
    )
    _apply_layout(fig, title=title, height=500)
    return fig


def plot_embedding_scatter(
    data: list[dict],
    color_col: str = "label",
    title: str = "Embedding Space",
    color_map: dict | None = None,
) -> go.Figure:
    """PCA / t-SNE scatter plot from a list of dicts with x, y, label keys."""
    df = pd.DataFrame(data)
    labels = df[color_col].unique()
    if color_map is None:
        color_map = {lab: MODEL_COLORS[i % len(MODEL_COLORS)] for i, lab in enumerate(labels)}

    fig = go.Figure()
    for lab in labels:
        sub = df[df[color_col] == lab]
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                marker=dict(color=color_map.get(lab, ACCENT_CYAN), size=6, opacity=0.7),
                name=str(lab),
            )
        )
    _apply_layout(fig, title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    return fig


def plot_comparison_bars(
    df: pd.DataFrame,
    metric_col: str,
    model_col: str,
    dataset_col: str | None = None,
    title: str = "Model Comparison",
) -> go.Figure:
    """Grouped bar chart for model comparison."""
    models = df[model_col].unique()

    if dataset_col and dataset_col in df.columns:
        datasets = df[dataset_col].unique()
        fig = go.Figure()
        for i, model in enumerate(models):
            sub = df[df[model_col] == model]
            fig.add_trace(
                go.Bar(
                    x=sub[dataset_col],
                    y=sub[metric_col],
                    name=model,
                    marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                )
            )
        _apply_layout(fig, title=title, barmode="group", yaxis_title=metric_col)
    else:
        fig = go.Figure(
            go.Bar(
                x=df[model_col],
                y=df[metric_col],
                marker_color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(df))],
            )
        )
        _apply_layout(fig, title=title, yaxis_title=metric_col)
    return fig


def plot_ablation_chart(
    df: pd.DataFrame,
    variant_col: str = "variant",
    metric_col: str = "balanced_accuracy",
    baseline_label: str = "Full Model",
    title: str = "Ablation Study",
) -> go.Figure:
    """Bar chart showing ablation results with baseline comparison."""
    colors = []
    for _, row in df.iterrows():
        if row[variant_col] == baseline_label:
            colors.append(ACCENT_CYAN)
        else:
            colors.append(ACCENT_PURPLE)

    fig = go.Figure(
        go.Bar(x=df[variant_col], y=df[metric_col], marker_color=colors)
    )
    _apply_layout(fig, title=title, yaxis_title=metric_col)
    return fig


def plot_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    band_ranges: dict | None = None,
    title: str = "Power Spectral Density",
) -> go.Figure:
    """Plot PSD with optional colored band regions."""
    from utils.eeg_synth import BAND_COLORS, BANDS

    if band_ranges is None:
        band_ranges = BANDS

    fig = go.Figure()

    # Band shading
    for band, (low, high) in band_ranges.items():
        color = BAND_COLORS.get(band, ACCENT_CYAN)
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([freqs[mask], freqs[mask][::-1]]),
                    y=np.concatenate([psd[mask], np.zeros(mask.sum())]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, alpha=0.2),
                    line=dict(width=0),
                    name=f"{band} ({low}-{high} Hz)",
                    hoverinfo="name",
                )
            )

    fig.add_trace(
        go.Scatter(x=freqs, y=psd, mode="lines", line=dict(color=TEXT_PRIMARY, width=1.5), name="PSD")
    )
    _apply_layout(fig, title=title, xaxis_title="Frequency (Hz)", yaxis_title="Power (µV²/Hz)")
    return fig


def plot_timeline(events: list[dict], title: str = "Research Timeline") -> go.Figure:
    """Simple horizontal timeline. events = [{year, label, color, description}]."""
    fig = go.Figure()
    for i, ev in enumerate(events):
        fig.add_trace(
            go.Scatter(
                x=[ev["year"]],
                y=[0],
                mode="markers+text",
                marker=dict(size=18, color=ev.get("color", ACCENT_CYAN)),
                text=[ev["label"]],
                textposition="top center",
                textfont=dict(size=11, color=TEXT_PRIMARY),
                name=ev["label"],
                hovertext=ev.get("description", ""),
            )
        )
    _apply_layout(
        fig,
        title=title,
        yaxis=dict(visible=False, range=[-1, 2]),
        xaxis=dict(gridcolor="#1e293b", dtick=1),
        height=250,
        showlegend=False,
    )
    return fig


def create_architecture_diagram(arch_type: str) -> str:
    """Return HTML/CSS for architecture diagrams."""
    if arch_type == "dual_path_encoder":
        return _dual_path_html()
    elif arch_type == "spechinet":
        return _spechinet_html()
    elif arch_type == "spectral_moe":
        return _spectral_moe_html()
    elif arch_type == "cnn_transformer":
        return _cnn_transformer_html()
    return "<p>Unknown architecture type.</p>"


def _box(label: str, color: str, width: str = "140px") -> str:
    return (
        f'<div style="display:inline-block;background:{color}22;border:2px solid {color};'
        f'border-radius:8px;padding:8px 14px;margin:4px;color:{color};'
        f'font-weight:600;font-size:0.85rem;min-width:{width};text-align:center;">{label}</div>'
    )


def _arrow_down() -> str:
    return f'<div style="text-align:center;color:{TEXT_SECONDARY};font-size:1.4rem;">&#8595;</div>'


def _arrow_right() -> str:
    return f'<span style="color:{TEXT_SECONDARY};font-size:1.4rem;margin:0 8px;">&#8594;</span>'


def _dual_path_html() -> str:
    return f"""
    <div style="text-align:center;padding:16px;">
        <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:12px;">Dual-Path Dilated CNN Encoder</div>
        {_box("Input EEG Patches", TEXT_SECONDARY)}
        {_arrow_down()}
        <div style="display:flex;justify-content:center;gap:24px;margin:8px 0;">
            <div style="text-align:center;">
                {_box("Path 1: Small Kernels", ACCENT_CYAN)}
                <div style="color:{TEXT_SECONDARY};font-size:0.75rem;">Fine-grained features<br>High-frequency</div>
            </div>
            <div style="text-align:center;">
                {_box("Path 2: Large Dilated Kernels", ACCENT_PURPLE)}
                <div style="color:{TEXT_SECONDARY};font-size:0.75rem;">Broad patterns<br>Low-frequency</div>
            </div>
        </div>
        {_arrow_down()}
        {_box("Concatenation + Dropout", ACCENT_GREEN)}
        {_arrow_down()}
        {_box("Multi-scale Representation", ACCENT_AMBER)}
    </div>
    """


def _spechinet_html() -> str:
    return f"""
    <div style="text-align:center;padding:16px;">
        <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:12px;">SpecHi-Net: U-shaped Hierarchical Encoder-Decoder</div>
        {_box("STFT Spectrogram Input", TEXT_SECONDARY, "200px")}
        {_arrow_down()}
        <div style="display:flex;justify-content:center;gap:8px;align-items:center;">
            <div style="text-align:center;">
                {_box("Down-1", ACCENT_CYAN, "80px")}
                {_arrow_down()}
                {_box("Down-2", ACCENT_CYAN, "80px")}
                {_arrow_down()}
                {_box("Down-3", ACCENT_CYAN, "80px")}
            </div>
            <div style="color:{TEXT_SECONDARY};font-size:0.75rem;padding:0 12px;">
                ← Skip<br>connections →
            </div>
            <div style="text-align:center;">
                {_box("Up-1", ACCENT_PURPLE, "80px")}
                {_arrow_down()}
                {_box("Up-2", ACCENT_PURPLE, "80px")}
                {_arrow_down()}
                {_box("Up-3", ACCENT_PURPLE, "80px")}
            </div>
        </div>
        {_arrow_down()}
        <div style="display:flex;justify-content:center;gap:8px;margin:8px 0;">
            {_box("Global Transformer", ACCENT_GREEN, "160px")}
        </div>
        {_arrow_down()}
        {_box("Multi-level Reconstruction Loss", ACCENT_AMBER, "220px")}
    </div>
    """


def _spectral_moe_html() -> str:
    return f"""
    <div style="text-align:center;padding:16px;">
        <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:12px;">Spectral Mixture of Experts</div>
        {_box("Input EEG", TEXT_SECONDARY)}
        {_arrow_down()}
        {_box("PSD Computation", ACCENT_AMBER, "160px")}
        {_arrow_down()}
        {_box("Spectral Gating Network", ACCENT_GREEN, "180px")}
        {_arrow_down()}
        <div style="display:flex;justify-content:center;gap:12px;margin:8px 0;">
            {_box("Expert 1", ACCENT_CYAN, "100px")}
            {_box("Expert 2", ACCENT_PURPLE, "100px")}
            {_box("Expert 3", ACCENT_AMBER, "100px")}
        </div>
        {_arrow_down()}
        {_box("Weighted Combination", ACCENT_GREEN, "180px")}
        {_arrow_down()}
        {_box("Task Prediction", TEXT_PRIMARY, "140px")}
    </div>
    """


def _cnn_transformer_html() -> str:
    return f"""
    <div style="text-align:center;padding:16px;">
        <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;margin-bottom:12px;">CNN + Transformer (Paper 1)</div>
        {_box("Raw EEG Segment", TEXT_SECONDARY)}
        {_arrow_down()}
        {_box("CNN Feature Extractor", ACCENT_CYAN, "180px")}
        {_arrow_down()}
        {_box("Transformer Encoder", ACCENT_PURPLE, "180px")}
        {_arrow_down()}
        {_box("Classification Head", ACCENT_GREEN, "160px")}
        {_arrow_down()}
        <div style="display:flex;justify-content:center;gap:12px;">
            {_box("Seizure", ACCENT_RED, "80px")}
            {_box("Non-seizure", ACCENT_GREEN, "100px")}
        </div>
    </div>
    """
