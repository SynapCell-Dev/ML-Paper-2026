"""Page 4 -- Multi-Scale Encoding (Paper 2: Encoder).

For researchers and students: explains why EEG requires multi-scale
feature extraction and how dilated convolutions achieve it.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.eeg_synth import BAND_COLORS, BANDS, generate_band_signals, generate_eeg
from utils.style import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_CARD,
    BG_PRIMARY,
    FOOTER_HTML,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    PAPER_COLORS,
    callout_box,
    inject_css,
    metric_card,
    paper_badge,
    paper_identity_banner,
    render_bottom_nav,
    render_paper_legend_sidebar,
    section_header,
)
from utils.viz import (
    MODEL_COLORS,
    create_architecture_diagram,
    plot_eeg_signal,
)

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(4)

render_paper_legend_sidebar()

st.markdown(
    paper_identity_banner(
        "Paper 2",
        "CoSupFormer — Dual-path dilated CNN encoder",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY};margin-top:12px;">Multi-Scale Encoding</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:1.05rem;margin-top:-8px;">'
    "How the Dual-Path Dilated CNN captures information at every timescale</p>",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "Why this matters: EEG signals contain clinically relevant information "
        "across multiple frequency bands simultaneously. A single-scale encoder "
        "inevitably misses part of the picture. This page shows why multi-scale "
        "encoding matters and how dilated convolutions achieve it efficiently.",
        "🔑",
        ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 1 -- The Problem: EEG Has Information at Many Scales
# =========================================================================
st.markdown(
    section_header(
        "The Problem: EEG Has Information at Many Scales",
        "Different brain rhythms encode different cognitive and clinical states",
    ),
    unsafe_allow_html=True,
)

band_descriptions = {
    "delta": "Deep sleep, brain injury markers -- slow oscillations (1-4 Hz)",
    "theta": "Memory encoding, drowsiness -- moderate pace (4-8 Hz)",
    "alpha": "Relaxed wakefulness, eyes-closed rest (8-12 Hz)",
    "beta": "Active thinking, motor planning (12-30 Hz)",
    "gamma": "High-level cognition, perception binding -- fast bursts (30-100 Hz)",
}

# Generate combined signal and individual bands
band_signals = generate_band_signals(duration_sec=3.0, fs=256, seed=42)
combined = generate_eeg(duration_sec=3.0, fs=256, noise_level=0.1, seed=42)
t = np.arange(len(combined)) / 256

# Combined signal plot
fig_combined = plot_eeg_signal(
    combined, fs=256,
    title="Combined EEG: slow oscillations overlaid with fast bursts",
    color=ACCENT_CYAN,
)
st.plotly_chart(fig_combined, use_container_width=True)

st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.95rem;">'
    "The signal above is a mix of five canonical frequency bands. "
    "Expand below to see each band in isolation and understand what "
    "information lives at each timescale.</p>",
    unsafe_allow_html=True,
)

with st.expander("Show individual frequency bands", expanded=True):
    for band_name, sig_arr in band_signals.items():
        low, high = BANDS[band_name]
        color = BAND_COLORS[band_name]
        desc = band_descriptions.get(band_name, "")
        fig_band = plot_eeg_signal(
            sig_arr,
            fs=256,
            title=f"{band_name.capitalize()} ({low}-{high} Hz) — {desc}",
            color=color,
        )
        fig_band.update_layout(height=180, margin=dict(l=60, r=30, t=40, b=30))
        st.plotly_chart(fig_band, use_container_width=True)

st.markdown(
    f"""
<div class="neuro-card">
    <div style="color:{TEXT_PRIMARY};font-size:1rem;line-height:1.7;">
        <strong>Key takeaway:</strong> each band encodes different brain states:
    </div>
    <ul style="color:{TEXT_SECONDARY};font-size:0.93rem;line-height:1.8;margin-top:8px;">
        <li><strong style="color:{BAND_COLORS['delta']};">Delta</strong> dominates
            during deep sleep and is a marker for sedation depth.</li>
        <li><strong style="color:{BAND_COLORS['theta']};">Theta</strong> rises
            during memory encoding and is altered in ADHD.</li>
        <li><strong style="color:{BAND_COLORS['alpha']};">Alpha</strong> reflects
            relaxed wakefulness; it is the first rhythm suppressed by many drugs.</li>
        <li><strong style="color:{BAND_COLORS['beta']};">Beta</strong> indicates
            active cortical processing and motor planning.</li>
        <li><strong style="color:{BAND_COLORS['gamma']};">Gamma</strong> is linked
            to high-level cognition and is an emerging biomarker for cognitive enhancers.</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "A drug that suppresses alpha rhythms while boosting gamma would be "
        "invisible to an encoder that only 'sees' one timescale.",
        "💊",
        ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 2 -- Dilated Convolutions Explained
# =========================================================================
st.markdown(
    section_header(
        "Dilated Convolutions Explained",
        "Expanding the receptive field without adding parameters",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;">'
    "Standard convolutions have a receptive field equal to their kernel size. "
    "To capture slow oscillations you would need enormous kernels with millions "
    "of parameters. <strong style=\"color:{0};\">Dilated convolutions</strong> "
    "solve this by inserting gaps between kernel taps, expanding the receptive "
    "field while keeping the parameter count fixed.</p>".format(ACCENT_CYAN),
    unsafe_allow_html=True,
)

col_k, col_d = st.columns(2)
with col_k:
    kernel_size = st.slider("Kernel size", min_value=3, max_value=15, value=3, step=2)
with col_d:
    dilation_rate = st.slider("Dilation rate", min_value=1, max_value=8, value=1)

receptive_field = kernel_size + (kernel_size - 1) * (dilation_rate - 1)

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown(metric_card("Kernel Size", str(kernel_size)), unsafe_allow_html=True)
with mc2:
    st.markdown(metric_card("Dilation Rate", str(dilation_rate)), unsafe_allow_html=True)
with mc3:
    st.markdown(
        metric_card("Receptive Field", f"{receptive_field} samples"),
        unsafe_allow_html=True,
    )

# Visual explanation: standard vs dilated columns
col_std, col_dil = st.columns(2)

std_rf = kernel_size  # standard conv always has dilation=1

with col_std:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {TEXT_SECONDARY};">
    <div style="color:{TEXT_PRIMARY};font-weight:600;text-align:center;margin-bottom:8px;">
        Standard Conv (dilation=1)</div>
    <div style="color:{TEXT_SECONDARY};text-align:center;font-size:0.9rem;">
        Receptive field = <strong style="color:{TEXT_PRIMARY};">{std_rf}</strong> samples<br>
        The kernel reads {kernel_size} consecutive positions.<br>
        Good for <strong style="color:{ACCENT_CYAN};">high-frequency</strong> detail.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with col_dil:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {ACCENT_CYAN};">
    <div style="color:{TEXT_PRIMARY};font-weight:600;text-align:center;margin-bottom:8px;">
        Dilated Conv (dilation={dilation_rate})</div>
    <div style="color:{TEXT_SECONDARY};text-align:center;font-size:0.9rem;">
        Receptive field = <strong style="color:{ACCENT_CYAN};">{receptive_field}</strong> samples<br>
        Same {kernel_size} taps, but spaced {dilation_rate} apart.<br>
        Captures <strong style="color:{ACCENT_PURPLE};">low-frequency</strong> patterns.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

# Plotly figure: kernel positions on a synthetic signal
n_vis = max(receptive_field + 10, 40)
sig_vis = generate_eeg(duration_sec=n_vis / 256, fs=256, noise_level=0.2, seed=7)[:n_vis]
x_positions = np.arange(n_vis)

# Compute which positions the dilated kernel touches
center = n_vis // 2
half_k = kernel_size // 2
kernel_positions = [center + (i - half_k) * dilation_rate for i in range(kernel_size)]
kernel_positions = [p for p in kernel_positions if 0 <= p < n_vis]

# Also compute standard conv positions for comparison
std_positions = [center + (i - half_k) for i in range(kernel_size)]
std_positions = [p for p in std_positions if 0 <= p < n_vis]

fig_kernel = go.Figure()
fig_kernel.add_trace(
    go.Scatter(
        x=x_positions,
        y=sig_vis,
        mode="lines",
        line=dict(color=TEXT_SECONDARY, width=1),
        name="Signal",
    )
)
# Standard conv positions (grey)
fig_kernel.add_trace(
    go.Scatter(
        x=std_positions,
        y=sig_vis[std_positions],
        mode="markers",
        marker=dict(color=TEXT_SECONDARY, size=9, symbol="circle-open", line=dict(width=2)),
        name=f"Standard conv (RF={std_rf})",
    )
)
# Dilated conv positions (cyan)
fig_kernel.add_trace(
    go.Scatter(
        x=kernel_positions,
        y=sig_vis[kernel_positions],
        mode="markers",
        marker=dict(color=ACCENT_CYAN, size=12, symbol="circle"),
        name=f"Dilated conv (RF={receptive_field})",
    )
)
# Shade receptive field span for dilated conv
if len(kernel_positions) >= 2:
    rf_start = min(kernel_positions)
    rf_end = max(kernel_positions)
    fig_kernel.add_vrect(
        x0=rf_start - 0.5,
        x1=rf_end + 0.5,
        fillcolor=ACCENT_CYAN,
        opacity=0.08,
        line_width=0,
        annotation_text="Dilated receptive field",
        annotation_position="top left",
        annotation_font_color=ACCENT_CYAN,
    )

fig_kernel.update_layout(
    title=f"Positions Covered: kernel_size={kernel_size}, dilation={dilation_rate}",
    paper_bgcolor=BG_PRIMARY,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    xaxis_title="Sample index",
    yaxis_title="Amplitude",
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
    margin=dict(l=60, r=30, t=50, b=50),
    height=350,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
st.plotly_chart(fig_kernel, use_container_width=True)

st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.9rem;">'
    "Each <strong style=\"color:{0}\">cyan dot</strong> is a position the dilated "
    "kernel reads; <strong>open circles</strong> show where a standard conv would "
    "read. By increasing dilation, the kernel covers a wider span of the signal "
    "using the <em>same number of parameters</em>. This lets it capture "
    "low-frequency patterns without expensive large kernels.</p>".format(ACCENT_CYAN),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 3 -- The Dual-Path Encoder
# =========================================================================
st.markdown(
    section_header(
        "The Dual-Path Encoder",
        "Two parallel convolution paths: one for fast, one for slow rhythms",
    ),
    unsafe_allow_html=True,
)

# Architecture diagram
arch_html = create_architecture_diagram("dual_path_encoder")
st.markdown(
    f'<div class="neuro-card">{arch_html}</div>',
    unsafe_allow_html=True,
)

# Side-by-side explanation
path1_col, path2_col = st.columns(2)

with path1_col:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {ACCENT_CYAN};">
    <div style="color:{ACCENT_CYAN};font-size:1.1rem;font-weight:700;margin-bottom:8px;">
        Path 1: Small Kernels &rarr; High-Frequency</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.6;">
        <strong>Kernel size:</strong> 3-5<br>
        <strong>Dilation:</strong> 1 (standard)<br>
        <strong>Captures:</strong> High-frequency details (beta, gamma)<br>
        <strong>Receptive field:</strong> Narrow -- a few milliseconds<br><br>
        <em style="color:{TEXT_PRIMARY};">Think of this as the "microscope" path:
        it resolves fast, fine-grained oscillatory events.</em>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with path2_col:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {ACCENT_PURPLE};">
    <div style="color:{ACCENT_PURPLE};font-size:1.1rem;font-weight:700;margin-bottom:8px;">
        Path 2: Large Dilated Kernels &rarr; Low-Frequency</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.6;">
        <strong>Kernel size:</strong> 7-15<br>
        <strong>Dilation:</strong> 2-8<br>
        <strong>Captures:</strong> Low-frequency patterns (delta, theta)<br>
        <strong>Receptive field:</strong> Wide -- spanning hundreds of ms<br><br>
        <em style="color:{TEXT_PRIMARY};">Think of this as the "telescope" path:
        it captures slow, broad oscillatory trends.</em>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

# What each path "sees" -- using generate_band_signals
st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.95rem;margin-top:16px;">'
    "Below: the frequency bands each path is best at capturing.</p>",
    unsafe_allow_html=True,
)

band_sigs = generate_band_signals(duration_sec=2.0, fs=256, seed=10)
t_band = np.arange(int(2.0 * 256)) / 256

fig_paths = go.Figure()
# Path 1 sees high-freq
hf_signal = band_sigs["beta"] + band_sigs["gamma"]
fig_paths.add_trace(
    go.Scatter(
        x=t_band, y=hf_signal, mode="lines",
        line=dict(color=ACCENT_CYAN, width=1.5),
        name="Path 1 sees: Beta + Gamma",
    )
)
# Path 2 sees low-freq
lf_signal = band_sigs["delta"] + band_sigs["theta"]
fig_paths.add_trace(
    go.Scatter(
        x=t_band, y=lf_signal - 4, mode="lines",
        line=dict(color=ACCENT_PURPLE, width=1.5),
        name="Path 2 sees: Delta + Theta",
    )
)

fig_paths.update_layout(
    title="What Each Path Captures",
    paper_bgcolor=BG_PRIMARY,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (offset for clarity)",
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b"),
    margin=dict(l=60, r=30, t=50, b=50),
    height=350,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)
st.plotly_chart(fig_paths, use_container_width=True)

st.markdown(
    callout_box(
        "After both paths process the signal independently, their outputs are "
        "concatenated to form a multi-scale representation. This is then passed "
        "to the attention module (see next page) for cross-channel reasoning.",
        "🔗",
        ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 4 -- Why This Matters for Pharma
# =========================================================================
st.markdown(
    section_header(
        "Why This Matters for Pharma",
        "Drug effects manifest across different frequency bands",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="neuro-card">
    <div style="color:{TEXT_PRIMARY};font-size:1rem;line-height:1.7;">
        Many CNS drugs produce <em>band-specific</em> EEG biomarkers:
    </div>
    <ul style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.8;margin-top:8px;">
        <li><strong style="color:{BAND_COLORS['alpha']};">Alpha</strong> power
            increases with benzodiazepines (anxiolytic effect).</li>
        <li><strong style="color:{BAND_COLORS['beta']};">Beta</strong> suppression
            is a marker for sedative compounds.</li>
        <li><strong style="color:{BAND_COLORS['gamma']};">Gamma</strong> changes
            correlate with antipsychotic action on glutamate pathways.</li>
        <li><strong style="color:{BAND_COLORS['delta']};">Delta</strong> alterations
            track sleep-stage transitions under hypnotics.</li>
    </ul>
    <div style="color:{TEXT_PRIMARY};font-size:0.95rem;line-height:1.7;margin-top:12px;">
        A single-scale encoder would miss either alpha or gamma effects.
        The dual-path architecture captures both simultaneously.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Simulated drug effect illustration
st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.95rem;">'
    "Example: a drug that <strong>boosts alpha</strong> and <strong>suppresses gamma</strong>.</p>",
    unsafe_allow_html=True,
)

baseline_amps = {"delta": 1.0, "theta": 0.7, "alpha": 1.0, "beta": 0.4, "gamma": 0.15}
drug_amps = {"delta": 1.0, "theta": 0.7, "alpha": 2.0, "beta": 0.4, "gamma": 0.05}

col_base, col_drug = st.columns(2)

with col_base:
    sig_base = generate_eeg(
        duration_sec=2.0, fs=256, band_amplitudes=baseline_amps, noise_level=0.1, seed=20,
    )
    fig_base = plot_eeg_signal(sig_base, fs=256, title="Baseline EEG", color=TEXT_SECONDARY)
    fig_base.update_layout(height=250)
    st.plotly_chart(fig_base, use_container_width=True)

with col_drug:
    sig_drug = generate_eeg(
        duration_sec=2.0, fs=256, band_amplitudes=drug_amps, noise_level=0.1, seed=20,
    )
    fig_drug = plot_eeg_signal(sig_drug, fs=256, title="Post-Drug EEG", color=ACCENT_GREEN)
    fig_drug.update_layout(height=250)
    st.plotly_chart(fig_drug, use_container_width=True)

st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.9rem;text-align:center;">'
    "Notice the increased slow oscillation (alpha boost) and reduced fast ripples "
    "(gamma suppression). The dual-path encoder picks up both changes.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "The dual-path encoder is the foundation that makes CoSupFormer robust to "
        "the multi-scale nature of pharmacological EEG effects. Next, the attention "
        "and gating modules (Page 4) learn which channels and time-points matter most.",
        "🎯",
        ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render_bottom_nav(
    prev_page=("pages/3_Detection_Challenge.py", "Detection Challenge"),
    next_page=("pages/5_Attention_and_Gating.py", "Attention & Gating"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
