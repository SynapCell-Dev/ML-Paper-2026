"""Page 5 — Attention & Gating (Paper 2 — Attention Module)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_attention, load_results
from utils.eeg_synth import generate_multichannel_eeg
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
from utils.viz import MODEL_COLORS, plot_attention_heatmap, plot_multichannel_eeg

inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(5)

render_paper_legend_sidebar()

# ---- Header ----
st.markdown(
    paper_identity_banner(
        "Paper 2",
        "CoSupFormer — Global attention + channel gating",
    ),
    unsafe_allow_html=True,
)
st.title("🎯 Attention & Gating")
st.markdown(
    callout_box(
        "How CoSupFormer models complex dependencies across channels and time, "
        "and why a gating mechanism is essential for noisy real-world EEG.",
        "🧠",
        ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1 — Three Types of Interactions
# =====================================================================
st.markdown(section_header("1. Three Types of Interactions", "What makes CoSupFormer's attention truly global"), unsafe_allow_html=True)

i1, i2, i3 = st.columns(3)
with i1:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_CYAN};min-height:180px;">
        <h4 style="color:{ACCENT_CYAN};margin-top:0;">Within-Channel</h4>
        <p style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Patches from the <strong>same channel</strong> interact with each other.
        Captures temporal dynamics within a single electrode.
        </p></div>""",
        unsafe_allow_html=True,
    )
with i2:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_GREEN};min-height:180px;">
        <h4 style="color:{ACCENT_GREEN};margin-top:0;">Cross-Channel, Same Time</h4>
        <p style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Patches at the <strong>same time</strong> from <strong>different channels</strong>.
        Captures spatial relationships across brain regions.
        </p></div>""",
        unsafe_allow_html=True,
    )
with i3:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_PURPLE};min-height:180px;">
        <h4 style="color:{ACCENT_PURPLE};margin-top:0;">Cross-Channel, Different Time ✨</h4>
        <p style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">
        Patches from <strong>different channels</strong> at <strong>different times</strong>.
        The <em>novel contribution</em> — captures delayed interactions across brain regions.
        </p></div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        'Like comparing frontal electrode activity <em>now</em> with occipital electrode activity '
        '<em>a few hundred milliseconds ago</em> — these delayed cross-channel interactions '
        'reveal how neural information propagates across brain regions.',
        "🏥", ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# Attention heatmap
matrix, labels_x, labels_y = load_attention("attention_heatmap_example.json")
fig_att = plot_attention_heatmap(matrix, labels_x, labels_y, title="Global Attention Weights (4 channels × 3 time patches)")
st.plotly_chart(fig_att, use_container_width=True)

st.caption("Diagonal blocks (bright) = within-channel attention. Off-diagonal same-column positions = cross-channel same-time. Sparse off-diagonal entries = cross-channel different-time (the novel contribution).")

# =====================================================================
# Section 2 — The Gating Mechanism
# =====================================================================
st.markdown(section_header("2. The Gating Mechanism", "From soft attention to hard suppression"), unsafe_allow_html=True)

st.markdown(
    callout_box(
        "Attention <em>downweights</em> bad channels by assigning low scores, "
        "but those channels still influence downstream decisions. "
        "Gating <em>completely removes</em> them via element-wise multiplication with a sigmoid mask.",
        "🚪", ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# Interactive gating demo
noisy_channels = [3, 5]
signals = generate_multichannel_eeg(n_channels=6, duration_sec=3.0, fs=256, noisy_channels=noisy_channels, noise_boost=3.0, seed=42)
channel_names = [f"Ch {i+1}" for i in range(6)]

col_orig, col_gated = st.columns(2)

# Simulate gate values
gate_values = np.array([0.92, 0.88, 0.95, 0.08, 0.91, 0.05])  # low for noisy channels

with col_orig:
    st.markdown(f"**Original EEG** (channels 4 & 6 are noisy)")
    fig_orig = plot_multichannel_eeg(signals, fs=256, channel_names=channel_names, title="Before Gating")
    st.plotly_chart(fig_orig, use_container_width=True)

with col_gated:
    st.markdown(f"**After Gating** (noisy channels suppressed)")
    gated_signals = signals * gate_values[:, np.newaxis]
    fig_gated = plot_multichannel_eeg(gated_signals, fs=256, channel_names=channel_names, title="After Gating")
    st.plotly_chart(fig_gated, use_container_width=True)

# Gate values bar chart
fig_gates = go.Figure(
    go.Bar(
        x=channel_names,
        y=gate_values,
        marker_color=[ACCENT_RED if i in noisy_channels else ACCENT_GREEN for i in range(6)],
        text=[f"{v:.2f}" for v in gate_values],
        textposition="outside",
    )
)
fig_gates.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title="Gate Values (Sigmoid Output)",
    yaxis=dict(range=[0, 1.15], title="Gate Value", gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    height=300, margin=dict(l=60, r=30, t=50, b=50),
)
st.plotly_chart(fig_gates, use_container_width=True)
st.caption("Red bars = noisy channels, automatically suppressed by the gating mechanism.")

# =====================================================================
# Section 3 — Ablation Study
# =====================================================================
st.markdown(section_header("3. Ablation Study Results", "Table C.1: What happens when we remove each component?"), unsafe_allow_html=True)

abl_df = load_results("cosupformer_ablation.csv")

# Select metric to view
abl_metric = st.selectbox("Metric", ["accuracy", "f1", "auroc", "auprc"], index=0, key="abl_met")

datasets = abl_df["dataset"].unique()
variants = abl_df["variant"].unique()

fig_abl = go.Figure()
for i, var in enumerate(variants):
    sub = abl_df[abl_df["variant"] == var]
    fig_abl.add_trace(
        go.Bar(x=sub["dataset"], y=sub[abl_metric], name=var, marker_color=MODEL_COLORS[i % len(MODEL_COLORS)])
    )
fig_abl.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title=f"Ablation Study — {abl_metric.upper()}",
    yaxis_title=abl_metric.upper(), barmode="group",
    yaxis=dict(gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=450, margin=dict(l=60, r=30, t=50, b=80),
)
st.plotly_chart(fig_abl, use_container_width=True)

# Highlight key findings
h1, h2 = st.columns(2)
with h1:
    st.markdown(
        metric_card("TDBrain+Noise: w/o Gating", "89.51%", "↓ 4.7% accuracy drop"),
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        metric_card("MACO: w/o Gating", "73.85%", "↓ 4.2% accuracy drop"),
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "Gating is crucial on noisy datasets — removing it causes up to 4.7% accuracy drop on TDBrain+Noise "
        "and 4.2% on MACO. The contrastive loss provides a consistent but smaller improvement across all datasets.",
        "🔑", ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 4 — Performance Comparison
# =====================================================================
st.markdown(section_header("4. Performance Comparison", "CoSupFormer vs. baselines on clean and noisy data"), unsafe_allow_html=True)

tab_clean, tab_noisy = st.tabs(["📋 Clean Datasets", "📋 Noisy / Challenging Datasets"])

with tab_clean:
    clean_df = load_results("cosupformer_clean.csv")
    perf_metric = st.selectbox("Metric", ["accuracy", "f1", "auroc", "auprc"], index=0, key="clean_met")
    fig_c = go.Figure()
    for i, model in enumerate(clean_df["model"].unique()):
        sub = clean_df[clean_df["model"] == model]
        fig_c.add_trace(go.Bar(x=sub["dataset"], y=sub[perf_metric], name=model, marker_color=MODEL_COLORS[i]))
    fig_c.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY),
        title=f"Clean Datasets — {perf_metric.upper()}",
        yaxis_title=perf_metric.upper(), barmode="group",
        yaxis=dict(gridcolor="#1e293b"), xaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=450, margin=dict(l=60, r=30, t=50, b=80),
    )
    st.plotly_chart(fig_c, use_container_width=True)

with tab_noisy:
    noisy_df = load_results("cosupformer_noisy.csv")
    perf_metric_n = st.selectbox("Metric", ["accuracy", "f1", "auroc", "auprc"], index=0, key="noisy_met")
    fig_n = go.Figure()
    for i, model in enumerate(noisy_df["model"].unique()):
        sub = noisy_df[noisy_df["model"] == model]
        fig_n.add_trace(go.Bar(x=sub["dataset"], y=sub[perf_metric_n], name=model, marker_color=MODEL_COLORS[i]))
    fig_n.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY),
        title=f"Noisy/Challenging Datasets — {perf_metric_n.upper()}",
        yaxis_title=perf_metric_n.upper(), barmode="group",
        yaxis=dict(gridcolor="#1e293b"), xaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=450, margin=dict(l=60, r=30, t=50, b=80),
    )
    st.plotly_chart(fig_n, use_container_width=True)

    st.markdown(
        callout_box(
            "On the MACO dataset, CoSupFormer achieves <strong>74.9% accuracy</strong> while "
            "baselines collapse to 32-39%. The combination of gating + contrastive loss makes "
            "CoSupFormer uniquely robust to real-world noise.",
            "🏆", ACCENT_GREEN,
        ),
        unsafe_allow_html=True,
    )

render_bottom_nav(
    prev_page=("pages/4_MultiScale_Encoding.py", "Multi-Scale Encoding"),
    next_page=("pages/6_Learning_Representations.py", "Learning Representations"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
