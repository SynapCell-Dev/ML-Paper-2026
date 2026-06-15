"""Page 3 — The Detection Challenge (Paper 1)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.data_loader import load_results
from utils.eeg_synth import generate_seizure_signal
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

inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(3)

render_paper_legend_sidebar()

# ---- Header ----
st.markdown(
    paper_identity_banner(
        "Paper 1",
        "Thomas et al. — Seizure detection, not just classification",
    ),
    unsafe_allow_html=True,
)
st.title("🔍 The Detection Challenge")
st.markdown(
    callout_box(
        "Why 95% accuracy doesn't mean your seizure detector works — "
        "and why the distinction between classification and detection matters for every preclinical study.",
        "⚠️",
        ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1 — Classification vs Detection
# =====================================================================
st.markdown(section_header("1. Classification vs. Detection — The Hidden Gap"), unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {ACCENT_GREEN};">
    <h4 style="color:{ACCENT_GREEN};margin-top:0;">Pre-processing I — "Easy Mode"</h4>
    <p style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.7;">
        Each EEG segment is pre-labelled as <strong>entirely seizure</strong> or
        <strong>entirely seizure-free</strong>. The model only sees clean,
        unambiguous windows.<br><br>
        <strong>Result:</strong> High reported accuracy, but the hard part
        (finding where a seizure starts and ends) has already been done for the model.
    </p>
</div>""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
<div class="neuro-card" style="border-top:3px solid {ACCENT_RED};">
    <h4 style="color:{ACCENT_RED};margin-top:0;">Pre-processing II — "Real World"</h4>
    <p style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.7;">
        Segments are cut from a <strong>continuous recording</strong> with a
        sliding window. A segment can contain <em>mixed</em> seizure and
        non-seizure activity.<br><br>
        <strong>Result:</strong> The model must discover seizure boundaries itself
        — and performance drops dramatically.
    </p>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box("Classification <em>presupposes</em> what detection must <em>discover</em>.", "💡", ACCENT_CYAN),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 2 — Interactive Pipeline Comparison
# =====================================================================
st.markdown(section_header("2. Interactive Pipeline Comparison", "See how window size and shift affect seizure detection"), unsafe_allow_html=True)

sig_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    window_size = st.select_slider("Window size (s)", options=[2, 4], value=4, key="win_sz")
    shift = st.select_slider("Shift (s)", options=[0.5, 1.0, 2.0], value=2.0, key="shift")

signal, labels = generate_seizure_signal(duration_sec=60.0, fs=100, seed=42)
fs = 100
t = np.arange(len(signal)) / fs

with sig_col:
    fig = go.Figure()
    # Background EEG
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", line=dict(color=ACCENT_CYAN, width=0.8), name="EEG"))
    # Highlight seizure regions
    seizure_starts = np.where(np.diff(labels.astype(int)) == 1)[0]
    seizure_ends = np.where(np.diff(labels.astype(int)) == -1)[0]
    for s, e in zip(seizure_starts, seizure_ends):
        fig.add_vrect(x0=s / fs, x1=e / fs, fillcolor=ACCENT_RED, opacity=0.2, line_width=0, annotation_text="Seizure", annotation_position="top left")

    # Show window segments
    n_windows = int((60.0 - window_size) / shift) + 1
    shown = 0
    for i in range(min(n_windows, 200)):
        start_s = i * shift
        end_s = start_s + window_size
        if end_s > 60.0:
            break
        i_s = int(start_s * fs)
        i_e = int(end_s * fs)
        seg_labels = labels[i_s:i_e]
        frac_seizure = seg_labels.mean()
        if 0 < frac_seizure < 1 and shown < 6:
            fig.add_vrect(x0=start_s, x1=end_s, fillcolor=ACCENT_AMBER, opacity=0.12, line_width=1, line_color=ACCENT_AMBER)
            shown += 1

    fig.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY),
        title="60-second EEG with Seizure Events",
        xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
        height=350, margin=dict(l=60, r=30, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Orange highlighted windows contain **mixed** seizure/non-seizure activity — "
               f"these are the segments that make detection harder than classification. "
               f"Window = {window_size}s, Shift = {shift}s")

# =====================================================================
# Section 3 — Architecture Comparison
# =====================================================================
st.markdown(section_header("3. Architecture Comparison", "Which models perform best — and how much does the task matter?"), unsafe_allow_html=True)

st.markdown(create_architecture_diagram("cnn_transformer"), unsafe_allow_html=True)

# Classification results
clf_df = load_results("thomas_classification.csv")
fig_clf = go.Figure(
    go.Bar(
        x=clf_df["model"], y=clf_df["f1"],
        marker_color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(clf_df))],
        text=[f"{v:.3f}" for v in clf_df["f1"]],
        textposition="outside",
    )
)
fig_clf.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title="Classification F1-Score (Table 1)",
    yaxis_title="F1-Score", yaxis=dict(range=[0, 1.05], gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    height=400, margin=dict(l=60, r=30, t=50, b=80),
)
st.plotly_chart(fig_clf, use_container_width=True)

# Detection vs Classification comparison
det_df = load_results("thomas_detection.csv")
# Get the best detection setting for each model (4s / 0.5s)
models_compare = ["CNN – 6 layers", "CNN-6 + biLSTM", "Custom. U-Time", "CNN + Transformer"]
clf_f1 = []
det_f1 = []
for m in models_compare:
    clf_row = det_df[(det_df["model"] == m) & (det_df["task"] == "Classification") & (det_df["window"] == "4s")]
    det_row = det_df[(det_df["model"] == m) & (det_df["task"] == "Detection") & (det_df["window"] == "4s") & (det_df["shift"] == "0.5s")]
    clf_f1.append(clf_row["f1"].values[0] if len(clf_row) > 0 else 0)
    det_f1.append(det_row["f1"].values[0] if len(det_row) > 0 else 0)

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(x=models_compare, y=clf_f1, name="Classification", marker_color=ACCENT_CYAN))
fig_cmp.add_trace(go.Bar(x=models_compare, y=det_f1, name="Detection", marker_color=ACCENT_RED))
fig_cmp.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title="Classification vs. Detection F1-Score (4s window)",
    yaxis_title="F1-Score", barmode="group",
    yaxis=dict(range=[0, 1.05], gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=400, margin=dict(l=60, r=30, t=50, b=80),
)
st.plotly_chart(fig_cmp, use_container_width=True)

st.markdown(
    callout_box(
        "The CNN+Transformer is the only architecture that achieves meaningful detection performance. "
        "U-Time completely fails at detection (F1=0.0) despite reasonable classification scores.",
        "📉", ACCENT_RED,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 4 — Cross-Species Generalization
# =====================================================================
st.markdown(section_header("4. Cross-Species Generalization", "Trained on mice, tested on humans"), unsafe_allow_html=True)

cross_df = load_results("thomas_crossspecies.csv")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(metric_card("Best Cross-Species F1", "0.935", "CNN+Transformer, balanced"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Training Data", "Mouse EEG", "Dataset 1"), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Test Data", "Human EEG", "Bonn Dataset"), unsafe_allow_html=True)

st.dataframe(
    cross_df[["model", "subset", "recall", "f1"]].rename(
        columns={"model": "Model", "subset": "Subset", "recall": "Recall", "f1": "F1-Score"}
    ),
    use_container_width=True,
    hide_index=True,
)

fig_cross = go.Figure()
for i, model in enumerate(cross_df["model"].unique()):
    sub = cross_df[cross_df["model"] == model]
    fig_cross.add_trace(
        go.Bar(x=sub["subset"], y=sub["f1"], name=model, marker_color=MODEL_COLORS[i])
    )
fig_cross.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title="Cross-Species F1-Score by Subset",
    yaxis_title="F1-Score", barmode="group",
    yaxis=dict(range=[0, 1.05], gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=400, margin=dict(l=60, r=30, t=50, b=80),
)
st.plotly_chart(fig_cross, use_container_width=True)

st.markdown(
    callout_box(
        "Key takeaway for pharma: preclinical models <strong>CAN</strong> translate. "
        "The CNN+Transformer trained on mouse EEG achieves F1 = 0.935 on human data, "
        "demonstrating that attention-based architectures learn cross-species seizure signatures.",
        "🔬", ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 5 — Key Takeaway
# =====================================================================
st.markdown("---")
st.markdown(
    f"""
<div style="background:{ACCENT_AMBER}11;border:2px solid {ACCENT_AMBER};border-radius:12px;
            padding:24px 28px;margin:24px 0;text-align:center;">
    <div style="font-size:1.4rem;margin-bottom:8px;">🎯</div>
    <div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:600;line-height:1.6;">
        If your seizure detection paper only reports classification metrics,<br>
        it likely overestimates real-world performance.
    </div>
</div>""",
    unsafe_allow_html=True,
)

render_bottom_nav(
    prev_page=("pages/2_DeepLearning_Primer.py", "Deep Learning Primer"),
    next_page=("pages/4_MultiScale_Encoding.py", "Multi-Scale Encoding"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
