"""Page 8 — Results & Impact."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_results
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
from utils.viz import MODEL_COLORS, plot_comparison_bars, plot_timeline

inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(8)

render_paper_legend_sidebar()

st.markdown(
    paper_identity_banner(
        "Paper 1, Paper 2, Paper 3",
        "Unified dashboard — results, limitations, and the combined research narrative",
    ),
    unsafe_allow_html=True,
)
st.title("📊 Results & Impact")
st.markdown(
    callout_box(
        "The big picture: from seizure detection to universal brain decoders — "
        "what three papers achieved and what it means for the field.",
        "🎯", ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1 — Unified Results Dashboard
# =====================================================================
st.markdown(section_header("1. Unified Results Dashboard"), unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    f"🔍 Paper 1 — Detection Challenge",
    f"🎯 Paper 2 — CoSupFormer",
    f"🏗️ Paper 3 — SpecMoE",
])

with tab1:
    st.markdown(f"{paper_badge('Paper 1')}", unsafe_allow_html=True)

    # Classification F1
    clf_df = load_results("thomas_classification.csv")
    fig1 = go.Figure(
        go.Bar(
            x=clf_df["model"], y=clf_df["f1"],
            marker_color=[MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(clf_df))],
            text=[f"{v:.3f}" for v in clf_df["f1"]], textposition="outside",
        )
    )
    fig1.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
        title="Seizure Classification F1-Score (Mouse EEG)",
        yaxis=dict(range=[0, 1.05], title="F1-Score", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        height=400, margin=dict(l=60, r=30, t=50, b=100),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Cross-species table
    st.markdown("**Cross-Species Generalization (Mouse → Human)**")
    cross_df = load_results("thomas_crossspecies.csv")
    st.dataframe(
        cross_df[["model", "subset", "recall", "f1"]].rename(
            columns={"model": "Model", "subset": "Subset", "recall": "Recall", "f1": "F1"}
        ),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(metric_card("Best Detection F1", "0.565", "CNN+Transformer, 4s/0.5s"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Best Cross-Species F1", "0.935", "CNN+Transformer, balanced"), unsafe_allow_html=True)

with tab2:
    st.markdown(f"{paper_badge('Paper 2')}", unsafe_allow_html=True)

    metric_sel = st.selectbox("Metric", ["accuracy", "f1", "auroc", "auprc"], key="tab2_met")

    # Clean
    st.markdown("**Clean Datasets**")
    clean_df = load_results("cosupformer_clean.csv")
    fig2c = go.Figure()
    for i, model in enumerate(clean_df["model"].unique()):
        sub = clean_df[clean_df["model"] == model]
        fig2c.add_trace(go.Bar(x=sub["dataset"], y=sub[metric_sel], name=model, marker_color=MODEL_COLORS[i]))
    fig2c.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
        title=f"Clean — {metric_sel.upper()}", barmode="group",
        yaxis=dict(title=metric_sel.upper(), gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=60, r=30, t=50, b=80),
    )
    st.plotly_chart(fig2c, use_container_width=True)

    # Noisy
    st.markdown("**Noisy / Challenging Datasets**")
    noisy_df = load_results("cosupformer_noisy.csv")
    fig2n = go.Figure()
    for i, model in enumerate(noisy_df["model"].unique()):
        sub = noisy_df[noisy_df["model"] == model]
        fig2n.add_trace(go.Bar(x=sub["dataset"], y=sub[metric_sel], name=model, marker_color=MODEL_COLORS[i]))
    fig2n.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
        title=f"Noisy — {metric_sel.upper()}", barmode="group",
        yaxis=dict(title=metric_sel.upper(), gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=400, margin=dict(l=60, r=30, t=50, b=80),
    )
    st.plotly_chart(fig2n, use_container_width=True)

    st.markdown(metric_card("MACO Accuracy", "74.9%", "↑ vs 38.8% CrossFormer"), unsafe_allow_html=True)

with tab3:
    st.markdown(f"{paper_badge('Paper 3')}", unsafe_allow_html=True)

    spec_df = load_results("specmoe_main.csv")
    # Exclude SEED-VIG (regression task with RMSE)
    cls_df = spec_df[spec_df["dataset"] != "SEED-VIG"]

    fig3 = go.Figure()
    for i, model in enumerate(cls_df["model"].unique()):
        sub = cls_df[cls_df["model"] == model]
        fig3.add_trace(go.Bar(x=sub["dataset"], y=sub["balanced_accuracy"], name=model, marker_color=MODEL_COLORS[i]))
    fig3.update_layout(
        paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
        title="Balanced Accuracy Across 8 Classification Benchmarks", barmode="group",
        yaxis=dict(title="Balanced Accuracy", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        height=500, margin=dict(l=60, r=30, t=50, b=100),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Count wins
    wins = {}
    for ds in cls_df["dataset"].unique():
        ds_sub = cls_df[cls_df["dataset"] == ds]
        best = ds_sub.loc[ds_sub["balanced_accuracy"].idxmax(), "model"]
        wins[best] = wins.get(best, 0) + 1
    # Include SEED-VIG (SpecMoE best with lowest RMSE)
    wins["SpecMoE"] = wins.get("SpecMoE", 0) + 1

    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown(metric_card("SpecMoE Wins", f"{wins.get('SpecMoE', 0)}/9", "benchmarks"), unsafe_allow_html=True)
    with w2:
        st.markdown(metric_card("Best MACO Score", "0.853", "↑ +8.8% vs CBraMod"), unsafe_allow_html=True)
    with w3:
        st.markdown(metric_card("SEED-VIG RMSE", "0.152", "↓ Nearly halves error"), unsafe_allow_html=True)

# =====================================================================
# Section 2 — The Research Narrative
# =====================================================================
st.markdown(section_header("2. The Research Narrative", "How each paper builds on the previous"), unsafe_allow_html=True)

events = [
    {"year": 2025, "label": "CoSupFormer", "color": ACCENT_CYAN,
     "description": "Task-specific but robust: dilated CNN + global attention + contrastive loss"},
    {"year": 2026.0, "label": "Thomas et al.", "color": ACCENT_AMBER,
     "description": "Real-world detection pipeline, cross-species validation (F1=0.935)"},
    {"year": 2026.25, "label": "SpecMoE", "color": ACCENT_PURPLE,
     "description": "Universal foundation model: 9K hours pre-training, spectral MoE, 7/9 SOTA"},
]
fig_tl = plot_timeline(events, title="Research Timeline")
st.plotly_chart(fig_tl, use_container_width=True)

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.8;">
    <strong style="color:{ACCENT_CYAN};">2025 — CoSupFormer</strong> showed that combining multi-scale encoding,
    global attention, gating, and contrastive learning yields robust EEG classifiers even on noisy data.<br>
    <strong style="color:{ACCENT_AMBER};">2026 — Thomas et al.</strong> revealed the critical gap between
    classification and detection, and proved that mouse-trained models can generalize to humans.<br>
    <strong style="color:{ACCENT_PURPLE};">2026 — SpecMoE</strong> scaled up to a foundation model,
    leveraging thousands of hours of data and a spectral MoE framework to achieve state-of-the-art
    on 7 of 9 diverse benchmarks.
    </div>""",
    unsafe_allow_html=True,
)

# =====================================================================
# Section 3 — What This Means for Pharma
# =====================================================================
st.markdown(section_header("3. What This Means for Pharma"), unsafe_allow_html=True)

apps = [
    ("🔬", "Automated Seizure Quantification", "Replace manual annotation in preclinical studies, saving weeks of expert time per study."),
    ("💊", "Drug Effect Classification", "Classify therapeutic effects from EEG across multiple CNS drug classes (antidepressants, antipsychotics, anxiolytics)."),
    ("🐭→🧑", "Cross-Species Translation", "Models trained on mouse EEG translate to human — de-risking preclinical-to-clinical transitions."),
    ("📡", "Noise-Robust Analysis", "Gating and contrastive learning reduce data loss from noisy recordings, especially in preclinical settings."),
]
for icon, title, desc in apps:
    st.markdown(
        f"""<div style="display:flex;align-items:flex-start;gap:16px;margin:12px 0;padding:12px 16px;
        background:{BG_CARD};border-radius:8px;border:1px solid #1e293b;">
        <span style="font-size:1.5rem;">{icon}</span>
        <div><strong style="color:{TEXT_PRIMARY};">{title}</strong><br>
        <span style="color:{TEXT_SECONDARY};font-size:0.9rem;">{desc}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

# =====================================================================
# Section 4 — Model Efficiency Comparison
# =====================================================================
st.markdown(section_header("4. Model Efficiency Comparison", "Parameter count vs. performance"), unsafe_allow_html=True)

spec_df = load_results("specmoe_main.csv")
cls_df = spec_df[spec_df["dataset"] != "SEED-VIG"]
avg_perf = cls_df.groupby("model")["balanced_accuracy"].mean().reset_index()
avg_perf.columns = ["model", "avg_balanced_accuracy"]

# Add approximate parameter counts
param_map = {
    "EEGNet": 0.1, "EEGConformer": 0.5, "FFCL": 0.8,
    "LaBraM": 10.0, "CBraMod": 8.0, "CSBrain": 15.0, "SpecMoE": 4.3,
}
avg_perf["params_M"] = avg_perf["model"].map(param_map)

fig_eff = go.Figure()
for i, row in avg_perf.iterrows():
    fig_eff.add_trace(
        go.Scatter(
            x=[row["params_M"]], y=[row["avg_balanced_accuracy"]],
            mode="markers+text",
            marker=dict(size=18, color=MODEL_COLORS[i % len(MODEL_COLORS)]),
            text=[row["model"]], textposition="top center",
            textfont=dict(size=10, color=TEXT_PRIMARY),
            name=row["model"],
            hovertemplate=f"<b>{row['model']}</b><br>Params: {row['params_M']}M<br>Avg Bal. Acc: {row['avg_balanced_accuracy']:.3f}<extra></extra>",
        )
    )
fig_eff.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD, font=dict(color=TEXT_PRIMARY),
    title="Parameter Efficiency: Size vs. Average Balanced Accuracy",
    xaxis=dict(title="Parameters (Millions, log scale)", type="log", gridcolor="#1e293b"),
    yaxis=dict(title="Avg Balanced Accuracy (8 tasks)", gridcolor="#1e293b"),
    showlegend=False, height=450,
    margin=dict(l=60, r=30, t=50, b=60),
)
st.plotly_chart(fig_eff, use_container_width=True)

st.markdown(
    callout_box(
        "SpecMoE achieves the best average performance with only <strong>4.3M parameters</strong>, "
        "while models like CSBrain and LaBraM require 10-15M+ parameters. "
        "CoSupFormer is even more compact (0.4-2.4M) for task-specific use cases.",
        "⚡", ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 5 — Limitations & Future Directions
# =====================================================================
st.markdown(section_header("5. Limitations & Future Directions"), unsafe_allow_html=True)

lim_data = [
    ("Paper 1", ACCENT_AMBER, [
        "Limited to MTLE mouse model — other epilepsy types may behave differently",
        "Small human test set (Bonn, 5 subjects per class)",
        "Detection pipeline requires dense overlapping windows, increasing computation",
    ]),
    ("Paper 2", ACCENT_CYAN, [
        "Focused exclusively on EEG — applicability to ECG/EMG untested",
        "Combined loss can occasionally hurt in very low-data regimes",
        "Training time is higher than simpler baselines due to contrastive component",
    ]),
    ("Paper 3", ACCENT_PURPLE, [
        "Requires full-parameter fine-tuning — no zero-shot capability yet",
        "Fixed 50% masking ratio may be suboptimal for all tasks",
        "Expert specialization is stochastic — domain-specific partitioning could help",
        "Three experts is a practical trade-off, not necessarily optimal",
    ]),
]

for paper, color, lims in lim_data:
    with st.expander(f"{paper} — Limitations"):
        for lim in lims:
            st.markdown(f"- {lim}")

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;margin-top:16px;">
    <strong style="color:{TEXT_PRIMARY};">Open Questions:</strong><br>
    • Can foundation models achieve zero-shot EEG decoding?<br>
    • How should experts be specialized — by domain, frequency content, or task?<br>
    • Can cross-species transfer work beyond rodent → human (e.g., NHP → human)?<br>
    • What masking ratio and geometry are optimal for different EEG paradigms?
    </div>""",
    unsafe_allow_html=True,
)

# =====================================================================
# Section 6 — Glossary
# =====================================================================
st.markdown(section_header("6. Glossary"), unsafe_allow_html=True)

glossary = {
    "EEG (Electroencephalography)": "Recording of electrical brain activity via electrodes on the scalp or in the brain.",
    "STFT (Short-Time Fourier Transform)": "Decomposes a signal into its frequency content over time, producing a spectrogram.",
    "PSD (Power Spectral Density)": "Measures the distribution of signal power across frequencies. Computed via Welch's method.",
    "Attention Mechanism": "Neural network component that learns which parts of the input are most relevant for the output.",
    "Contrastive Learning": "Training paradigm that pulls same-class representations together and pushes different-class representations apart.",
    "Foundation Model": "A model pre-trained on large-scale data that can be fine-tuned for many downstream tasks.",
    "MoE (Mixture of Experts)": "Architecture using multiple specialized sub-networks (experts) with a gating mechanism to route inputs.",
    "AUROC": "Area Under the Receiver Operating Characteristic curve — measures discriminative ability across all thresholds.",
    "AUPRC": "Area Under the Precision-Recall Curve — especially useful for imbalanced datasets.",
    "F1-Score": "Harmonic mean of precision and recall — balances false positives and false negatives.",
    "Dilated Convolution": "Convolution with gaps between kernel elements, expanding the receptive field without increasing parameters.",
    "Cross-Entropy Loss": "Standard classification loss measuring divergence between predicted and true probability distributions.",
    "NT-Xent (Normalized Temperature-scaled Cross-Entropy)": "Contrastive loss function that encourages similar samples to have similar representations.",
    "Balanced Accuracy": "Average of per-class recall — corrects for class imbalance by weighting each class equally.",
}

for term, definition in glossary.items():
    with st.expander(term):
        st.markdown(f"*{definition}*")

# =====================================================================
# Section 7 — References & Links
# =====================================================================
st.markdown(section_header("7. References & Links"), unsafe_allow_html=True)

st.markdown(
    f"""
**Paper 1** — D. Darankoum, M. Villalba, C. Allioux, B. Caraballo, C. Dumont, E. Gronlier, C. Roucard, Y. Roche, C. Habermacher, S. Grudinin, J. Volle.
*From epilepsy seizure classification to detection: a deep learning-based approach for raw EEG signals.*
Neuroscience Informatics, 2026.

**Paper 2** — D. Darankoum, J. Volle, C. Habermacher, S. Grudinin.
*CoSupFormer: A Contrastive Supervised Learning Approach for EEG Signal Classification.*
arXiv:2509.20489, 2025.

**Paper 3** — D. Darankoum, C. Habermacher, J. Volle, S. Grudinin.
*SpecMoE: Spectral Mixture-of-Experts Foundation Model for Cross-Species EEG Decoding.*
arXiv:2603.16739, 2026.

---

**Affiliations:** Univ. Grenoble Alpes, CNRS, Grenoble INP, LJK · SynapCell SAS
""",
    unsafe_allow_html=True,
)

render_bottom_nav(
    prev_page=("pages/7_Foundation_Models.py", "Foundation Models"),
    next_page=None,
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
