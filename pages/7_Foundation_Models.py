"""Page 7 — Foundation Models for EEG (Paper 3 — SpecMoE)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_embeddings, load_results
from utils.eeg_synth import (
    apply_gaussian_mask,
    apply_rectangular_mask,
    compute_stft,
    generate_eeg,
)
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
    plot_embedding_scatter,
    plot_spectrogram,
)

inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(7)

render_paper_legend_sidebar()

# ---- Header ----
st.markdown(
    paper_identity_banner(
        "Paper 3",
        "SpecMoE — Spectral Mixture-of-Experts foundation model",
    ),
    unsafe_allow_html=True,
)
st.title("🏗️ Foundation Models for EEG")
st.markdown(
    callout_box(
        "From task-specific models to a universal EEG decoder — SpecMoE leverages 9,000 hours of "
        "pre-training data and a novel spectral Mixture of Experts to achieve state-of-the-art across "
        "7 of 9 benchmarks, including cross-species tasks.",
        "🧬", ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1 — What is a Foundation Model?
# =====================================================================
st.markdown(section_header("1. What is a Foundation Model?"), unsafe_allow_html=True)

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:800px;">
    A <strong>foundation model</strong> is pre-trained on a massive, diverse dataset using
    self-supervised learning, then fine-tuned for specific downstream tasks.
    Think GPT for language or CLIP for vision — <strong>SpecMoE</strong> does this for brain signals.
    </div>""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(metric_card("Pre-training Data", "~9,000 hrs", "TUEG corpus"), unsafe_allow_html=True)
with m2:
    st.markdown(metric_card("Subjects", "14,987", "Clinical EEG"), unsafe_allow_html=True)
with m3:
    st.markdown(metric_card("Samples", "1.1M", "30-sec segments"), unsafe_allow_html=True)
with m4:
    st.markdown(metric_card("Parameters", "4.3M", "Compact model"), unsafe_allow_html=True)

# =====================================================================
# Section 2 — The Masking Problem
# =====================================================================
st.markdown(section_header("2. The Masking Problem", "Why Gaussian-smoothed masks outperform rectangular masks"), unsafe_allow_html=True)

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:800px;margin-bottom:16px;">
    During pre-training, parts of the EEG spectrogram are <strong>masked</strong> and the model
    learns to reconstruct them. The shape of the mask matters enormously — rectangular masks
    create artificial boundaries that the model can exploit, while Gaussian masks force it to learn
    genuine neural patterns.
    </div>""",
    unsafe_allow_html=True,
)

# Generate synthetic EEG and STFT
eeg_sig = generate_eeg(duration_sec=2.0, fs=256, seed=123)
freqs, times, Zxx = compute_stft(eeg_sig, fs=256, nperseg=64, noverlap=48)

# Controls
ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
    mask_type = st.selectbox("Mask Type", ["Joint Time-Frequency", "Frequency Only", "Time Only"], key="mask_type")
with ctrl2:
    sigma_f = st.slider("σ_f (frequency spread)", 2.0, 20.0, 5.0, 1.0, key="sig_f")
with ctrl3:
    sigma_t = st.slider("σ_t (time spread)", 2.0, 20.0, 5.0, 1.0, key="sig_t")

# Adjust sigmas based on mask type
if mask_type == "Frequency Only":
    eff_sigma_t = 100.0  # very large = mask spans all time
    eff_sigma_f = sigma_f
elif mask_type == "Time Only":
    eff_sigma_f = 100.0  # very large = mask spans all frequencies
    eff_sigma_t = sigma_t
else:
    eff_sigma_f = sigma_f
    eff_sigma_t = sigma_t

# Apply masks
masked_gauss, mask_g = apply_gaussian_mask(Zxx, sigma_f=eff_sigma_f, sigma_t=eff_sigma_t)
masked_rect, mask_r = apply_rectangular_mask(Zxx, width_f=int(sigma_f * 2), width_t=int(sigma_t * 2))

col_orig, col_rect, col_gauss = st.columns(3)

with col_orig:
    fig_o = plot_spectrogram(Zxx, times, freqs, title="Original Spectrogram", max_freq=80)
    st.plotly_chart(fig_o, use_container_width=True)

with col_rect:
    fig_r = plot_spectrogram(mask_r, times, freqs, title="Rectangular Mask", colorscale="RdBu", max_freq=80)
    st.plotly_chart(fig_r, use_container_width=True)

with col_gauss:
    fig_g = plot_spectrogram(mask_g, times, freqs, title="Gaussian Mask", colorscale="RdBu", max_freq=80)
    st.plotly_chart(fig_g, use_container_width=True)

st.markdown(
    callout_box(
        "<strong>Rectangular masks</strong> create sharp edges that the model can exploit as shortcuts. "
        "<strong>Gaussian masks</strong> have smooth transitions that force the model to learn real "
        "spectral-temporal relationships. Additionally, 50% of masks are biased toward physiological "
        "EEG bands (δ, θ, α, β), ensuring the model focuses on biologically meaningful frequencies.",
        "💡", ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# Section 3 — SpecHi-Net Architecture
# =====================================================================
st.markdown(section_header("3. SpecHi-Net Architecture", "A U-shaped hierarchical encoder-decoder"), unsafe_allow_html=True)

st.markdown(create_architecture_diagram("spechinet"), unsafe_allow_html=True)

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:800px;margin-top:16px;">
    SpecHi-Net uses a <strong>U-shaped</strong> architecture with:<br>
    • <strong>3 downsampling stages</strong>: dual-path convolutions at each level extract multi-scale features<br>
    • <strong>Global transformer layers</strong>: capture long-range dependencies across the compressed representation<br>
    • <strong>3 upsampling stages</strong>: skip connections preserve fine-grained details from earlier layers<br>
    • <strong>Multi-level reconstruction losses</strong>: ensure accurate signal recovery at every scale
    </div>""",
    unsafe_allow_html=True,
)

# =====================================================================
# Section 4 — Spectral Mixture of Experts
# =====================================================================
st.markdown(section_header("4. Spectral Mixture of Experts", "Dynamic expert routing based on signal spectral content"), unsafe_allow_html=True)

st.markdown(create_architecture_diagram("spectral_moe"), unsafe_allow_html=True)

st.markdown(
    f"""<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:800px;margin:16px 0;">
    <strong>Three experts</strong> are pretrained on different TUEG subsets (~300k-400k samples each).
    During fine-tuning, a <strong>PSD-based gating mechanism</strong> computes the signal's spectral
    fingerprint and routes information to the most relevant expert. Different tasks activate different
    expert combinations.
    </div>""",
    unsafe_allow_html=True,
)

# Show embedding visualization
tab_maco, tab_da = st.tabs(["🐭 MACO Drug Classification", "💊 DA-Pharmaco"])
with tab_maco:
    emb_maco = load_embeddings("tsne_specmoe_maco.json")
    fig_m = plot_embedding_scatter(emb_maco, color_col="label", title="t-SNE — MACO (5 Drug Classes)")
    st.plotly_chart(fig_m, use_container_width=True)
with tab_da:
    emb_da = load_embeddings("tsne_specmoe_dapharmaco.json")
    fig_d = plot_embedding_scatter(emb_da, color_col="label", title="t-SNE — DA-Pharmaco (5 Dopaminergic Compounds)")
    st.plotly_chart(fig_d, use_container_width=True)

# =====================================================================
# Section 5 — Ablation Results
# =====================================================================
st.markdown(section_header("5. Ablation Results", "Quantifying the contribution of each component"), unsafe_allow_html=True)

abl_df = load_results("specmoe_ablation.csv")
dataset_choice = st.selectbox("Dataset", ["BCIC2020_3", "DA_Pharmaco", "PhysioNet_MI"], key="spec_abl_ds")

fig_abl = go.Figure()
colors = []
for _, row in abl_df.iterrows():
    if row["variant"] == "SpecMoE (Full)":
        colors.append(ACCENT_CYAN)
    elif "Non-Gaussian" in row["variant"]:
        colors.append(ACCENT_RED)
    elif "CBraMod" in row["variant"]:
        colors.append(ACCENT_RED)
    else:
        colors.append(ACCENT_PURPLE)

fig_abl.add_trace(
    go.Bar(
        x=abl_df["variant"],
        y=abl_df[dataset_choice],
        marker_color=colors,
        text=[f"{v:.4f}" for v in abl_df[dataset_choice]],
        textposition="outside",
    )
)

# Add reference line for full model
full_val = abl_df[abl_df["variant"] == "SpecMoE (Full)"][dataset_choice].values[0]
fig_abl.add_hline(y=full_val, line_dash="dash", line_color=ACCENT_CYAN, opacity=0.5)

fig_abl.update_layout(
    paper_bgcolor=BG_PRIMARY, plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY),
    title=f"Ablation — {dataset_choice.replace('_', '-')}",
    yaxis_title="Balanced Accuracy",
    yaxis=dict(range=[0, max(abl_df[dataset_choice]) * 1.15], gridcolor="#1e293b"),
    xaxis=dict(gridcolor="#1e293b", tickangle=-30),
    height=450, margin=dict(l=60, r=30, t=50, b=120),
)
st.plotly_chart(fig_abl, use_container_width=True)

if dataset_choice == "BCIC2020_3":
    st.markdown(
        callout_box(
            "On BCIC2020-3 (imagined speech), replacing SpecHi-Net with CBraMod causes a "
            "<strong>catastrophic collapse</strong> from 0.626 to 0.245 — confirming that the hierarchical "
            "architecture is essential for complex spectral tasks.",
            "💥", ACCENT_RED,
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        callout_box(
            "Non-Gaussian masking and single-dimension masking consistently underperform, "
            "confirming that the joint spectral-temporal Gaussian corruption strategy is critical.",
            "📊", ACCENT_AMBER,
        ),
        unsafe_allow_html=True,
    )

render_bottom_nav(
    prev_page=("pages/6_Learning_Representations.py", "Learning Representations"),
    next_page=("pages/8_Results_and_Impact.py", "Results & Impact"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
