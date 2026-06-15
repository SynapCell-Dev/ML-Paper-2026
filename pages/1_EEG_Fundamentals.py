"""Page 1 -- EEG Fundamentals: What is EEG and why it matters for pharma."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.eeg_synth import (
    BAND_COLORS,
    BANDS,
    compute_psd,
    generate_band_signals,
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
from utils.viz import plot_eeg_signal, plot_psd

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(1)

render_paper_legend_sidebar()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    paper_identity_banner(
        "Paper 1, Paper 2, Paper 3",
        "Shared foundation — terminology and datasets used across all three papers",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY};margin-bottom:0;">EEG Fundamentals</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:1.05rem;margin-top:4px;">'
    "Everything you need to know about brain signals before diving into the models.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "Why this matters: EEG is one of the most promising non-invasive biomarkers "
        "for drug development. Understanding the signal is the first step toward "
        "understanding how deep learning can decode it.",
        "🧠",
        ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 1 -- What is EEG?
# =========================================================================
st.markdown(section_header("What is EEG?", "Electroencephalography in 60 seconds"), unsafe_allow_html=True)

col_text, col_signal = st.columns([1, 1])

with col_text:
    st.markdown(
        f"""
<div class="neuro-card">
<p style="color:{TEXT_PRIMARY};line-height:1.7;font-size:0.95rem;">
<strong>Electroencephalography (EEG)</strong> measures the electrical activity
of the brain through electrodes placed on the scalp. Each electrode picks up
the summed post-synaptic potentials of millions of cortical neurons.
</p>
<p style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.95rem;">
The resulting signal is:</p>
<ul style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.8;">
    <li><strong style="color:{ACCENT_CYAN};">Fast</strong> &mdash; millisecond temporal resolution</li>
    <li><strong style="color:{ACCENT_GREEN};">Non-invasive</strong> &mdash; no surgery, no radiation</li>
    <li><strong style="color:{ACCENT_PURPLE};">Cheap</strong> &mdash; portable devices cost under $500</li>
    <li><strong style="color:{ACCENT_AMBER};">Noisy</strong> &mdash; eye blinks, muscle artifacts, electrode drift</li>
</ul>
<p style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.95rem;">
The signal is typically sampled at 256 Hz and amplitudes are in the range
of 10&ndash;100 &mu;V &mdash; about 1000x weaker than an ECG.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col_signal:
    st.markdown(
        f'<p style="color:{TEXT_SECONDARY};font-size:0.85rem;">Adjust the sliders to see how each frequency band contributes to the composite signal.</p>',
        unsafe_allow_html=True,
    )
    amp_delta = st.slider("Delta (1-4 Hz)", 0.0, 3.0, 1.0, 0.1, key="amp_delta")
    amp_theta = st.slider("Theta (4-8 Hz)", 0.0, 3.0, 0.7, 0.1, key="amp_theta")
    amp_alpha = st.slider("Alpha (8-12 Hz)", 0.0, 3.0, 1.0, 0.1, key="amp_alpha")
    amp_beta = st.slider("Beta (12-30 Hz)", 0.0, 3.0, 0.4, 0.1, key="amp_beta")
    amp_gamma = st.slider("Gamma (30-100 Hz)", 0.0, 3.0, 0.15, 0.05, key="amp_gamma")

    band_amps = {
        "delta": amp_delta,
        "theta": amp_theta,
        "alpha": amp_alpha,
        "beta": amp_beta,
        "gamma": amp_gamma,
    }
    eeg_signal = generate_eeg(duration_sec=5.0, fs=256, band_amplitudes=band_amps, noise_level=0.2, seed=42)
    fig_eeg = plot_eeg_signal(eeg_signal, fs=256, title="Interactive Synthetic EEG")
    st.plotly_chart(fig_eeg, use_container_width=True)

# =========================================================================
# Section 2 -- Frequency Bands and Brain States
# =========================================================================
st.markdown(
    section_header(
        "Frequency Bands and Brain States",
        "The five canonical rhythms of the brain",
    ),
    unsafe_allow_html=True,
)

band_info = {
    "delta": {
        "range": "1 - 4 Hz",
        "state": "Deep sleep, unconsciousness",
        "clinical": "Elevated in sedation, brain injury; key marker for anaesthesia depth",
    },
    "theta": {
        "range": "4 - 8 Hz",
        "state": "Drowsiness, light sleep, meditation",
        "clinical": "Increased in ADHD; changes with anxiolytics and antiepileptics",
    },
    "alpha": {
        "range": "8 - 12 Hz",
        "state": "Relaxed wakefulness, eyes closed",
        "clinical": "Reduced in Alzheimer's; sensitive to benzodiazepines",
    },
    "beta": {
        "range": "12 - 30 Hz",
        "state": "Active thinking, concentration",
        "clinical": "Altered by stimulants and antidepressants; marker of cortical activation",
    },
    "gamma": {
        "range": "30 - 100 Hz",
        "state": "High-level cognition, perception binding",
        "clinical": "Linked to schizophrenia; emerging biomarker for cognitive enhancers",
    },
}

# Visual table
band_table_html = f"""
<div style="overflow-x:auto;margin:8px 0;">
<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
<tr style="border-bottom:2px solid #1e293b;">
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Band</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Frequency</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Brain State</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Clinical Relevance</th>
</tr>
"""
for band, info in band_info.items():
    color = BAND_COLORS[band]
    band_table_html += f"""
<tr style="border-bottom:1px solid #1e293b;">
    <td style="padding:10px;color:{color};font-weight:700;">{band.capitalize()}</td>
    <td style="padding:10px;color:{TEXT_PRIMARY};">{info["range"]}</td>
    <td style="padding:10px;color:{TEXT_PRIMARY};">{info["state"]}</td>
    <td style="padding:10px;color:{TEXT_SECONDARY};">{info["clinical"]}</td>
</tr>
"""
band_table_html += "</table></div>"

st.markdown(band_table_html, unsafe_allow_html=True)

# PSD plot
st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.9rem;margin-top:16px;">'
    "Power spectral density of the synthetic signal above, with colored frequency bands.</p>",
    unsafe_allow_html=True,
)
freqs_psd, psd_vals = compute_psd(eeg_signal, fs=256)
fig_psd = plot_psd(freqs_psd, psd_vals, title="Power Spectral Density with Frequency Bands")
st.plotly_chart(fig_psd, use_container_width=True)

# =========================================================================
# Section 3 -- Why EEG for Drug Development?
# =========================================================================
st.markdown(
    section_header(
        "Why EEG for Drug Development?",
        "From bedside monitor to digital biomarker",
    ),
    unsafe_allow_html=True,
)

col_pharma1, col_pharma2 = st.columns(2)

with col_pharma1:
    st.markdown(
        f"""
<div class="neuro-card">
<h3 style="color:{ACCENT_CYAN};font-size:1.1rem;margin-top:0;">EEG as a Biomarker</h3>
<p style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.92rem;">
Unlike imaging modalities such as fMRI or PET, EEG is:
</p>
<ul style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.8;">
    <li><strong style="color:{TEXT_PRIMARY};">Continuous</strong> &mdash; can record 24/7 for days</li>
    <li><strong style="color:{TEXT_PRIMARY};">Sensitive</strong> &mdash; detects sub-second changes in brain state</li>
    <li><strong style="color:{TEXT_PRIMARY};">Scalable</strong> &mdash; hundreds of subjects in multi-site trials</li>
    <li><strong style="color:{TEXT_PRIMARY};">Translatable</strong> &mdash; same technique works in mice and humans</li>
</ul>
<p style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.92rem;">
This makes EEG ideal for <strong style="color:{TEXT_PRIMARY};">longitudinal monitoring</strong>
in clinical trials, where you need to track drug effects over weeks or months.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col_pharma2:
    st.markdown(
        f"""
<div class="neuro-card">
<h3 style="color:{ACCENT_PURPLE};font-size:1.1rem;margin-top:0;">PharmacoEEG</h3>
<p style="color:{TEXT_SECONDARY};line-height:1.7;font-size:0.92rem;">
<strong style="color:{TEXT_PRIMARY};">PharmacoEEG</strong> is the study of how drugs alter brain
electrical activity. It provides:</p>
<ul style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.8;">
    <li><strong style="color:{TEXT_PRIMARY};">Dose-response curves</strong> from spectral power changes</li>
    <li><strong style="color:{TEXT_PRIMARY};">Drug classification</strong> &mdash; different drug classes have
        distinct EEG signatures (sedatives increase delta, stimulants increase beta)</li>
    <li><strong style="color:{TEXT_PRIMARY};">Target engagement proof</strong> &mdash; confirming the drug
        reaches and modulates the intended brain circuits</li>
    <li><strong style="color:{TEXT_PRIMARY};">Safety monitoring</strong> &mdash; early detection of pro-convulsant
        risk before clinical seizures occur</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "The three papers in this app all address the same core challenge: "
        "how to build reliable, automated EEG analysis that could replace "
        "the hours of manual review currently required in clinical trials.",
        "💊",
        ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# =========================================================================
# Section 4 -- Datasets Used
# =========================================================================
st.markdown(
    section_header(
        "Datasets Used Across the Three Papers",
        "From 88 subjects to nearly 15,000",
    ),
    unsafe_allow_html=True,
)

datasets = [
    {"name": "TDBrain", "subjects": "1,274", "channels": "33", "species": "Human", "focus": "Parkinson's subset", "paper": "Paper 2"},
    {"name": "ADFTD", "subjects": "88", "channels": "19", "species": "Human", "focus": "AD vs FTD vs HC", "paper": "Paper 2"},
    {"name": "MACO", "subjects": "128–336", "channels": "2–3", "species": "Mouse", "focus": "Drug classification", "paper": "Paper 2"},
    {"name": "Bonn", "subjects": "5×100", "channels": "1", "species": "Human", "focus": "Epilepsy", "paper": "Paper 1"},
    {"name": "TUEG", "subjects": "14,987", "channels": "~23", "species": "Human", "focus": "Pre-training", "paper": "Paper 3"},
]

# Metric cards row
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
for col, ds in zip([mcol1, mcol2, mcol3, mcol4, mcol5], datasets):
    with col:
        st.markdown(
            metric_card(ds["name"], ds["subjects"], f"{ds['channels']} ch"),
            unsafe_allow_html=True,
        )

# Detailed table
st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

dataset_table_html = f"""
<div style="overflow-x:auto;margin:8px 0;">
<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
<tr style="border-bottom:2px solid #1e293b;">
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Dataset</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Subjects</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Channels</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Species</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Clinical Focus</th>
    <th style="padding:10px;text-align:left;color:{TEXT_SECONDARY};">Used In</th>
</tr>
"""
species_colors = {"Human": ACCENT_CYAN, "Mouse": ACCENT_GREEN}
for ds in datasets:
    sp_color = species_colors.get(ds["species"], TEXT_PRIMARY)
    dataset_table_html += f"""
<tr style="border-bottom:1px solid #1e293b;">
    <td style="padding:10px;color:{TEXT_PRIMARY};font-weight:700;">{ds["name"]}</td>
    <td style="padding:10px;color:{TEXT_PRIMARY};">{ds["subjects"]}</td>
    <td style="padding:10px;color:{TEXT_PRIMARY};">{ds["channels"]}</td>
    <td style="padding:10px;color:{sp_color};font-weight:600;">{ds["species"]}</td>
    <td style="padding:10px;color:{TEXT_SECONDARY};">{ds["focus"]}</td>
    <td style="padding:10px;">{paper_badge(ds["paper"])}</td>
</tr>
"""
dataset_table_html += "</table></div>"

st.markdown(dataset_table_html, unsafe_allow_html=True)

st.markdown(
    f'<p style="color:{TEXT_SECONDARY};font-size:0.85rem;font-style:italic;margin-top:8px;">'
    "TUEG (Temple University EEG Corpus) contains ~27,000 hours of recordings and "
    "was used exclusively for self-supervised pretraining in Paper 3 (SpecMoE).</p>",
    unsafe_allow_html=True,
)

# =========================================================================
# Section 5 -- Mouse vs. Human EEG
# =========================================================================
st.markdown(
    section_header(
        "Mouse vs. Human EEG",
        "Same technique, different frequency profiles",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<p style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;">
Mouse brains are smaller and faster. Their EEG signals tend to have higher dominant
frequencies and less prominent low-frequency (delta/theta) activity compared to
humans. This is critical for <strong style="color:{TEXT_PRIMARY};">cross-species
transfer learning</strong> &mdash; a model trained on human data cannot simply be
applied to mouse data without accounting for these spectral differences.
</p>
""",
    unsafe_allow_html=True,
)

col_human, col_mouse = st.columns(2)

# Human EEG: strong alpha, moderate delta
human_amps = {"delta": 1.0, "theta": 0.7, "alpha": 1.2, "beta": 0.4, "gamma": 0.1}
human_signal = generate_eeg(duration_sec=5.0, fs=256, band_amplitudes=human_amps, noise_level=0.2, seed=100)

# Mouse EEG: weaker delta/alpha, stronger theta/beta/gamma
mouse_amps = {"delta": 0.3, "theta": 1.2, "alpha": 0.4, "beta": 1.0, "gamma": 0.6}
mouse_signal = generate_eeg(duration_sec=5.0, fs=256, band_amplitudes=mouse_amps, noise_level=0.3, seed=200)

with col_human:
    st.markdown(
        f'<div style="text-align:center;color:{ACCENT_CYAN};font-weight:700;font-size:1rem;margin-bottom:4px;">'
        "Human EEG</div>",
        unsafe_allow_html=True,
    )
    fig_human = plot_eeg_signal(human_signal, fs=256, title="Human: Strong alpha, moderate delta", color=ACCENT_CYAN)
    st.plotly_chart(fig_human, use_container_width=True)

    freqs_h, psd_h = compute_psd(human_signal, fs=256)
    fig_psd_h = plot_psd(freqs_h, psd_h, title="Human PSD")
    st.plotly_chart(fig_psd_h, use_container_width=True)

with col_mouse:
    st.markdown(
        f'<div style="text-align:center;color:{ACCENT_GREEN};font-weight:700;font-size:1rem;margin-bottom:4px;">'
        "Mouse EEG</div>",
        unsafe_allow_html=True,
    )
    fig_mouse = plot_eeg_signal(mouse_signal, fs=256, title="Mouse: Stronger theta/beta/gamma", color=ACCENT_GREEN)
    st.plotly_chart(fig_mouse, use_container_width=True)

    freqs_m, psd_m = compute_psd(mouse_signal, fs=256)
    fig_psd_m = plot_psd(freqs_m, psd_m, title="Mouse PSD")
    st.plotly_chart(fig_psd_m, use_container_width=True)

st.markdown(
    callout_box(
        "Paper 3 (SpecMoE) tackles this cross-species challenge directly by using "
        "a Mixture of Experts architecture where different experts can specialize "
        "in different spectral profiles.",
        "🐭",
        ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render_bottom_nav(
    prev_page=None,
    next_page=("pages/2_DeepLearning_Primer.py", "Deep Learning Primer"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
