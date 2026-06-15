"""Page 9 — Datasets Reference: all datasets used across the three papers."""

import streamlit as st

from utils.style import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_GREEN,
    ACCENT_PURPLE,
    ACCENT_RED,
    BG_CARD,
    BG_PRIMARY,
    FOOTER_HTML,
    PAPER_COLORS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    callout_box,
    html_table,
    inject_css,
    metric_card,
    paper_badge,
    paper_identity_banner,
    render_paper_legend_sidebar,
    section_header,
)

inject_css()
render_paper_legend_sidebar()

st.markdown(
    paper_identity_banner(
        "Paper 1, Paper 2, Paper 3",
        "Complete dataset reference — who used what, when, and why",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY};margin-bottom:0;">Datasets Reference</h1>'
    f'<p style="color:{TEXT_SECONDARY};font-size:1.05rem;margin-top:4px;">'
    "Every dataset appearing across the three papers, grouped by which paper "
    "used it and what role it played.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "Each dataset below is tagged by species (🧑 human, 🐭 mouse) and by "
        "its role: <em>pretraining</em>, <em>training</em>, <em>testing</em>, "
        "or <em>benchmarking</em>. Click through for paper-specific details.",
        icon="📚",
        color=ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# At-a-glance matrix
# ===========================================================================
st.markdown(section_header("At a Glance"), unsafe_allow_html=True)

matrix_rows = [
    ["MACO (mouse pharmacoEEG)", "🐭", "—", "train/test", "fine-tune/eval"],
    ["TDBrain", "🧑", "—", "train/test", "fine-tune/eval"],
    ["ADFTD", "🧑", "—", "train/test", "fine-tune/eval"],
    ["Bonn (seizure)", "🧑", "cross-species test", "—", "—"],
    ["MTLE mouse (internal)", "🐭", "train/test", "—", "—"],
    ["TUEG / TUH EEG", "🧑", "—", "—", "pretraining (9k h)"],
    ["DA-Pharmaco", "🐭", "—", "—", "fine-tune/eval"],
    ["BCIC2020-3", "🧑", "—", "—", "fine-tune/eval"],
    ["PhysioNet MI", "🧑", "—", "—", "fine-tune/eval"],
    ["SEED-VIG", "🧑", "—", "—", "fine-tune/eval"],
]

st.markdown(
    html_table(
        headers=["Dataset", "Species", "Paper 1", "Paper 2", "Paper 3"],
        rows=matrix_rows,
        col_colors=[
            TEXT_PRIMARY,
            TEXT_PRIMARY,
            ACCENT_AMBER,
            ACCENT_CYAN,
            ACCENT_PURPLE,
        ],
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.82rem;font-style:italic;'
    f'margin-top:8px;">A dash (—) means that paper did not use the dataset.</div>',
    unsafe_allow_html=True,
)

# ===========================================================================
# Paper 1 datasets
# ===========================================================================
st.markdown(section_header("Paper 1 — Thomas et al. (Detection Challenge)"),
            unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.92rem;line-height:1.7;max-width:900px;">'
    "Paper 1 tackles seizure <strong>detection</strong> in continuous recordings, "
    "not just classification of pre-segmented windows. Its datasets are chosen to "
    "stress both within-species performance and <em>cross-species generalization</em> "
    "from mouse to human."
    "</div>",
    unsafe_allow_html=True,
)

col1a, col1b = st.columns(2, gap="medium")

with col1a:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_AMBER};">
<div style="margin-bottom:8px;">{paper_badge("Paper 1")}</div>
<h4 style="color:{ACCENT_AMBER};margin:4px 0 6px 0;">MTLE mouse model (internal)</h4>
<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
<strong>Species:</strong> 🐭 Mouse &nbsp;·&nbsp; <strong>Role:</strong> train + test<br>
Chronic recordings from a mesial temporal lobe epilepsy (MTLE) model.
Multiple mice with long recording sessions containing spontaneous seizures
annotated by experts.
<br><br>
<strong>Why it's here:</strong> realistic preclinical scenario — continuous
long-duration recordings with sparse seizure events, exactly the setting
where detection metrics matter and classification metrics mislead.
</div></div>""",
        unsafe_allow_html=True,
    )

with col1b:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_AMBER};">
<div style="margin-bottom:8px;">{paper_badge("Paper 1")}</div>
<h4 style="color:{ACCENT_AMBER};margin:4px 0 6px 0;">Bonn Seizure Database</h4>
<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
<strong>Species:</strong> 🧑 Human &nbsp;·&nbsp; <strong>Role:</strong> cross-species test<br>
Classic benchmark from the University of Bonn — 5 subsets of 100 single-channel
EEG segments each (healthy, interictal, ictal conditions).
<br><br>
<strong>Why it's here:</strong> models are trained on mouse data and
evaluated <em>zero-shot</em> on human Bonn recordings. A true cross-species
generalization test that would be impossible with a single-species-only setup.
The CNN+Transformer reaches F1 ≈ 0.935 on this transfer task.
</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "Paper 1's <strong>affiliation</strong>: Univ. Grenoble Alpes · CNRS · "
        "Grenoble INP · LJK · SynapCell SAS. Published in Neuroscience Informatics, 2026.",
        icon="🏛️",
        color=ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# Paper 2 datasets
# ===========================================================================
st.markdown(section_header("Paper 2 — CoSupFormer (Darankoum et al.)"),
            unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.92rem;line-height:1.7;max-width:900px;">'
    "CoSupFormer is evaluated on a mix of <em>clean</em> clinical datasets "
    "and <em>noisy</em> or challenging datasets to show that its gating "
    "mechanism and contrastive loss pay off specifically in realistic "
    "(non-curated) settings."
    "</div>",
    unsafe_allow_html=True,
)

p2_data = [
    {
        "name": "MACO",
        "species": "🐭 Mouse",
        "role": "train / test (noisy)",
        "task": "Drug effect classification — 4 compound classes + vehicle (solvent) control",
        "why": (
            "MACO is the paper's flagship preclinical dataset: freely-behaving mice receiving "
            "different pharmacological compounds, recorded over long sessions. Because it's "
            "real recording, some electrodes are noisy or drift — making the gating unit "
            "essential. CoSupFormer jumps from 38.8% (baseline) to 74.9% accuracy here."
        ),
    },
    {
        "name": "TDBrain",
        "species": "🧑 Human",
        "role": "train / test (clean)",
        "task": "Resting-state EEG across psychiatric conditions",
        "why": (
            "A large clinical resting-state dataset. Used as the clean-regime benchmark "
            "to show CoSupFormer competes with state-of-the-art even when gating isn't "
            "strictly necessary."
        ),
    },
    {
        "name": "ADFTD",
        "species": "🧑 Human",
        "role": "train / test (clean)",
        "task": "Alzheimer's disease vs. frontotemporal dementia vs. healthy",
        "why": (
            "A three-class dementia classification benchmark. Challenging because the "
            "class-discriminative information is subtle and distributed across frequency "
            "bands — which motivates the multi-scale dilated encoder."
        ),
    },
]

for d in p2_data:
    st.markdown(
        f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_CYAN};">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
  {paper_badge("Paper 2")}
  <h4 style="color:{ACCENT_CYAN};margin:0;">{d['name']}</h4>
  <span style="color:{TEXT_SECONDARY};font-size:0.85rem;">{d['species']} · {d['role']}</span>
</div>
<div style="color:{TEXT_PRIMARY};font-size:0.9rem;margin-bottom:6px;">
<strong>Task:</strong> {d['task']}</div>
<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
<strong>Why it's here:</strong> {d['why']}
</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "Paper 2's <strong>affiliation</strong>: Univ. Grenoble Alpes · CNRS · "
        "Grenoble INP · LJK · SynapCell SAS. arXiv, 2025.",
        icon="🏛️",
        color=ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# Paper 3 datasets
# ===========================================================================
st.markdown(section_header("Paper 3 — SpecMoE (Darankoum et al.)"),
            unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.92rem;line-height:1.7;max-width:900px;">'
    "SpecMoE is a foundation model, so its dataset strategy splits cleanly in two: "
    "<strong>one huge corpus for self-supervised pretraining</strong>, and "
    "<strong>many smaller downstream datasets for fine-tuning / benchmarking</strong>. "
    "The goal is to show that pretraining transfers broadly."
    "</div>",
    unsafe_allow_html=True,
)

# Pretraining corpus — prominent card
st.markdown(
    f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_PURPLE};
background:linear-gradient(90deg, {ACCENT_PURPLE}11, transparent);">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
  {paper_badge("Paper 3")}
  <h4 style="color:{ACCENT_PURPLE};margin:0;">TUEG — Temple University EEG Corpus</h4>
  <span style="color:{TEXT_SECONDARY};font-size:0.85rem;">🧑 Human · pretraining only</span>
</div>
<div style="color:{TEXT_PRIMARY};font-size:0.9rem;margin-bottom:6px;">
<strong>Scale:</strong> ~27,000 hours · 14,987 subjects · 1.1M samples after windowing
</div>
<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
<strong>Why it's here:</strong> one of the largest publicly available EEG corpora.
SpecMoE uses it <em>without any labels</em> for masked STFT reconstruction pretraining
(see Module 7). The resulting representations transfer to every downstream task below.
The pretraining data is <strong>human-only</strong>; evaluating on mouse datasets
later tests cross-species transfer.
</div></div>""",
    unsafe_allow_html=True,
)

# Downstream benchmarks — grid
st.markdown(
    f'<div style="color:{TEXT_PRIMARY};font-weight:600;margin:20px 0 8px 0;'
    f'font-size:1.05rem;">Downstream benchmarks (fine-tuning / evaluation)</div>',
    unsafe_allow_html=True,
)

p3_benchmarks = [
    ("MACO", "🐭", "Mouse pharmacoEEG drug classification (5 classes). "
     "Cross-species transfer from a human-only pretrained model."),
    ("DA-Pharmaco", "🐭", "Dopaminergic compound discrimination in freely-moving mice "
     "(5 drugs). Tests fine-grained pharmacological decoding."),
    ("TDBrain", "🧑", "Clinical resting-state dataset; re-used from Paper 2 for "
     "direct comparison with CoSupFormer."),
    ("ADFTD", "🧑", "Dementia classification; same motivation as TDBrain for cross-paper comparison."),
    ("BCIC2020-3", "🧑", "BCI competition 2020, task 3 — motor imagery. "
     "Broad-community benchmark."),
    ("PhysioNet MI", "🧑", "PhysioNet motor-imagery dataset. Another BCI-focused "
     "evaluation emphasizing robust cross-subject generalization."),
    ("SEED-VIG", "🧑", "Vigilance estimation (continuous regression). The "
     "0.1522 score in the paper is an RMSE (lower is better), not balanced accuracy."),
    ("Bonn", "🧑", "Seizure classification — tiny test set, used here as an "
     "additional cross-task probe."),
]

for i in range(0, len(p3_benchmarks), 2):
    cols = st.columns(2, gap="medium")
    for j, c in enumerate(cols):
        if i + j >= len(p3_benchmarks):
            break
        name, species, desc = p3_benchmarks[i + j]
        with c:
            st.markdown(
                f"""<div class="neuro-card" style="border-top:3px solid {ACCENT_PURPLE};min-height:140px;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
  <h4 style="color:{ACCENT_PURPLE};margin:0;">{name}</h4>
  <span style="color:{TEXT_SECONDARY};font-size:0.82rem;">{species}</span>
</div>
<div style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.55;">
{desc}
</div></div>""",
                unsafe_allow_html=True,
            )

st.markdown(
    callout_box(
        "Paper 3's <strong>affiliation</strong>: Univ. Grenoble Alpes · CNRS · "
        "Grenoble INP · LJK · SynapCell SAS. arXiv, 2026.",
        icon="🏛️",
        color=ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# Notes on provenance
# ===========================================================================
st.markdown(section_header("A Few Notes on Provenance"), unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.7;max-width:900px;">'
    "<ul style='margin:0;padding-left:20px;'>"
    "<li><strong>MACO and DA-Pharmaco</strong> are proprietary SynapCell preclinical "
    "datasets collected in freely-behaving rodents.</li>"
    "<li><strong>TDBrain and ADFTD</strong> are publicly released clinical datasets "
    "commonly used as benchmarks in the EEG deep-learning literature.</li>"
    "<li><strong>TUEG</strong> is the Temple University Hospital EEG Corpus — one "
    "of the largest publicly available EEG resources, released by TUH.</li>"
    "<li><strong>Bonn, BCIC2020-3, PhysioNet MI, SEED-VIG</strong> are public "
    "benchmarks widely used across BCI and clinical EEG papers, which is why they "
    "are chosen here: apples-to-apples comparisons with prior work.</li>"
    "<li>All experiments in the three papers were approved under the ethical "
    "protocols cited in each manuscript. This app only visualizes <em>synthetic</em> "
    "data — no real patient or animal data is shown.</li>"
    "</ul>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
