"""Page 6 -- Learning Representations: from contrastive loss to foundation-model pretraining."""

import streamlit as st

from utils.data_loader import load_embeddings
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
from utils.viz import plot_embedding_scatter

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(6)

render_paper_legend_sidebar()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
st.markdown(
    paper_identity_banner(
        "Paper 2, Paper 3",
        "Bridge — from CoSupFormer's contrastive loss to SpecMoE's foundation model",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="text-align:center;color:{TEXT_PRIMARY};margin-top:8px;">'
    "Learning Representations</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="text-align:center;color:{TEXT_SECONDARY};font-size:1.05rem;margin-top:-8px;">'
    "How contrastive learning reshapes the embedding space -- and why it matters for EEG</p>",
    unsafe_allow_html=True,
)

# ===================================================================
# Section 1 -- The Generalization Problem
# ===================================================================
st.markdown(
    section_header(
        "The Generalization Problem in EEG",
        "Why standard cross-entropy alone is not enough",
    ),
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        f"""
<div class="neuro-card" style="min-height:170px;">
    <div style="font-size:1.5rem;margin-bottom:8px;">🧠</div>
    <div style="color:{ACCENT_CYAN};font-weight:700;margin-bottom:6px;">Inter-subject Variability</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
        Every brain is unique. Cortical folding, skull thickness, and
        neural dynamics differ across individuals, shifting signal
        distributions dramatically between subjects.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
<div class="neuro-card" style="min-height:170px;">
    <div style="font-size:1.5rem;margin-bottom:8px;">⏱</div>
    <div style="color:{ACCENT_PURPLE};font-weight:700;margin-bottom:6px;">Temporal Fluctuations</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
        Alertness, fatigue, and circadian rhythms cause the same
        person's EEG to look different minute-to-minute. Models
        that memorize one session fail on the next.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        f"""
<div class="neuro-card" style="min-height:170px;">
    <div style="font-size:1.5rem;margin-bottom:8px;">📡</div>
    <div style="color:{ACCENT_GREEN};font-weight:700;margin-bottom:6px;">Electrode Inconsistencies</div>
    <div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">
        Montage differences, impedance drift, and placement errors
        mean that two recordings with the same hardware can produce
        very different raw signals.
    </div>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "Cross-entropy loss only cares about the decision boundary. It does not "
        "explicitly structure the latent space, so features of different subjects "
        "can scatter unpredictably -- hurting generalization.",
        "⚠️",
        ACCENT_AMBER,
    ),
    unsafe_allow_html=True,
)

# ===================================================================
# Section 2 -- Contrastive Learning Intuition
# ===================================================================
st.markdown(
    section_header(
        "Contrastive Learning -- Intuitive Explanation",
        "Pull same-class samples together, push different-class samples apart",
    ),
    unsafe_allow_html=True,
)

# Visual diagram using HTML/CSS
st.markdown(
    f"""
<div style="background:{BG_CARD};border:1px solid #1e293b;border-radius:12px;
            padding:28px 24px;margin:16px 0;">
    <div style="display:flex;justify-content:center;gap:60px;flex-wrap:wrap;">
        <!-- Before contrastive -->
        <div style="text-align:center;">
            <div style="color:{TEXT_SECONDARY};font-size:0.9rem;font-weight:600;
                        margin-bottom:14px;">Before (CE only)</div>
            <div style="position:relative;width:200px;height:200px;
                        background:{BG_PRIMARY};border-radius:12px;margin:0 auto;">
                <!-- Scattered dots -->
                <div style="position:absolute;top:25px;left:30px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:65px;left:140px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:150px;left:80px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:40px;left:90px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:120px;left:30px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:100px;left:155px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:80px;left:60px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
                <div style="position:absolute;top:170px;left:150px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
                <div style="position:absolute;top:30px;left:160px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
            </div>
            <div style="color:{TEXT_SECONDARY};font-size:0.78rem;margin-top:8px;">
                Classes overlap &amp; scatter</div>
        </div>
        <!-- Arrow -->
        <div style="display:flex;align-items:center;color:{ACCENT_AMBER};
                    font-size:2.5rem;font-weight:700;">&#10230;</div>
        <!-- After contrastive -->
        <div style="text-align:center;">
            <div style="color:{TEXT_SECONDARY};font-size:0.9rem;font-weight:600;
                        margin-bottom:14px;">After (CE + Contrastive)</div>
            <div style="position:relative;width:200px;height:200px;
                        background:{BG_PRIMARY};border-radius:12px;margin:0 auto;">
                <!-- Clustered dots -->
                <div style="position:absolute;top:30px;left:25px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:45px;left:45px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:25px;left:50px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_CYAN};"></div>
                <div style="position:absolute;top:120px;left:25px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:135px;left:45px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:125px;left:55px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_PURPLE};"></div>
                <div style="position:absolute;top:80px;left:130px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
                <div style="position:absolute;top:95px;left:150px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
                <div style="position:absolute;top:75px;left:155px;width:14px;height:14px;
                            border-radius:50%;background:{ACCENT_GREEN};"></div>
            </div>
            <div style="color:{TEXT_SECONDARY};font-size:0.78rem;margin-top:8px;">
                Tight, well-separated clusters</div>
        </div>
    </div>
    <!-- Legend -->
    <div style="display:flex;justify-content:center;gap:24px;margin-top:18px;">
        <span style="color:{ACCENT_CYAN};font-size:0.82rem;font-weight:600;">
            ● Solvent</span>
        <span style="color:{ACCENT_PURPLE};font-size:0.82rem;font-weight:600;">
            ● Antidepressant</span>
        <span style="color:{ACCENT_GREEN};font-size:0.82rem;font-weight:600;">
            ● Antipsychotic</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# NT-Xent explanation
st.markdown(
    f"""
<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:780px;
            margin:16px auto;">
<strong style="color:{TEXT_PRIMARY};">NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy):</strong><br>
For a given anchor sample, the contrastive loss treats every same-class sample in the
batch as a <em>positive</em> and every other sample as a <em>negative</em>. The loss
minimizes the distance between positives and maximizes the distance to negatives in a
temperature-scaled softmax:
</div>
""",
    unsafe_allow_html=True,
)

st.latex(
    r"\mathcal{L}_{\text{NT-Xent}} = "
    r"-\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}"
    r"{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}"
)

st.markdown(
    f"""
<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:780px;
            margin:8px auto;">
<strong style="color:{TEXT_PRIMARY};">The &lambda; balancing parameter:</strong><br>
CoSupFormer combines cross-entropy and contrastive objectives via a weighted sum:
</div>
""",
    unsafe_allow_html=True,
)

st.latex(
    r"\mathcal{L}_{\text{total}} = "
    r"\mathcal{L}_{\text{CE}} + \lambda \, \mathcal{L}_{\text{contrastive}}"
)

st.markdown(
    callout_box(
        "When &lambda; = 0 we recover standard cross-entropy (SupFormer). "
        "As &lambda; increases, the embedding space is increasingly structured by "
        "class relationships -- improving cross-subject generalization.",
        "🎯",
        ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# ===================================================================
# Section 3 -- Interactive Embedding Explorer
# ===================================================================
st.markdown(
    section_header(
        "Interactive Embedding Explorer",
        "PCA projections of learned representations -- with and without contrastive loss",
    ),
    unsafe_allow_html=True,
)

CLASS_COLORS = {
    "Solvent": ACCENT_CYAN,
    "Antidepressant": ACCENT_PURPLE,
    "Antipsychotic": ACCENT_GREEN,
}

emb_without = load_embeddings("pca_without_contrastive.json")
emb_with = load_embeddings("pca_with_contrastive.json")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        f'<div style="text-align:center;color:{TEXT_PRIMARY};font-weight:600;'
        f'margin-bottom:-8px;">SupFormer (CE only)</div>',
        unsafe_allow_html=True,
    )
    fig_without = plot_embedding_scatter(
        emb_without,
        color_col="label",
        title="Without Contrastive Loss",
        color_map=CLASS_COLORS,
    )
    st.plotly_chart(fig_without, use_container_width=True)

with col_right:
    st.markdown(
        f'<div style="text-align:center;color:{TEXT_PRIMARY};font-weight:600;'
        f'margin-bottom:-8px;">CoSupFormer (CE + Contrastive)</div>',
        unsafe_allow_html=True,
    )
    fig_with = plot_embedding_scatter(
        emb_with,
        color_col="label",
        title="With Contrastive Loss",
        color_map=CLASS_COLORS,
    )
    st.plotly_chart(fig_with, use_container_width=True)

st.markdown(
    callout_box(
        "The contrastive loss pulls same-class embeddings into compact clusters and "
        "pushes different classes apart. This cleaner geometry makes downstream "
        "classification more robust across subjects and sessions.",
        "✨",
        ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# ===================================================================
# Section 4 -- Bridge to Foundation Models
# ===================================================================
st.markdown(
    section_header(
        "Bridge to Foundation Models",
        "From better losses to large-scale pretraining",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="neuro-card" style="max-width:800px;margin:0 auto;">
    <div style="color:{TEXT_PRIMARY};font-size:1rem;line-height:1.8;">
        CoSupFormer demonstrated a key insight: <strong style="color:{ACCENT_CYAN};">
        better loss design improves generalization</strong>. By structuring the
        embedding space through contrastive learning, we obtained more robust
        representations -- even with limited labeled data.<br><br>
        But a natural question follows: <em style="color:{ACCENT_PURPLE};">What if
        we could pre-train on thousands of hours of EEG data?</em><br><br>
        Rather than relying solely on clever loss functions applied to small
        datasets, <strong style="color:{ACCENT_PURPLE};">foundation models</strong>
        learn universal EEG representations from massive unlabeled corpora. The
        model learns the "language of the brain" first, then adapts to specific
        tasks with minimal fine-tuning.
    </div>
    <div style="display:flex;justify-content:center;gap:16px;margin-top:20px;
                flex-wrap:wrap;">
        <div style="background:{ACCENT_CYAN}15;border:1px solid {ACCENT_CYAN}44;
                    border-radius:8px;padding:12px 20px;text-align:center;">
            <div style="color:{ACCENT_CYAN};font-weight:700;font-size:0.95rem;">
                Paper 2: CoSupFormer</div>
            <div style="color:{TEXT_SECONDARY};font-size:0.82rem;">Better loss
                &rarr; better embeddings</div>
        </div>
        <div style="display:flex;align-items:center;color:{ACCENT_AMBER};
                    font-size:1.8rem;font-weight:700;">&#10230;</div>
        <div style="background:{ACCENT_PURPLE}15;border:1px solid {ACCENT_PURPLE}44;
                    border-radius:8px;padding:12px 20px;text-align:center;">
            <div style="color:{ACCENT_PURPLE};font-weight:700;font-size:0.95rem;">
                Paper 3: SpecMoE</div>
            <div style="color:{TEXT_SECONDARY};font-size:0.82rem;">Massive pre-training
                &rarr; universal representations</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "Next up: we explore how SpecMoE uses 9,000+ hours of unlabeled EEG, "
        "Gaussian masking in the time-frequency domain, and a Mixture of Experts "
        "architecture to build a true foundation model for brain signals.",
        "🚀",
        ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render_bottom_nav(
    prev_page=("pages/5_Attention_and_Gating.py", "Attention & Gating"),
    next_page=("pages/7_Foundation_Models.py", "Foundation Models"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
