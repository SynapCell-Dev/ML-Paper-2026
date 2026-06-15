"""Home -- Landing page for EEG DeepDive."""

import streamlit as st

from utils.style import (
    ACCENT_AMBER,
    ACCENT_CYAN,
    ACCENT_PURPLE,
    FOOTER_HTML,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    audience_card,
    glow_title,
    info_paragraph,
    inject_css,
    paper_badge,
    paper_card,
)

inject_css()

# ---------------------------------------------------------------------------
# Title & subtitle
# ---------------------------------------------------------------------------
st.markdown(glow_title("EEG DeepDive"), unsafe_allow_html=True)
st.markdown(
    f'<p style="text-align:center;color:{TEXT_SECONDARY};font-size:1.15rem;margin-top:-8px;">'
    "A Visual Tour Through Three Papers on EEG Deep Learning</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Narrative paragraph
# ---------------------------------------------------------------------------
st.markdown(
    info_paragraph(
        "EEG signals are rich, noisy, and deeply personal.  Over the course of three "
        "research papers we tackled a single question from multiple angles: "
        "<em>How can deep learning reliably decode brain activity "
        "</em>  This interactive app walks you through the ideas, the "
        "architectures, and the results&mdash;whether you work in pharma, study "
        "neuroscience, or build machine-learning models."
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Audience selector
# ---------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:center;margin-bottom:8px;">'
    f'<span style="color:{TEXT_PRIMARY};font-size:1.2rem;font-weight:600;">'
    "Choose your path</span></div>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        audience_card(
            "🏢", "I'm in Pharma",
            'Start with <strong>EEG Fundamentals</strong>, then '
            '<strong>Detection Challenge</strong>, and finish at '
            '<strong>Results &amp; Impact</strong>.',
            ACCENT_AMBER,
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_EEG_Fundamentals.py", label="Start here →", icon="🧠")

with col2:
    st.markdown(
        audience_card(
            "🎓", "I'm a Student",
            "Follow all seven modules in order for a comprehensive, "
            "step-by-step learning experience.",
            ACCENT_CYAN,
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_EEG_Fundamentals.py", label="Begin Module 1 →", icon="📚")

with col3:
    st.markdown(
        audience_card(
            "🔬", "I'm a Researcher",
            "Jump directly to any technical module&mdash;each one is "
            "self-contained with architecture diagrams and ablations.",
            ACCENT_PURPLE,
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_MultiScale_Encoding.py", label="Jump to architectures →", icon="🔬")

# ---------------------------------------------------------------------------
# Paper cards
# ---------------------------------------------------------------------------
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
st.markdown(
    f'<div style="text-align:center;margin-bottom:8px;">'
    f'<span style="color:{TEXT_PRIMARY};font-size:1.2rem;font-weight:600;">'
    "The Papers</span></div>",
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        paper_card(paper_badge("Paper 1"), "The Detection Challenge",
                   "Darankoum et al.", "Neuroscience Informatics, 2026", ACCENT_AMBER),
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        paper_card(paper_badge("Paper 2"), "CoSupFormer",
                   "Darankoum et al.", "arXiv, 2025", ACCENT_CYAN),
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        paper_card(paper_badge("Paper 3"), "SpecMoE",
                   "Darankoum et al.", "arXiv, 2026", ACCENT_PURPLE),
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
