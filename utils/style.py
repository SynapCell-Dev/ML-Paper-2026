"""Shared CSS injection, color constants, and card helpers for EEG DeepDive."""

import streamlit as st

# ---------------------------------------------------------------------------
# Color palette – scientific / neuroscience dark theme
# ---------------------------------------------------------------------------
BG_PRIMARY = "#0a0e17"
BG_CARD = "#111827"
ACCENT_CYAN = "#06b6d4"
ACCENT_PURPLE = "#8b5cf6"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#94a3b8"

PAPER_COLORS = {
    "Paper 1": ACCENT_AMBER,
    "Paper 2": ACCENT_CYAN,
    "Paper 3": ACCENT_PURPLE,
}


def inject_css():
    """Inject the global CSS theme into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def paper_badge(paper: str) -> str:
    """Return an HTML badge for a paper label."""
    color = PAPER_COLORS.get(paper, ACCENT_CYAN)
    return (
        f'<span style="background:{color}22;color:{color};'
        f'padding:4px 12px;border-radius:20px;font-size:0.8rem;'
        f'font-weight:600;border:1px solid {color}44;">{paper}</span>'
    )


def metric_card(label: str, value: str, delta: str = "") -> str:
    """Return HTML for a compact metric display card."""
    delta_html = ""
    if delta:
        color = ACCENT_GREEN if delta.startswith("+") or delta.startswith("↑") else ACCENT_RED
        delta_html = f'<div style="color:{color};font-size:0.85rem;margin-top:4px;">{delta}</div>'
    return (
        f'<div class="metric-card">'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.8rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.8rem;font-weight:700;">{value}</div>'
        f'{delta_html}</div>'
    )


def callout_box(text: str, icon: str = "💡", color: str = ACCENT_CYAN) -> str:
    """Return HTML for a highlighted insight callout box."""
    return (
        f'<div style="background:{color}11;border-left:4px solid {color};'
        f'padding:16px 20px;border-radius:0 8px 8px 0;margin:16px 0;">'
        f'<span style="font-size:1.2rem;margin-right:8px;">{icon}</span>'
        f'<span style="color:{TEXT_PRIMARY};font-size:0.95rem;">{text}</span></div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Return HTML for a styled section header."""
    sub = ""
    if subtitle:
        sub = f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;margin-top:4px;">{subtitle}</div>'
    return (
        f'<div style="margin:32px 0 16px 0;">'
        f'<h2 style="color:{TEXT_PRIMARY};margin:0;font-size:1.6rem;">{title}</h2>'
        f'{sub}</div>'
    )


def neuro_card(content_html: str, border_color: str = "") -> str:
    """Wrap *content_html* in a ``.neuro-card`` div with optional border-top color."""
    border_style = f"border-top:3px solid {border_color};" if border_color else ""
    return f'<div class="neuro-card" style="{border_style}">{content_html}</div>'


def audience_card(icon: str, title: str, description: str, color: str) -> str:
    """Return HTML for an audience card with the given border-top *color*."""
    return (
        f'<div class="audience-card" style="border-top:3px solid {color};">'
        f'<div style="font-size:2rem;margin-bottom:8px;">{icon}</div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.1rem;font-weight:700;'
        f'margin-bottom:8px;">{title}</div>'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.5;">'
        f'{description}</div></div>'
    )


def paper_card(badge_html: str, title: str, authors: str, venue: str, color: str) -> str:
    """Return HTML for a paper card on the home page."""
    return (
        f'<div class="paper-card" style="border-top:3px solid {color};">'
        f'<div style="margin-bottom:12px;">{badge_html}</div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;'
        f'margin-bottom:6px;">{title}</div>'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.85rem;line-height:1.6;">'
        f'{authors}<br><em>{venue}</em></div></div>'
    )


def info_paragraph(text: str) -> str:
    """Return HTML for a styled paragraph (centered, max-width 800px)."""
    return (
        f'<div style="max-width:800px;margin:24px auto 32px auto;color:{TEXT_SECONDARY};'
        f'font-size:1rem;line-height:1.7;text-align:center;">{text}</div>'
    )


def colored_header(text: str, color: str) -> str:
    """Return HTML for a colored sub-heading ``<h4>``."""
    return f'<h4 style="color:{color};margin-top:0;">{text}</h4>'


def paper_identity_banner(paper: str, title: str, subtitle: str = "") -> str:
    """Return a wide, colored banner identifying the paper a page belongs to.

    Shown near the top of each paper-scoped module so the active paper stays
    visible regardless of scroll position. Accepts "Paper 1", "Paper 2",
    "Paper 3", or a comma-separated list like "Paper 1, Paper 2, Paper 3" for
    pages that cover multiple papers.
    """
    papers = [p.strip() for p in paper.split(",")]
    colors = [PAPER_COLORS.get(p, ACCENT_CYAN) for p in papers]
    # Use the first paper's color as the dominant border/gradient.
    primary = colors[0]
    badges = " ".join(paper_badge(p) for p in papers)
    gradient_stops = ", ".join(f"{c}22" for c in colors)
    sub = ""
    if subtitle:
        sub = (
            f'<div style="color:{TEXT_SECONDARY};font-size:0.9rem;'
            f'margin-top:4px;">{subtitle}</div>'
        )
    return (
        f'<div style="display:flex;align-items:center;gap:16px;'
        f'background:linear-gradient(90deg, {gradient_stops}, transparent);'
        f'border-left:5px solid {primary};border-radius:0 10px 10px 0;'
        f'padding:14px 20px;margin:4px 0 20px 0;">'
        f'<div style="flex-shrink:0;">{badges}</div>'
        f'<div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;">'
        f"{title}</div>"
        f"{sub}"
        f"</div></div>"
    )


def paper_section_header(title: str, paper: str, subtitle: str = "") -> str:
    """Section header whose left border and title color match the paper color."""
    color = PAPER_COLORS.get(paper, ACCENT_CYAN)
    sub = ""
    if subtitle:
        sub = (
            f'<div style="color:{TEXT_SECONDARY};font-size:0.92rem;margin-top:4px;">'
            f"{subtitle}</div>"
        )
    return (
        f'<div style="margin:32px 0 16px 0;padding-left:14px;'
        f'border-left:4px solid {color};">'
        f'<h2 style="color:{color};margin:0;font-size:1.5rem;">{title}</h2>'
        f"{sub}</div>"
    )


def render_paper_legend_sidebar():
    """Render a compact paper color legend in the Streamlit sidebar.

    Call this once per page (after ``inject_css``). It is a no-op if the
    sidebar isn't available.
    """
    import streamlit as st

    items_html = ""
    for paper, color in PAPER_COLORS.items():
        items_html += (
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'margin:6px 0;font-size:0.82rem;color:{TEXT_SECONDARY};">'
            f'<div style="width:14px;height:14px;border-radius:3px;'
            f'background:{color};border:1px solid {color}aa;"></div>'
            f"<span>{paper}</span></div>"
        )
    st.sidebar.markdown(
        f'<div style="border-top:1px solid #1e293b;margin-top:16px;padding-top:12px;">'
        f'<div style="color:{TEXT_PRIMARY};font-size:0.85rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
        "Paper legend</div>"
        f"{items_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_bottom_nav(prev_page: tuple | None = None, next_page: tuple | None = None):
    """Render a Previous / Next module navigation row at the bottom of a page.

    Each argument is either ``None`` or a tuple ``(path, label)``.
    """
    import streamlit as st

    st.markdown(
        f'<hr style="border-color:#1e293b;margin-top:32px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    col_prev, col_spacer, col_next = st.columns([1, 2, 1])
    with col_prev:
        if prev_page is not None:
            path, label = prev_page
            st.page_link(path, label=f"← {label}")
    with col_next:
        if next_page is not None:
            path, label = next_page
            st.markdown(
                '<div style="text-align:right;">',
                unsafe_allow_html=True,
            )
            st.page_link(path, label=f"{label} →")
            st.markdown("</div>", unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list[str]], col_colors: list[str] | None = None) -> str:
    """Return a styled HTML table matching the dark theme."""
    th_cells = "".join(
        f'<th style="color:{TEXT_SECONDARY};padding:8px 12px;border-bottom:1px solid #1e293b;'
        f'text-align:left;">{h}</th>'
        for h in headers
    )
    body = ""
    for row in rows:
        cells = ""
        for i, cell in enumerate(row):
            color = col_colors[i] if col_colors and i < len(col_colors) else TEXT_PRIMARY
            cells += (
                f'<td style="color:{color};padding:8px 12px;'
                f'border-bottom:1px solid #1e293b;">{cell}</td>'
            )
        body += f"<tr>{cells}</tr>"
    return (
        f'<table style="width:100%;border-collapse:collapse;background:{BG_CARD};'
        f'border-radius:8px;overflow:hidden;">'
        f"<thead><tr>{th_cells}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def app_item(icon: str, title: str, description: str) -> str:
    """Return HTML for an app item card (icon + title + description in a flex row)."""
    return (
        f'<div style="display:flex;align-items:flex-start;gap:16px;'
        f'background:{BG_CARD};border:1px solid #1e293b;border-radius:12px;'
        f'padding:20px;margin:8px 0;">'
        f'<div style="font-size:1.8rem;">{icon}</div>'
        f'<div>'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;'
        f'margin-bottom:4px;">{title}</div>'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.9rem;line-height:1.5;">'
        f'{description}</div></div></div>'
    )


def glow_title(text: str) -> str:
    """Return the ``<h1 class="glow-header">`` pattern."""
    return f'<h1 class="glow-header">{text}</h1>'


FOOTER_HTML = f"""
<hr style="border-color:{BG_CARD};margin-top:48px;">
<div style="text-align:center;padding:24px 0;color:{TEXT_SECONDARY};font-size:0.8rem;">
    <strong>EEG DeepDive</strong> — An educational companion to three research papers<br>
    Authors: David Darankoum, Romain Thomas &amp; collaborators<br>
    Affiliations: Univ. Grenoble Alpes &middot; SynapCell<br>
    <em>This app is for educational purposes only. No clinical decisions should be based on its content.</em>
</div>
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = f"""<style>
/* ---- Global ---- */
.stApp {{
    background-color: {BG_PRIMARY};
}}
section[data-testid="stSidebar"] {{
    background-color: {BG_CARD};
}}

/* ---- Glow header ---- */
.glow-header {{
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, {ACCENT_CYAN}, {ACCENT_PURPLE}, {ACCENT_GREEN});
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
    text-align: center;
    margin-bottom: 0;
}}
@keyframes gradient-shift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ---- Neuro card ---- */
.neuro-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.neuro-card:hover {{
    transform: translateY(-2px);
    border-color: {ACCENT_CYAN}55;
}}

/* ---- Metric card ---- */
.metric-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}}

/* ---- Callout box ---- */
.callout-box {{
    background: {ACCENT_CYAN}11;
    border-left: 4px solid {ACCENT_CYAN};
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 16px 0;
}}

/* ---- Custom scrollbar ---- */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}
::-webkit-scrollbar-track {{
    background: {BG_PRIMARY};
}}
::-webkit-scrollbar-thumb {{
    background: #334155;
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #475569;
}}

/* ---- Links ---- */
a {{
    color: {ACCENT_CYAN} !important;
    text-decoration: none !important;
}}
a:hover {{
    text-decoration: underline !important;
}}

/* ---- Audience card ---- */
.audience-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    min-height: 180px;
    transition: transform 0.2s ease, border-color 0.2s ease;
    cursor: default;
}}
.audience-card:hover {{
    transform: translateY(-3px);
    border-color: {ACCENT_CYAN}66;
}}

/* ---- Paper card ---- */
.paper-card {{
    background: {BG_CARD};
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 24px;
    min-height: 200px;
    transition: border-color 0.2s ease;
}}

/* ---- Tab styling ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {BG_CARD};
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    color: {TEXT_SECONDARY};
}}
.stTabs [aria-selected="true"] {{
    background-color: {ACCENT_CYAN}22;
    color: {ACCENT_CYAN};
}}
</style>"""
