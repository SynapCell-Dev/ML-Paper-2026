"""Page 2 — Deep Learning Primer: Convolution, CNN, and Transformer explainers."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.eeg_synth import generate_eeg
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
    neuro_card,
    paper_badge,
    paper_identity_banner,
    render_bottom_nav,
    render_paper_legend_sidebar,
    section_header,
)

inject_css()

if "visited_modules" not in st.session_state:
    st.session_state.visited_modules = set()
st.session_state.visited_modules.add(2)

render_paper_legend_sidebar()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    paper_identity_banner(
        "Paper 1, Paper 2, Paper 3",
        "Deep learning building blocks used across all three papers",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY};margin-top:8px;">Deep Learning Primer</h1>'
    f'<p style="color:{TEXT_SECONDARY};font-size:1.1rem;">'
    "Before diving into the three papers, let's build intuition for the three core "
    "building blocks they all rely on: <strong>convolution</strong>, "
    "<strong>CNNs</strong>, and <strong>self-attention / transformers</strong>.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    callout_box(
        "All three papers mix convolutional layers (to capture local time patterns in EEG) "
        "with attention layers (to capture long-range and cross-channel relationships). "
        "If these ideas feel fuzzy, this module is for you.",
        icon="🧭",
        color=ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 1 — Convolution (1D) step-by-step
# ===========================================================================
st.markdown(section_header("1. What is a Convolution?"), unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:900px;">'
    "A 1D convolution slides a small <strong>kernel</strong> (a tiny window of weights) "
    "along a signal. At each position it multiplies the kernel values by the signal "
    "values underneath, sums them up, and writes the result into an output signal called "
    "a <strong>feature map</strong>. The same kernel is reused everywhere — that's why "
    "convolutions are <em>translation-equivariant</em> and have very few parameters."
    "</div>",
    unsafe_allow_html=True,
)

# Build a synthetic EEG signal and a few kernels
@st.cache_data
def _primer_signal():
    sig = generate_eeg(duration_sec=2.0, fs=64, seed=7, noise_level=0.15)
    return sig

full_signal = _primer_signal()
# Downsample for visualisation clarity
signal_short = full_signal[:64]

KERNELS = {
    "Smoothing (average)": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    "Edge detector (derivative)": np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    "Peak detector (Mexican hat)": np.array([-0.3, 0.2, 1.0, 0.2, -0.3]),
}

col_ctrl, col_plot = st.columns([1, 2], gap="large")

with col_ctrl:
    st.markdown(
        f'<div style="color:{TEXT_PRIMARY};font-weight:600;margin-bottom:4px;">Kernel</div>',
        unsafe_allow_html=True,
    )
    kernel_name = st.selectbox(
        "Kernel type",
        list(KERNELS.keys()),
        label_visibility="collapsed",
    )
    kernel = KERNELS[kernel_name]

    position = st.slider(
        "Kernel position (slide it across the signal!)",
        min_value=0,
        max_value=len(signal_short) - len(kernel),
        value=10,
        step=1,
    )

    # Compute the dot product at the current position
    window = signal_short[position : position + len(kernel)]
    dot = float(np.sum(window * kernel))

    st.markdown(metric_card("Dot product", f"{dot:+.3f}"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{TEXT_SECONDARY};font-size:0.82rem;margin-top:8px;line-height:1.5;">'
        f"Kernel weights: <code>{np.round(kernel, 2).tolist()}</code><br>"
        f"Signal window: <code>{np.round(window, 2).tolist()}</code><br>"
        "The dot product becomes one output sample."
        "</div>",
        unsafe_allow_html=True,
    )

with col_plot:
    # Full convolution output
    full_conv = np.convolve(signal_short, kernel[::-1], mode="valid")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        subplot_titles=("Input signal (kernel window highlighted)", "Output feature map"),
    )

    x_sig = np.arange(len(signal_short))
    fig.add_trace(
        go.Scatter(
            x=x_sig,
            y=signal_short,
            mode="lines+markers",
            line=dict(color=ACCENT_CYAN, width=2),
            marker=dict(size=5, color=ACCENT_CYAN),
            name="signal",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    # Highlight the window
    fig.add_trace(
        go.Scatter(
            x=x_sig[position : position + len(kernel)],
            y=signal_short[position : position + len(kernel)],
            mode="markers",
            marker=dict(size=14, color=ACCENT_AMBER, line=dict(color=ACCENT_AMBER, width=2)),
            name="window",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_vrect(
        x0=position - 0.4,
        x1=position + len(kernel) - 0.6,
        fillcolor=ACCENT_AMBER,
        opacity=0.12,
        line_width=0,
        row=1,
        col=1,
    )

    # Output feature map
    x_out = np.arange(len(full_conv))
    fig.add_trace(
        go.Scatter(
            x=x_out,
            y=full_conv,
            mode="lines+markers",
            line=dict(color=ACCENT_PURPLE, width=2),
            marker=dict(size=5, color=ACCENT_PURPLE),
            name="feature map",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    # Mark the current output sample
    if position < len(full_conv):
        fig.add_trace(
            go.Scatter(
                x=[position],
                y=[full_conv[position]],
                mode="markers",
                marker=dict(size=16, color=ACCENT_AMBER, symbol="diamond",
                            line=dict(color=ACCENT_AMBER, width=2)),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=420,
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    callout_box(
        "<strong>Try each kernel:</strong> Smoothing blurs high frequencies, the edge "
        "detector lights up where the signal changes fast, and the peak detector lights "
        "up on local bumps. A CNN <em>learns</em> such kernels from data instead of "
        "hand-designing them.",
        icon="💡",
        color=ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 2 — Building a CNN
# ===========================================================================
st.markdown(section_header("2. Stacking Convolutions = a CNN"), unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:900px;">'
    "A <strong>Convolutional Neural Network</strong> is just many convolution layers "
    "stacked on top of each other, usually with two extra ingredients between them:"
    "<ul style='margin-top:8px;'>"
    "<li><strong>Non-linearity (ReLU)</strong> — sets negative values to zero, so the "
    "network can express more than a linear filter.</li>"
    "<li><strong>Pooling / striding</strong> — shrinks the signal (e.g. keeps every "
    "other sample or the local max), which forces deeper layers to look at larger "
    "chunks of time.</li>"
    "</ul>"
    "Early layers learn tiny local patterns (sharp transients, oscillations). Deeper "
    "layers combine them into larger, more abstract features (spindles, bursts, "
    "seizure-like rhythms)."
    "</div>",
    unsafe_allow_html=True,
)

# Interactive: receptive field grows with depth
col_ctrl2, col_plot2 = st.columns([1, 2], gap="large")

with col_ctrl2:
    n_layers = st.slider("Number of conv layers", 1, 6, 3, 1, key="n_layers")
    k_size = st.slider("Kernel size", 3, 9, 3, 2, key="k_size")
    stride = st.slider("Stride / pooling factor", 1, 3, 2, 1, key="stride")

    # Receptive field grows: RF_L = RF_{L-1} + (k-1) * prod(strides up to L-1)
    rf = 1
    jump = 1
    for _ in range(n_layers):
        rf = rf + (k_size - 1) * jump
        jump = jump * stride

    st.markdown(metric_card("Receptive field", f"{rf} samples"), unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{TEXT_SECONDARY};font-size:0.82rem;margin-top:8px;line-height:1.5;">'
        "The receptive field is how many input samples a single deep-layer neuron "
        "'sees'. Deeper layers + pooling = larger receptive field = coarser, more "
        "abstract patterns."
        "</div>",
        unsafe_allow_html=True,
    )

with col_plot2:
    # Show stacked feature maps as horizontal bars shrinking with depth
    fig2 = go.Figure()
    input_len = 64
    current_len = input_len
    layer_colors = [ACCENT_CYAN, ACCENT_GREEN, ACCENT_AMBER, ACCENT_PURPLE, ACCENT_RED, "#ec4899"]
    labels = []
    y_positions = []

    for layer_i in range(n_layers + 1):
        y = n_layers - layer_i
        if layer_i == 0:
            label = f"Input  ({input_len} samples)"
            color = TEXT_SECONDARY
            length = input_len
        else:
            length = max(1, current_len // stride)
            current_len = length
            label = f"Conv {layer_i}  ({length} samples)"
            color = layer_colors[(layer_i - 1) % len(layer_colors)]

        fig2.add_trace(
            go.Bar(
                x=[length],
                y=[y],
                orientation="h",
                marker=dict(color=color, line=dict(color=color, width=0)),
                hovertemplate=f"{label}<extra></extra>",
                showlegend=False,
                width=0.6,
            )
        )
        labels.append(label)
        y_positions.append(y)

    fig2.update_layout(
        height=60 + 50 * (n_layers + 1),
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        margin=dict(l=150, r=20, t=30, b=30),
        xaxis=dict(title="Feature-map length", gridcolor="#1e293b", range=[0, input_len + 2]),
        yaxis=dict(
            tickmode="array",
            tickvals=y_positions,
            ticktext=labels,
            gridcolor="#1e293b",
        ),
        bargap=0.35,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    callout_box(
        "This exact idea is what Paper 1 (CNN+Transformer for seizure detection) and "
        "Paper 2 (CoSupFormer's dual-path dilated CNN encoder) rely on. Paper 3's "
        "SpecHi-Net uses a U-shaped CNN over STFT spectrograms.",
        icon="🧠",
        color=ACCENT_CYAN,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 3 — Self-Attention & Transformers
# ===========================================================================
st.markdown(section_header("3. Self-Attention: the Transformer Core"), unsafe_allow_html=True)

st.markdown(
    f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.7;max-width:900px;">'
    "Convolutions are great at <em>local</em> patterns but struggle with very long-range "
    "dependencies. <strong>Self-attention</strong> fixes that: every position in the "
    "sequence looks at every other position and decides how much to listen to it."
    "<br><br>"
    "For each position the model computes three vectors: a <strong>Query (Q)</strong> — "
    "'what am I looking for?', a <strong>Key (K)</strong> — 'what do I offer?', and a "
    "<strong>Value (V)</strong> — 'what information do I carry?'. The attention weight "
    "from position <em>i</em> to position <em>j</em> is the similarity between "
    "<em>Q<sub>i</sub></em> and <em>K<sub>j</sub></em>, passed through a softmax."
    "</div>",
    unsafe_allow_html=True,
)

st.latex(r"\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V")

# Interactive: pick a query position, see attention weights to all others
st.markdown(
    f'<div style="color:{TEXT_PRIMARY};font-weight:600;margin-top:16px;margin-bottom:4px;">'
    "Interactive demo: which time steps does a chosen position attend to?</div>",
    unsafe_allow_html=True,
)

col_ctrl3, col_plot3 = st.columns([1, 2], gap="large")

# Build a toy sequence: 16 tokens representing "time steps" with simple feature vectors
N_TOKENS = 16
rng = np.random.default_rng(0)
# Make some clear structure: 4 groups of 4 similar tokens, so attention has something to find
group_protos = rng.standard_normal((4, 8))
tokens = np.vstack([
    group_protos[i // 4] + 0.25 * rng.standard_normal(8) for i in range(N_TOKENS)
])

with col_ctrl3:
    query_pos = st.slider("Query position (which time step is looking?)",
                          0, N_TOKENS - 1, 5, 1, key="query_pos")
    temperature = st.slider("Softmax temperature √d_k",
                            0.5, 4.0, 2.0, 0.1, key="attn_temp",
                            help="Larger = softer/smoother attention; smaller = sharper, more selective")

    # Compute dot products of the query with all keys (here Q=K=tokens for illustration)
    q = tokens[query_pos]
    scores = tokens @ q / temperature
    scores = scores - scores.max()  # numerical stability
    weights = np.exp(scores)
    weights = weights / weights.sum()

    top_idx = int(np.argmax(weights))
    st.markdown(
        metric_card("Strongest match", f"pos {top_idx}", f"w = {weights[top_idx]:.2f}"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="color:{TEXT_SECONDARY};font-size:0.82rem;margin-top:8px;line-height:1.5;">'
        "The attention output at the query position is a <em>weighted average</em> of "
        "all value vectors, using these weights. Positions in the same underlying "
        "'group' get the highest weight."
        "</div>",
        unsafe_allow_html=True,
    )

with col_plot3:
    # Bar chart of attention weights
    colors_bar = [
        ACCENT_AMBER if i == query_pos else (ACCENT_PURPLE if i == top_idx else ACCENT_CYAN)
        for i in range(N_TOKENS)
    ]
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=list(range(N_TOKENS)),
            y=weights,
            marker=dict(color=colors_bar),
            hovertemplate="pos %{x}<br>weight %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    # Mark the 4 conceptual groups
    for g in range(4):
        fig3.add_vrect(
            x0=g * 4 - 0.5,
            x1=g * 4 + 3.5,
            fillcolor=["#1e293b", "#0f172a"][g % 2],
            opacity=0.35,
            line_width=0,
            layer="below",
        )
    fig3.update_layout(
        height=360,
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_PRIMARY, family="Inter, sans-serif"),
        margin=dict(l=40, r=20, t=40, b=40),
        title=dict(text=f"Attention weights from position {query_pos}",
                   font=dict(color=TEXT_PRIMARY, size=14)),
        xaxis=dict(title="Key position", gridcolor="#1e293b",
                   tickmode="linear", dtick=1),
        yaxis=dict(title="Weight", gridcolor="#1e293b", range=[0, max(weights) * 1.15]),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    callout_box(
        "<strong>Why it matters for EEG:</strong> a seizure can start on one electrode "
        "and spread to another hundreds of milliseconds later. Self-attention lets "
        "the model directly connect 'frontal electrode at t=2s' with 'occipital electrode "
        "at t=2.4s' — something local convolutions cannot do cheaply. This is exactly "
        "the motivation for the attention module in Paper 2 and the transformer in "
        "Paper 1's pipeline.",
        icon="⚡",
        color=ACCENT_PURPLE,
    ),
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 4 — How the three papers use these ingredients
# ===========================================================================
st.markdown(section_header("4. How Each Paper Uses These Ingredients"),
            unsafe_allow_html=True)

col_p1, col_p2, col_p3 = st.columns(3, gap="medium")

with col_p1:
    st.markdown(
        neuro_card(
            f'<div style="margin-bottom:8px;">{paper_badge("Paper 1")}</div>'
            f'<h4 style="color:{ACCENT_AMBER};margin:4px 0 8px 0;">CNN + Transformer</h4>'
            f'<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">'
            "A <strong>1D CNN</strong> extracts local features from raw EEG, then a "
            "<strong>transformer</strong> aggregates them across time to detect "
            "seizure events in long continuous recordings."
            "</div>",
            border_color=ACCENT_AMBER,
        ),
        unsafe_allow_html=True,
    )

with col_p2:
    st.markdown(
        neuro_card(
            f'<div style="margin-bottom:8px;">{paper_badge("Paper 2")}</div>'
            f'<h4 style="color:{ACCENT_CYAN};margin:4px 0 8px 0;">Dilated CNN + Multi-Scale Attention</h4>'
            f'<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">'
            "CoSupFormer uses <strong>dilated convolutions</strong> (CNNs with gaps in "
            "the kernel) to cover multiple frequency bands, then a "
            "<strong>specialized attention block</strong> models within- and "
            "cross-channel interactions across time — plus a gating unit to suppress "
            "bad electrodes."
            "</div>",
            border_color=ACCENT_CYAN,
        ),
        unsafe_allow_html=True,
    )

with col_p3:
    st.markdown(
        neuro_card(
            f'<div style="margin-bottom:8px;">{paper_badge("Paper 3")}</div>'
            f'<h4 style="color:{ACCENT_PURPLE};margin:4px 0 8px 0;">U-Net CNN + Transformer + MoE</h4>'
            f'<div style="color:{TEXT_SECONDARY};font-size:0.88rem;line-height:1.6;">'
            "SpecMoE operates on STFT spectrograms. A <strong>U-shaped CNN</strong> "
            "encodes/decodes them, a <strong>transformer</strong> reasons over the "
            "compressed representation, and a <strong>spectral Mixture-of-Experts</strong> "
            "routes different frequency regimes to different sub-networks."
            "</div>",
            border_color=ACCENT_PURPLE,
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    callout_box(
        "Notice the pattern: <strong>CNN for local time/frequency structure</strong>, "
        "<strong>attention for long-range and cross-channel relationships</strong>. "
        "Every page from here on builds on this duo.",
        icon="🧩",
        color=ACCENT_GREEN,
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
render_bottom_nav(
    prev_page=("pages/1_EEG_Fundamentals.py", "EEG Fundamentals"),
    next_page=("pages/3_Detection_Challenge.py", "Detection Challenge"),
)

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
