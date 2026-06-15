"""Page 10 — Interactive Quiz: test your understanding across the three papers."""

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
    inject_css,
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
        "Test yourself — 10 questions covering the key ideas from all three papers",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f'<h1 style="color:{TEXT_PRIMARY};margin-bottom:0;">Knowledge Check</h1>'
    f'<p style="color:{TEXT_SECONDARY};font-size:1.05rem;margin-top:4px;">'
    "Pick one answer per question, then click <strong>Submit</strong> at the bottom. "
    "You will see per-question feedback and a final score.</p>",
    unsafe_allow_html=True,
)

# ===========================================================================
# Quiz definition
# ===========================================================================
# Each question: id, paper, difficulty, prompt, options list, correct index, explanation
QUESTIONS = [
    {
        "id": "q1",
        "paper": "EEG Basics",
        "prompt": "Which frequency band is most strongly associated with relaxed, "
                  "eyes-closed wakefulness in humans?",
        "options": [
            "Delta (1–4 Hz)",
            "Theta (4–8 Hz)",
            "Alpha (8–12 Hz)",
            "Gamma (30–100 Hz)",
        ],
        "correct": 2,
        "explanation": "Alpha rhythms (≈10 Hz) dominate the posterior scalp in "
                       "awake, eyes-closed subjects and are suppressed by visual input.",
    },
    {
        "id": "q2",
        "paper": "DL Primer",
        "prompt": "What does a 1D convolutional kernel compute at each position "
                  "as it slides along a signal?",
        "options": [
            "The variance of the signal in its window",
            "A weighted sum (dot product) of the signal values under the kernel",
            "The Fourier transform of the signal window",
            "The median of the values in the window",
        ],
        "correct": 1,
        "explanation": "A convolution produces one output sample per position by "
                       "multiplying kernel weights with the signal window and summing. "
                       "This is exactly the operation the primer's interactive slider shows.",
    },
    {
        "id": "q3",
        "paper": "DL Primer",
        "prompt": "In self-attention, what role does the <strong>Query</strong> vector play?",
        "options": [
            "It stores the information that will be averaged in the output",
            "It describes what the current position is looking for, so it can be matched against Keys",
            "It fixes the softmax temperature of the attention weights",
            "It is the positional encoding added to each token",
        ],
        "correct": 1,
        "explanation": "Q represents 'what am I looking for'. It is matched against "
                       "every Key K via a dot product, and the resulting weights "
                       "decide how much of each Value V to mix into the output.",
    },
    {
        "id": "q4",
        "paper": "Paper 1",
        "prompt": "What is the core distinction Paper 1 makes between "
                  "<strong>classification</strong> and <strong>detection</strong> of seizures?",
        "options": [
            "Classification uses more electrodes; detection uses fewer",
            "Classification uses pre-segmented labeled windows, while detection "
            "must localize events in continuous recordings",
            "Classification is unsupervised; detection is supervised",
            "Classification is for mice only; detection is for humans only",
        ],
        "correct": 1,
        "explanation": "Classification assumes inputs are already cropped to a "
                       "class. Detection works on long continuous recordings and "
                       "must answer 'is a seizure happening right now?' at every "
                       "moment — a harder, more realistic problem.",
    },
    {
        "id": "q5",
        "paper": "Paper 1",
        "prompt": "Why does Paper 1 test models trained on mouse EEG against the "
                  "Bonn (human) dataset?",
        "options": [
            "Because the Bonn dataset has the largest number of subjects",
            "To show cross-species generalization — training on animal data and "
            "evaluating on humans without fine-tuning",
            "Because mice and humans have identical EEG spectra",
            "Because Bonn is the only dataset with ictal recordings",
        ],
        "correct": 1,
        "explanation": "The mouse→human transfer is a major contribution of "
                       "Paper 1 — it's evidence that the model learns something "
                       "transferable about seizure dynamics rather than species-specific artifacts.",
    },
    {
        "id": "q6",
        "paper": "Paper 2",
        "prompt": "CoSupFormer uses <strong>dilated convolutions</strong> in its "
                  "encoder. What is the main benefit?",
        "options": [
            "They reduce the number of parameters to zero",
            "They make the signal shorter so the transformer runs faster",
            "They enlarge the receptive field cheaply, covering multiple "
            "frequency scales without extra parameters",
            "They remove the need for a non-linear activation function",
        ],
        "correct": 2,
        "explanation": "Dilation inserts gaps between kernel taps, so the "
                       "receptive field grows as kernel_size + (kernel_size−1)·(dilation−1) "
                       "without increasing parameter count. This is why the encoder can "
                       "see both fast (gamma) and slow (delta) rhythms with a small stack.",
    },
    {
        "id": "q7",
        "paper": "Paper 2",
        "prompt": "What problem does CoSupFormer's <strong>gating</strong> mechanism solve?",
        "options": [
            "It prevents overfitting on the training set",
            "It suppresses noisy or broken electrodes by down-weighting them "
            "before attention pools across channels",
            "It converts mouse EEG into human EEG",
            "It replaces the need for any convolution in the encoder",
        ],
        "correct": 1,
        "explanation": "Gating produces a per-channel weight via a sigmoid, so "
                       "channels that look noisy are multiplied by ~0 and effectively "
                       "silenced. This is critical on noisy datasets like MACO, where "
                       "accuracy jumps from 38.8% to 74.9%.",
    },
    {
        "id": "q8",
        "paper": "Paper 2",
        "prompt": "What is the purpose of the <strong>contrastive loss</strong> "
                  "added on top of cross-entropy in CoSupFormer?",
        "options": [
            "To speed up training by a factor of 10",
            "To force embeddings of the same class to cluster together and "
            "different classes to separate — improving generalization",
            "To perform data augmentation",
            "To replace the need for labels",
        ],
        "correct": 1,
        "explanation": "Cross-entropy alone only cares about the final class score. "
                       "The NT-Xent contrastive term pulls same-class embeddings "
                       "closer and pushes different-class embeddings apart, yielding "
                       "better-structured latent space and stronger cross-subject transfer.",
    },
    {
        "id": "q9",
        "paper": "Paper 3",
        "prompt": "SpecMoE is pretrained with a <strong>Gaussian mask</strong> on "
                  "STFT spectrograms rather than a rectangular mask. Why?",
        "options": [
            "Because Gaussians are faster to compute",
            "Because a smooth mask prevents the model from using the sharp mask "
            "edges as a shortcut and forces it to learn real spectral structure",
            "Because rectangular masks cannot be differentiated",
            "Because Gaussian masks save memory",
        ],
        "correct": 1,
        "explanation": "A hard rectangular boundary leaks information ('something "
                       "was clearly removed here'). A smooth Gaussian mask blends "
                       "seamlessly, so the model must actually reconstruct the masked "
                       "content from surrounding spectral context.",
    },
    {
        "id": "q10",
        "paper": "Paper 3",
        "prompt": "What does the <strong>Mixture-of-Experts (MoE)</strong> layer "
                  "in SpecMoE do?",
        "options": [
            "It replaces attention with a recurrent network",
            "It routes different spectral regimes to different expert "
            "sub-networks via a gating function, so specialists handle what "
            "they know best",
            "It compresses the model to zero parameters",
            "It performs data augmentation during fine-tuning",
        ],
        "correct": 1,
        "explanation": "The spectral gating looks at the input's PSD and routes "
                       "it to the most appropriate expert. Different experts end up "
                       "specializing in different frequency regimes, which improves "
                       "performance across diverse downstream EEG tasks.",
    },
]

# ===========================================================================
# State handling
# ===========================================================================
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {q["id"]: None for q in QUESTIONS}


def _reset_quiz():
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {q["id"]: None for q in QUESTIONS}


# ===========================================================================
# Render questions
# ===========================================================================
TAG_COLORS = {
    "EEG Basics": ACCENT_GREEN,
    "DL Primer": ACCENT_CYAN,
    "Paper 1": ACCENT_AMBER,
    "Paper 2": ACCENT_CYAN,
    "Paper 3": ACCENT_PURPLE,
}

for i, q in enumerate(QUESTIONS, start=1):
    tag_color = TAG_COLORS.get(q["paper"], ACCENT_CYAN)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:22px 0 8px 0;">'
        f'<div style="color:{TEXT_PRIMARY};font-size:1.05rem;font-weight:700;">'
        f"Question {i}</div>"
        f'<span style="background:{tag_color}22;color:{tag_color};'
        f'padding:2px 10px;border-radius:12px;font-size:0.75rem;'
        f'font-weight:600;border:1px solid {tag_color}44;">{q["paper"]}</span>'
        f"</div>"
        f'<div style="color:{TEXT_SECONDARY};font-size:0.95rem;line-height:1.6;'
        f'margin-bottom:8px;">{q["prompt"]}</div>',
        unsafe_allow_html=True,
    )

    current = st.session_state.quiz_answers.get(q["id"])
    selection = st.radio(
        label=f"Answer for Q{i}",
        options=list(range(len(q["options"]))),
        format_func=lambda idx, _q=q: _q["options"][idx],
        index=current if current is not None else None,
        key=f"radio_{q['id']}",
        label_visibility="collapsed",
        disabled=st.session_state.quiz_submitted,
    )
    st.session_state.quiz_answers[q["id"]] = selection

    # Feedback after submission
    if st.session_state.quiz_submitted:
        picked = st.session_state.quiz_answers[q["id"]]
        correct_idx = q["correct"]
        if picked == correct_idx:
            st.markdown(
                callout_box(
                    f"<strong>Correct.</strong> {q['explanation']}",
                    icon="✅",
                    color=ACCENT_GREEN,
                ),
                unsafe_allow_html=True,
            )
        else:
            picked_text = (
                q["options"][picked] if picked is not None else "<em>no answer</em>"
            )
            st.markdown(
                callout_box(
                    f"<strong>Not quite.</strong> You picked: {picked_text}.<br>"
                    f"<strong>Correct answer:</strong> {q['options'][correct_idx]}.<br>"
                    f"{q['explanation']}",
                    icon="❌",
                    color=ACCENT_RED,
                ),
                unsafe_allow_html=True,
            )

# ===========================================================================
# Submit / score
# ===========================================================================
st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

if not st.session_state.quiz_submitted:
    col_submit, _ = st.columns([1, 3])
    with col_submit:
        if st.button("Submit answers", type="primary", use_container_width=True):
            st.session_state.quiz_submitted = True
            st.rerun()
else:
    # Compute score
    score = sum(
        1 for q in QUESTIONS
        if st.session_state.quiz_answers.get(q["id"]) == q["correct"]
    )
    total = len(QUESTIONS)
    pct = int(round(100 * score / total))

    if pct >= 90:
        verdict = "🏆 Excellent — you clearly understood the three papers."
        color = ACCENT_GREEN
    elif pct >= 70:
        verdict = "👍 Solid grasp of the core ideas. A quick re-read of the modules you missed will seal it."
        color = ACCENT_CYAN
    elif pct >= 50:
        verdict = "🧠 You've got the basics. Revisit the modules flagged in red above to go deeper."
        color = ACCENT_AMBER
    else:
        verdict = "📚 This material rewards a second pass. Start from the Deep Learning Primer and work forward."
        color = ACCENT_RED

    st.markdown(
        f'<div style="background:{color}11;border:1px solid {color}55;'
        f'border-radius:12px;padding:24px;text-align:center;margin:8px 0;">'
        f'<div style="color:{TEXT_SECONDARY};font-size:0.85rem;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:6px;">Your score</div>'
        f'<div style="color:{color};font-size:2.8rem;font-weight:800;line-height:1;">'
        f"{score} / {total} &nbsp;·&nbsp; {pct}%</div>"
        f'<div style="color:{TEXT_PRIMARY};font-size:1rem;margin-top:12px;">{verdict}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    col_reset, _ = st.columns([1, 3])
    with col_reset:
        if st.button("Retake quiz", use_container_width=True, on_click=_reset_quiz):
            pass

st.markdown(FOOTER_HTML, unsafe_allow_html=True)
