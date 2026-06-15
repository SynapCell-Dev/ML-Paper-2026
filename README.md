# EEG DeepDive

**A Visual Tour Through Three Papers on EEG Deep Learning**

<!-- Screenshot placeholder: replace with an actual screenshot of the app -->
<!-- ![EEG DeepDive Screenshot](screenshot.png) -->

---

## What You'll Learn

This interactive Streamlit app walks you through three research papers that push
the boundaries of EEG analysis with deep learning:

1. **The Detection Challenge** — Thomas et al., *Neuroscience Informatics*, 2026
   Why 95 % classification accuracy does not mean your seizure detector works, and
   how a CNN+Transformer trained on mouse EEG generalises to human data (F1 = 0.935).

2. **CoSupFormer** — Darankoum et al., *arXiv*, 2025
   A dual-path dilated CNN encoder, global gated attention, and a hybrid
   supervised + contrastive loss that delivers noise-robust EEG classification.

3. **SpecMoE** — Darankoum et al., *arXiv*, 2026
   A spectral Mixture-of-Experts foundation model pretrained on ~9 000 hours of
   clinical EEG, achieving state-of-the-art on 7/9 benchmarks including
   cross-species tasks.

---

## Intended Audiences

| Audience | Suggested Path |
|----------|---------------|
| **Pharma professionals** | EEG Fundamentals → Detection Challenge → Results & Impact |
| **Students** | All seven modules in order |
| **Researchers** | Jump to any technical module |

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/BaguettePrime/Davy_Dev.git
cd Davy_Dev

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Streamlit Cloud Deployment

1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select this repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.

No GPU or large dependencies are required — all results are pre-computed.

---

## Folder Structure

```
.
├── .streamlit/config.toml        # Theme and server configuration
├── app.py                        # Entry point (st.navigation)
├── home.py                       # Landing page
├── pages/
│   ├── 1_EEG_Fundamentals.py     # What is EEG, frequency bands, pharma context
│   ├── 2_Detection_Challenge.py  # Paper 1: classification vs detection
│   ├── 3_MultiScale_Encoding.py  # Paper 2: dilated CNN encoder
│   ├── 4_Attention_and_Gating.py # Paper 2: global attention + gating
│   ├── 5_Learning_Representations.py  # Paper 2→3 bridge: contrastive learning
│   ├── 6_Foundation_Models.py    # Paper 3: SpecMoE
│   └── 7_Results_and_Impact.py   # Dashboard, timeline, glossary
├── components/
│   └── progress_tracker/         # Custom Streamlit progress-bar component
├── utils/
│   ├── style.py                  # CSS theme, colour constants, HTML helpers
│   ├── eeg_synth.py              # Synthetic EEG generation (numpy/scipy)
│   ├── viz.py                    # Plotly visualisation helpers
│   └── data_loader.py            # CSV / JSON data loading
├── data/
│   ├── results/                  # Pre-computed result tables (CSV)
│   ├── embeddings/               # Synthetic 2-D projections (JSON)
│   ├── attention/                # Attention heatmap example (JSON)
│   ├── Paper1.pdf
│   ├── Paper2.pdf
│   └── Paper3.pdf
└── requirements.txt
```

---

## License

<!-- Replace with your chosen licence -->
All rights reserved.
