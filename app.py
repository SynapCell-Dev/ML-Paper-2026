"""EEG DeepDive -- Main entry point for the Streamlit application."""

import streamlit as st

st.set_page_config(
    page_title="EEG DeepDive",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Navigation ----
home_page = st.Page("home.py", title="Home", icon="🏠")

module_pages = [
    st.Page("pages/1_EEG_Fundamentals.py", title="EEG Fundamentals", icon="🧠"),
    st.Page("pages/2_DeepLearning_Primer.py", title="Deep Learning Primer", icon="🧭"),
    st.Page("pages/3_Detection_Challenge.py", title="Detection Challenge", icon="🔍"),
    st.Page("pages/4_MultiScale_Encoding.py", title="Multi-Scale Encoding", icon="📊"),
    st.Page("pages/5_Attention_and_Gating.py", title="Attention & Gating", icon="🎯"),
    st.Page("pages/6_Learning_Representations.py", title="Learning Representations", icon="📐"),
    st.Page("pages/7_Foundation_Models.py", title="Foundation Models", icon="🏗️"),
    st.Page("pages/8_Results_and_Impact.py", title="Results & Impact", icon="🏆"),
]

reference_pages = [
    st.Page("pages/9_Datasets.py", title="Datasets Reference", icon="📚"),
    st.Page("pages/10_Quiz.py", title="Knowledge Check", icon="🧩"),
]

page = st.navigation(
    {
        "Home": [home_page],
        "Modules": module_pages,
        "Reference": reference_pages,
    }
)

page.run()
