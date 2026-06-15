"""Load pre-computed JSON/CSV data files."""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@st.cache_data
def load_results(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/results/."""
    path = os.path.join(_DATA_DIR, "results", filename)
    return pd.read_csv(path)


@st.cache_data
def load_embeddings(filename: str) -> list[dict]:
    """Load a JSON embedding file from data/embeddings/."""
    path = os.path.join(_DATA_DIR, "embeddings", filename)
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_attention(filename: str) -> np.ndarray:
    """Load an attention matrix from data/attention/ (JSON with 2-D list)."""
    path = os.path.join(_DATA_DIR, "attention", filename)
    with open(path) as f:
        data = json.load(f)
    return np.array(data["matrix"]), data.get("labels_x", None), data.get("labels_y", None)
