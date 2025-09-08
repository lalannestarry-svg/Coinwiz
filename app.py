from __future__ import annotations
import io
import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import cv2

APP_NAME = "🪙 CoinWiz – Appraise with Your Own Guide"
st.set_page_config(page_title=APP_NAME, page_icon="🪙", layout="wide")

# ------------------------- Helpers -------------------------
GRADE_ORDER = ["G", "VG", "F", "VF", "XF", "AU", "MS"]
GRADE_MULT = {"G":0.25,"VG":0.4,"F":0.6,"VF":0.8,"XF":0.9,"AU":0.95,"MS":1.0}

DENOM_SYNONYMS = {
    "penny":"penny", "cent":"penny", "one cent":"penny",
    "nickel":"nickel", "five cent":"nickel",
    "dime":"dime", "ten cent":"dime",
    "quarter":"quarter","25c":"quarter",
    "half dollar":"half dollar", "50c":"half dollar",
    "dollar":"dollar","1$":"dollar",
    "morgan":"morgan dollar","morgan dollar":"morgan dollar",
    "lincoln":"lincoln cent","lincoln cent":"lincoln cent"
}

def normalize_denom(name: str) -> str:
    name = name.strip().lower()
    return DENOM_SYNONYMS.get(name, name)

def analyze_image(img: Image.Image) -> tuple[str,float]:
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    spread = gray.std()
    edge = cv2.Canny(gray,100,200).mean()

    score = 0.5*sharp + 0.3*spread + 0.2*edge
    if score < 20: grade = "G"
    elif score < 35: grade = "VG"
    elif score < 50: grade = "F"
    elif score < 65: grade = "VF"
    elif score < 80: grade = "XF"
    elif score < 100: grade = "AU"
    else: grade = "MS"
    return grade, float(score)

# ------------------------- Load Guide -------------------------
@st.cache_data
def load_priceguide() -> pd.DataFrame:
    path = "price_guide.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["country","denomination","year","mint","price"])
    df = pd.read_csv(path)
    df["denomination"] = df["denomination"].map(normalize_denom)
    return df

guide = load_priceguide()

# ------------------------- UI -------------------------
st.title(APP_NAME)
st.write("Upload a coin photo or
