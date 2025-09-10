from __future__ import annotations
import io
import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

APP_NAME = "🪙 CoinWizard — Appraise with Verdict"
st.set_page_config(page_title=APP_NAME, page_icon="🪙", layout="wide")

# ----------------------------- Helpers -----------------------------
GRADE_ORDER = ["G","VG","F","VF","XF","AU","MS"]
GRADE_MULT = {"G":0.25,"VG":0.4,"F":0.6,"VF":0.8,"XF":1.0,"AU":1.4,"MS":2.0}
DENOM_SYNONYMS = {
    "penny":"penny","cent":"penny","one cent":"penny",
    "nickel":"nickel","five cent":"nickel",
    "dime":"dime","ten cent":"dime",
    "quarter":"quarter","25c":"quarter",
    "half dollar":"half dollar","50c":"half dollar",
    "dollar":"dollar","1 dollar":"dollar",
}

@st.cache_data(show_spinner=False)
def load_guide() -> Optional[pd.DataFrame]:
    # Look for a local CSV in repo root
    for name in ["price_guide.csv", "prices.csv", "guide.csv"]:
        if os.path.exists(name):
            try:
                df = pd.read_csv(name)
                df.columns = [c.strip().lower() for c in df.columns]
                # minimal required
                req = {"country","denomination","year","mint","grade","price"}
                if not req.issubset(df.columns):
                    return None
                # normalize types/strings
                df = df.fillna("")
                df["country"] = df["country"].astype(str).str.strip().str.lower()
                df["denomination"] = df["denomination"].astype(str).str.strip().str.lower().map(lambda x: DENOM_SYNONYMS.get(x,x))
                df["year"] = df["year"].astype(str).str.strip()
                df["mint"] = df["mint"].astype(str).str.strip().str.lower()
                df["grade"] = df["grade"].astype(str).str.strip().str.upper()
                df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
                return df
            except Exception:
                return None
    return None


def lookup_price(df: pd.DataFrame, *, country: str, denom: str, year: str, mint: str, grade: str) -> Optional[float]:
    # normalize inputs to match guide
    country = str(country).strip().lower()
    denom = DENOM_SYNONYMS.get(str(denom).strip().lower(), str(denom).strip().lower())
    year = str(year).strip()
    mint_l = str(mint).strip().lower()
    grade = str(grade).strip().upper()

    # exact match
    q = (
        (df["country"]==country) &
        (df["denomination"]==denom) &
        (df["year"]==year) &
        (df["grade"]==grade)
    )
    cand = df[q]
    if len(cand):
        # prefer mint match if present
        with_mint = cand[cand["mint"]==mint_l]
        if len(with_mint):
            return float(with_mint.iloc[0]["price"]) 
        # fallback: any mint if exact not found
        return float(cand.iloc[0]["price"]) 
    # fallback: ignore mint & grade, then scale by grade multiplier
    q2 = (
        (df["country"]==country) & (df["denomination"]==denom) & (df["year"]==year)
    )
    cand2 = df[q2]
    if len(cand2):
        base = float(cand2.iloc[0]["price"])  # treat CSV price as baseline
        return base * GRADE_MULT.get(grade,1.0)
    return None


def verdict_from_price(price: float) -> tuple[str,str]:
    # thresholds you can tune later
    if price is None:
        return ("No match", "We couldn’t find this coin in your guide. Add a row to price_guide.csv.")
    if price >= 100:
        return ("Good 💎", "Potentially valuable. Consider a professional grading or listing.")
    if price >= 10:
        return ("Maybe 🤔", "Worth a closer look; check for varieties and condition.")
    return ("Skip 💤", "Common value. Keep if you’re collecting sets; otherwise low resale.")

# ----------------------------- UI -----------------------------
st.title(APP_NAME)
with st.sidebar:
    st.header("How to use")
    st.write("1) Add photo → 2) Fill fields → 3) Appraise → 4) See Verdict")
    guide = load_guide()
    st.success("Price guide loaded") if guide is not None else st.warning("No price_guide.csv found or wrong columns.")

st.subheader("1) Add a coin photo")
src = st.radio("Source", ["Upload photo", "Use camera"], horizontal=True)
image: Optional[Image.Image] = None
if src == "Upload photo":
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"])
    if up is not None:
        image = Image.open(up).convert("RGB")
        image = ImageOps.exif_transpose(image)
else:
    shot = st.camera_input("Take a photo")
    if shot is not None:
        image = Image.open(shot).convert("RGB")
        image = ImageOps.exif_transpose(image)

if image is not None:
    st.image(image, caption="Selected image", use_container_width=True)
else:
    st.info("Add a photo to continue.")

st.subheader("2) Appraisal inputs")
col1, col2 = st.columns(2)
with col1:
    country = st.text_input("Country", "united states")
    denom = st.text_input("Denomination", "penny")
    year  = st.text_input("Year", "2022")
with col2:
    mint  = st.text_input("Mint (optional)", "d")
    grade = st.selectbox("Grade", GRADE_ORDER, index=GRADE_ORDER.index("XF"))

if st.button("Appraise", use_container_width=True):
    df = load_guide()
    if df is None:
        st.error("No valid price_guide.csv. Expected columns: country, denomination, year, mint, grade, price")
    else:
        price = lookup_price(df, country=country, denom=denom, year=year, mint=mint, grade=grade)
        verdict, note = verdict_from_price(price)
        if price is None:
            st.error(note)
        else:
            st.success(f"Estimate: ${price:,.2f}")
            st.info(f"Verdict: **{verdict}** — {note}")
