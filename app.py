from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

APP_NAME = "🪙 CoinWiz — Appraise with Your Own Guide"
st.set_page_config(page_title=APP_NAME, page_icon="🪙", layout="wide")

# --- Helpers ---
DENOM_SYNONYMS = {
    "penny":"penny","cent":"penny","one cent":"penny",
    "nickel":"nickel","five cent":"nickel",
    "dime":"dime","ten cent":"dime",
    "quarter":"quarter","25c":"quarter",
    "half dollar":"half dollar","50c":"half dollar",
    "dollar":"dollar","1 dollar":"dollar",
    "morgan":"morgan dollar","morgan dollar":"morgan dollar",
    "lincoln":"lincoln cent","lincoln cent":"lincoln cent",
}

def norm(s: str) -> str:
    return (str(s or "")).strip().lower()

def norm_denom(d: str) -> str:
    d = norm(d)
    return DENOM_SYNONYMS.get(d, d)

@st.cache_data
def load_guide() -> pd.DataFrame:
    try:
        df = pd.read_csv("price_guide.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        # ensure needed columns exist
        for c in ["country","denomination","year","mint","grade","price","value_low","value_high"]:
            if c not in df.columns:
                df[c] = "" if c in {"country","denomination","year","mint","grade"} else np.nan
        return df
    except Exception:
        return pd.DataFrame()

def match_row(df: pd.DataFrame, country: str, denom: str, year: str, mint: str, grade: str) -> Optional[dict]:
    if df.empty:
        return None
    c, d, y, m, g = norm(country), norm_denom(denom), str(year).strip(), norm(mint), str(grade).strip().upper()
    df2 = df.copy()
    df2["country"]      = df2["country"].astype(str).str.lower().str.strip()
    df2["denomination"] = df2["denomination"].astype(str).str.lower().str.strip().replace({"cent":"penny"})
    df2["year"]         = df2["year"].astype(str).str.strip()
    df2["mint"]         = df2["mint"].astype(str).str.lower().str.strip()
    if "grade" in df2.columns:
        df2["grade"]    = df2["grade"].astype(str).str.upper().str.strip()
    q = (df2["country"]==c) & (df2["denomination"]==d) & (df2["year"]==y)
    cand = df2[q]
    if m:
        exact = cand[cand["mint"]==m]
        if len(exact):
            cand = exact
    if "grade" in cand.columns and g:
        gmatch = cand[cand["grade"]==g]
        if len(gmatch):
            cand = gmatch
    if cand.empty:
        return None
    return cand.iloc[0].to_dict()

def compute_estimate(row: Optional[dict]) -> Optional[float]:
    if row is None:
        return None
    # prefer value_low/value_high average if present
    try:
        lo, hi = float(row.get("value_low")), float(row.get("value_high"))
        if not np.isnan(lo) and not np.isnan(hi):
            return (lo + hi) / 2.0
    except Exception:
        pass
    # fallback to single price
    try:
        return float(row.get("price"))
    except Exception:
        return None

def verdict(estimate: Optional[float], grade: str) -> str:
    if estimate is None:
        return "No match"
    if estimate >= 100:
        return "Good 💎"
    if estimate >= 10:
        return "Maybe 🤔"
    return "Common 💤"

# --- UI ---
st.title(APP_NAME)
st.caption("Take or upload a coin photo (optional), fill details, then press **Appraise**.")

# 1) Photo (optional)
st.subheader("1) Add a coin photo")
src = st.radio("Source", ["Upload photo","Use camera"], horizontal=True)
image = None
if src == "Upload photo":
    up = st.file_uploader("Upload coin image", type=["jpg","jpeg","png","webp"])
    if up:
        image = ImageOps.exif_transpose(Image.open(up).convert("RGB"))
else:
    shot = st.camera_input("Take a photo")
    if shot:
        image = ImageOps.exif_transpose(Image.open(shot).convert("RGB"))
if image is not None:
    st.image(image, caption="Selected image", use_container_width=True)

# 2) Details
st.subheader("2) Details")
col1, col2 = st.columns(2)
with col1:
    country = st.text_input("Country", "united states")
    denom   = st.text_input("Denomination", "penny")
    year    = st.text_input("Year", "2022")
with col2:
    mint    = st.text_input("Mint (optional)", "d")
    grade   = st.selectbox("Grade", ["G","VG","F","VF","XF","AU","MS"], index=4)

# 3) Appraise
st.subheader("3) Appraise")
if st.button("Appraise", use_container_width=True):
    df = load_guide()
    if df.empty:
        st.error("No price_guide.csv found (or it’s empty). Add one to the repo and try again.")
    else:
        row = match_row(df, country, denom, year, mint, grade)
        est = compute_estimate(row)
        if est is None:
            st.error("Could not appraise. Add a matching row to price_guide.csv.")
            with st.expander("Show first rows of your guide"):
                st.dataframe(df.head(20), use_container_width=True)
        else:
            low, high = est*0.85, est*1.15
            tag = verdict(est, grade)
            st.success(f"Estimate: ${est:,.2f} (range ${low:,.2f}–${high:,.2f}) — **{tag}**")
            if row:
                st.caption(
                    f"Matched: {row.get('country','').title()} {row.get('denomination','')} "
                    f"{row.get('year','')} {str(row.get('mint','')).upper()} {row.get('grade','')}"
                )
