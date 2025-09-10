from __future__ import annotations
import io, os, re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# deps: easyocr + torch + opencv-headless
import easyocr
import cv2

APP_NAME = "🪙 CoinWiz — Snap → Identify → Appraise"
st.set_page_config(page_title=APP_NAME, page_icon="🪙", layout="centered")

# --------------------- Data loaders ---------------------
@st.cache_resource(show_spinner=False)
def ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

@st.cache_data
def load_guide(path: str = "price_guide.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["country","denomination","year","mint","grade","price"])
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["country","denomination","year","mint","grade"]:
        if c not in df.columns:
            df[c] = ""
    if "price" not in df.columns:
        df["price"] = np.nan
    df["country"] = df["country"].astype(str).str.strip().str.lower()
    df["denomination"] = df["denomination"].astype(str).str.strip().str.lower().replace({"cent":"penny"})
    df["year"] = df["year"].astype(str).str.strip()
    df["mint"] = df["mint"].astype(str).str.strip().str.lower()
    df["grade"] = df["grade"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

@st.cache_data
def load_varieties(path: str = "varieties.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["country","denomination","year","mint","variety","variety_code","markers","ref"])
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ["country","denomination","mint","variety","variety_code","markers","ref"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    if "year" in df.columns:
        df["year"] = df["year"].astype(str).str.strip()
    return df

# --------------------- OCR + heuristics ---------------------
DENOM_WORDS = {
    "penny": ["cent","one cent","lincoln","wheat","memorial","shield"],
    "nickel": ["nickel","five cent","monticello","jefferson"],
    "dime": ["dime","one dime","ten cent","roosevelt","mercury"],
    "quarter": ["quarter","quarter dollar","25 c","25c","washington","state quarter"],
    "half dollar": ["half dollar","50 c","50c","kennedy"],
    "morgan dollar": ["morgan"],
    "peace dollar": ["peace"],
    "dollar": ["one dollar","1 dollar","eisenhower","susan","sacagawea","native"]
}
YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")

def guess_from_ocr(img: Image.Image) -> dict:
    rdr = ocr_reader()
    # modest resize for speed
    W = 900
    if img.width > W:
        img = img.resize((W, int(img.height*W/img.width)))
    text_blocks = rdr.readtext(np.array(img), detail=0, paragraph=True)
    text = " ".join([t for t in text_blocks if isinstance(t,str)])
    t = text.lower()

    # year
    year = None
    m = YEAR_RE.search(t)
    if m:
        year = m.group(0)

    # mint mark (simple): D/S/P/W as standalone letters near date text
    mint = None
    for mm in ["d","s","p","w"]:
        if re.search(rf"\b{mm}\b", t):
            mint = mm; break

    # denomination by keywords
    denom = None
    best_score, best_d = 0, None
    for d, kws in DENOM_WORDS.items():
        sc = sum(1 for kw in kws if kw in t)
        if sc > best_score:
            best_score, best_d = sc, d
    denom = best_d

    return {"denomination": denom, "year": year, "mint": mint}

# simple grade from sharpness/contrast (no ML)
def grade_from_image(pil: Image.Image) -> str:
    arr = np.array(pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharp > 650: return "MS"
    if sharp > 450: return "AU"
    if sharp > 300: return "XF"
    if sharp > 180: return "VF"
    if sharp > 100: return "F"
    if sharp > 60:  return "VG"
    return "G"

# --------------------- Price + varieties ---------------------
GRADE_ORDER = ["G","VG","F","VF","XF","AU","MS"]

def pick_row(df: pd.DataFrame, country: str, denom: str, year: str, mint: str, grade: str) -> Optional[dict]:
    if df.empty: return None
    q = (df["country"]==country) & (df["denomination"]==denom) & (df["year"]==str(year))
    cand = df[q]
    if mint:
        exact = cand[cand["mint"]==mint]
        if len(exact): cand = exact
    if grade in GRADE_ORDER and "grade" in cand.columns:
        exact = cand[cand["grade"]==grade]
        if len(exact): cand = exact
    if cand.empty: return None
    return cand.iloc[0].to_dict()

def verdict_from_price(val: Optional[float], grade: str) -> str:
    if val is None or np.isnan(val): return "Unknown"
    if val >= 100: return "Good 💎"
    if val >= 10 or grade in {"AU","MS"}: return "Maybe 🤔"
    return "Common 💤"

# --------------------- UI ---------------------
st.title(APP_NAME)
st.caption("Zero typing: photo → OCR → value → variety alert")

up = st.file_uploader("Upload coin photo", type=["jpg","jpeg","png","webp"])
shot = st.camera_input("…or take a photo")
img = None
if up: img = Image.open(up).convert("RGB")
elif shot: img = Image.open(shot).convert("RGB")

if img is None:
    st.info("Add a photo to begin.")
    st.stop()

img = ImageOps.exif_transpose(img)
st.image(img, use_column_width=True)

with st.status("Analyzing…", expanded=False) as s:
    s.update(label="Reading text (OCR)…")
    guess = guess_from_ocr(img)

    s.update(label="Estimating grade…")
    grade = grade_from_image(img)

    # defaults/fallbacks
    country = "united states"
    denom   = (guess.get("denomination") or "quarter").lower()
    year    = guess.get("year") or "1986"
    mint    = (guess.get("mint") or "").lower()

    s.update(label="Looking up price…")
    guide = load_guide()
    row = pick_row(guide, country, denom, year, mint, grade)
    val = float(row["price"]) if row and not pd.isna(row.get("price")) else None

    s.update(label="Checking known varieties…")
    vdf = load_varieties()
    alert = None
    if not vdf.empty:
        hits = vdf[
            (vdf["country"]==country) &
            (vdf["denomination"]==denom) &
            (vdf["year"]==str(year)) &
            ((vdf["mint"]==mint) | (vdf["mint"]==""))
        ]
        if not hits.empty:
            alert = hits.iloc[0].to_dict()

    s.update(label="Done", state="complete")

colA, colB = st.columns(2)
with colA:
    st.metric("Denomination", denom.title())
    st.metric("Year", str(year))
with colB:
    st.metric("Mint", mint.upper() or "—")
    st.metric("Grade (est.)", grade)

if val is None:
    st.warning("No price found in your guide for this guess. You can add rows to price_guide.csv.")
else:
    tag = verdict_from_price(val, grade)
    low, high = val*0.85, val*1.15
    st.success(f"Estimate: **${val:,.2f}** (range ${low:,.2f}–${high:,.2f}) — **{tag}**")

if alert:
    st.warning(
        f"**Variety alert:** {alert.get('variety','').title()} — {alert.get('markers','')}\n\n"
        f"Reference: {alert.get('ref','')}",
        icon="🔎"
    )

# Optional: quick re-appraise controls (only if guess is off)
with st.expander("Adjust guess & re-appraise (optional)"):
    denom2 = st.selectbox("Denomination", ["penny","nickel","dime","quarter","half dollar","dollar","morgan dollar","peace dollar"], index=["penny","nickel","dime","quarter","half dollar","dollar","morgan dollar","peace dollar"].index(denom) if denom in ["penny","nickel","dime","quarter","half dollar","dollar","morgan dollar","peace dollar"] else 3)
    year2  = st.text_input("Year", str(year))
    mint2  = st.text_input("Mint (optional)", mint)
    grade2 = st.selectbox("Grade", GRADE_ORDER, index=GRADE_ORDER.index(grade) if grade in GRADE_ORDER else 3)
    if st.button("Re-appraise"):
        row2 = pick_row(guide, country, denom2, year2, mint2, grade2)
        val2 = float(row2["price"]) if row2 and not pd.isna(row2.get("price")) else None
        if val2 is None:
            st.error("Still not in guide. Add a row to price_guide.csv.")
        else:
            tag2 = verdict_from_price(val2, grade2)
            low2, high2 = val2*0.85, val2*1.15
            st.success(f"Estimate: **${val2:,.2f}** (range ${low2:,.2f}–${high2:,.2f}) — **{tag2}**")
