
from __future__ import annotations
import io
import os
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# ============================ CONFIG / SETTINGS ============================
APP_NAME = "🪙 Coin Identifier & Appraiser (CoinScope)"
SETTINGS_FILE = ".coin_app_settings.json"  # saved next to app.py (persists if using Drive)

# ---------------------------- Page setup ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🪙", layout="wide")

# ---------------------------- Helpers: settings ----------------------------
def load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "country": "united states",
        "denomination": "quarter",
        "year": "1932",
        "mint": "d",
        "grade": "XF",
        "confidence": 0.7,
    }

def save_settings(data: dict) -> bool:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

# ---------------------------- Grading heuristics ----------------------------
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False


def _variance_of_laplacian(gray: np.ndarray) -> float:
    if not CV2_OK:
        return float(np.var(np.gradient(gray.astype(np.float32))))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _hist_spread(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    return float(entropy)


def _edge_density(gray: np.ndarray) -> float:
    if CV2_OK:
        edges = cv2.Canny(gray, 50, 150)
        return float(edges.mean())
    gx = np.abs(np.gradient(gray.astype(np.float32), axis=0))
    gy = np.abs(np.gradient(gray.astype(np.float32), axis=1))
    mag = np.hypot(gx, gy)
    return float((mag > mag.mean()).mean())


def grade_image(rgb: np.ndarray):
    if CV2_OK:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    sharp = _variance_of_laplacian(gray)
    spread = _hist_spread(gray)
    edge = _edge_density(gray)
    s_norm = np.tanh(sharp / 250.0)
    e_norm = np.tanh(edge * 5)
    sp_norm = np.clip((spread - 4.5) / 3.0, 0, 1)
    score = 100.0 * (0.55 * s_norm + 0.25 * sp_norm + 0.20 * (1 - e_norm))
    if score < 20:
        band = "G"
    elif score < 35:
        band = "VG"
    elif score < 50:
        band = "F"
    elif score < 65:
        band = "VF"
    elif score < 78:
        band = "XF"
    elif score < 90:
        band = "AU"
    else:
        band = "MS"
    return float(score), band, {"sharpness": sharp, "hist_spread": spread, "edge_density": edge}

# ---------------------------- Appraisal engine ----------------------------
GRADE_MULT = {"G": 0.25, "VG": 0.4, "F": 0.6, "VF": 0.8, "XF": 1.0, "AU": 1.4, "MS": 2.0}

_DEFAULT_PG = """country,denomination,year,mint,price
united states,quarter,1932,d,120.00
united states,quarter,1932,s,140.00
united states,quarter,1932,,30.00
united states,morgan dollar,1921,,35.00
united states,lincoln cent,1909,s,950.00
canada,quarter,1967,,4.00
"""

class AppraisalResult:
    def __init__(self, low: float, estimate: float, high: float, explanation: str, record: Dict):
        self.low, self.estimate, self.high = low, estimate, high
        self.explanation = explanation
        self.record = record

    def to_csv(self) -> str:
        import csv
        buf = io.StringIO()
        keys = ["country", "denomination", "year", "mint", "grade", "base_price", "est", "low", "high"]
        row = {
            "country": self.record.get("country"),
            "denomination": self.record.get("denomination"),
            "year": self.record.get("year"),
            "mint": self.record.get("mint"),
            "grade": self.record.get("grade"),
            "base_price": self.record.get("price"),
            "est": self.estimate,
            "low": self.low,
            "high": self.high,
        }
        w = csv.DictWriter(buf, fieldnames=keys)
        w.writeheader(); w.writerow(row)
        return buf.getvalue()

class AppraisalEngine:
    def __init__(self, csv_bytes: Optional[bytes] = None):
        try:
            if csv_bytes is None:
                self.df = pd.read_csv(io.StringIO(_DEFAULT_PG))
            else:
                self.df = pd.read_csv(io.BytesIO(csv_bytes))
            self.df.columns = [c.strip().lower() for c in self.df.columns]
            required = {"country", "denomination", "year", "mint", "price"}
            self.ready = required.issubset(self.df.columns)
            if not self.ready:
                self.df = None
        except Exception:
            self.df = None
            self.ready = False

    def _norm(self, x: str) -> str:
        return str(x).strip().lower()

    def _lookup(self, country: str, denom: str, year: str, mint: str):
        if not self.ready:
            return None
        df = self.df
        q = (
            (df["country"].str.lower() == country)
            & (df["denomination"].str.lower() == denom)
            & (df["year"].astype(str).str.lower() == str(year).lower())
        )
        cand = df[q]
        if mint:
            c2 = cand[cand["mint"].astype(str).str.lower() == str(mint).lower()]
            if len(c2):
                cand = c2
        if not len(cand):
            return None
        return cand.iloc[0].to_dict()

    def appraise(self, *, country: str, denomination: str, year: str, mint: str, grade: str, conf: float) -> Optional[AppraisalResult]:
        if not self.ready:
            return None
        country = self._norm(country)
        denomination = self._norm(denomination)
        mint = self._norm(mint)
        row = self._lookup(country, denomination, year, mint)
        if row is None:
            return None
        base = float(row.get("price", 0.0))
        mult = GRADE_MULT.get(grade, 1.0)
        est = base * mult
        width = max(0.15, 0.50 * (1.0 - float(conf)))
        low, high = est * (1.0 - width), est * (1.0 + width)
        exp = (
            f"Base ${base:.2f} for {row.get('country','').title()} {row.get('denomination','')} "
            f"{row.get('year')} {row.get('mint','').upper()} × grade '{grade}' and confidence {conf:.2f}."
        )
        rec = {
            "country": row.get("country"),
            "denomination": row.get("denomination"),
            "year": row.get("year"),
            "mint": row.get("mint", ""),
            "grade": grade,
            "price": base,
        }
        return AppraisalResult(low, est, high, exp, rec)

# ---------------------------- UI: Sidebar ----------------------------
st.title(f"{APP_NAME} — Pro Ready")
with st.sidebar:
    st.header("📘 How to use")
    st.markdown(
        """
        1) **Add an image** → Upload or Use camera  
        2) **Fill details** → Country / Denomination / Year / Mint / Grade  
        3) **Appraise** → Estimate + Range, then **Download PDF/CSV**  

        *Tips:* bright light, coin fills frame, steady shot. If not found, remove mint or try nearby year.
        """
    )
    st.divider()
    st.subheader("Price guide")
    pg_file = st.file_uploader("Upload price_guide.csv", type=["csv"], help="country,denomination,year,mint,price")

    # Inline CSV editor (optional): lets users tweak and download their guide
    with st.expander("Edit sample guide (optional)"):
        sample_df = pd.read_csv(io.StringIO(_DEFAULT_PG))
        edited = st.data_editor(sample_df, use_container_width=True, num_rows="dynamic")
        st.download_button("Download edited CSV", edited.to_csv(index=False).encode(), "price_guide.csv", "text/csv")

    # Settings persistence
    st.subheader("Defaults")
    if st.button("Save current inputs as default"):
        # will be filled below once inputs exist; here we just flag via session
        st.session_state["save_defaults_request"] = True
        st.toast("Will save after you appraise.")

appraiser = AppraisalEngine(csv_bytes=pg_file.read() if pg_file else None)
if appraiser.ready:
    st.sidebar.success("Price guide ready")
else:
    st.sidebar.warning("Using built-in sample guide")

# ---------------------------- UI: Main Tabs ----------------------------
tab_app, tab_help = st.tabs(["Use the app", "How it works"])

with tab_help:
    st.markdown(
        """
        ### What this app does
        • Grades the image with a simple visual heuristic (G → MS).  
        • Looks up a base price from your CSV and adjusts by grade.  
        • Exports a one-page PDF and a CSV of the result.  

        ### Roadmap to sell this
        1. Host on Streamlit Cloud / Spaces.  
        2. Add a login wall (email magic link) and a paywall (Stripe/LemonSqueezy).  
        3. Expand price data; add more countries/denoms.  
        4. Train a small identifier model for top coin types (optional).  
        """
    )

with tab_app:
    st.subheader("1) Add a coin image")
    src = st.radio("Source", ["Upload photo", "Use camera"], horizontal=True)

    image: Optional[Image.Image] = None
    if src == "Upload photo":
        up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            image = Image.open(up).convert("RGB")
            image = ImageOps.exif_transpose(image)
    else:
        shot = st.camera_input("Take a photo")
        if shot is not None:
            image = Image.open(shot).convert("RGB")
            image = ImageOps.exif_transpose(image)

    if image is not None:
        st.image(image, caption="Selected image", use_column_width=True)
        score, band, detail = grade_image(np.array(image))
        st.info(f"Suggested grade: **{band}** (score {score:.1f})")
    else:
        st.info("Add a photo to see a suggested grade.")

    st.subheader("2) Appraisal inputs")
    col1, col2 = st.columns(2)
    with col1:
        country = st.text_input("Country", value=st.session_state.get("country", load_settings().get("country")))
        denom = st.text_input("Denomination", value=st.session_state.get("denomination", load_settings().get("denomination")))
        year = st.text_input("Year (e.g., 1932)", value=st.session_state.get("year", load_settings().get("year")))
    with col2:
        mint = st.text_input("Mint (optional)", value=st.session_state.get("mint", load_settings().get("mint")))
        grade = st.selectbox("Grade", ["G", "VG", "F", "VF", "XF", "AU", "MS"], index=["G","VG","F","VF","XF","AU","MS"].index(load_settings().get("grade","XF")))
        conf_guess = st.slider("Confidence (range width)", 0.0, 1.0, float(load_settings().get("confidence", 0.7)), 0.05)

    # Demo presets — one-tap fill
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    if demo_col1.button("Demo: 1932-D Quarter"):
        country, denom, year, mint, grade = "united states", "quarter", "1932", "d", "XF"
        st.experimental_rerun()
    if demo_col2.button("Demo: 1909-S Cent"):
        country, denom, year, mint, grade = "united states", "lincoln cent", "1909", "s", "VF"
        st.experimental_rerun()
    if demo_col3.button("Demo: 1921 Morgan"):
        country, denom, year, mint, grade = "united states", "morgan dollar", "1921", "", "XF"
        st.experimental_rerun()

    # Appraise
    if st.button("Appraise", use_container_width=True):
        res = appraiser.appraise(
            country=country,
            denomination=denom,
            year=year,
            mint=mint,
            grade=grade,
            conf=conf_guess,
        )
        if res is None:
            st.error("Could not appraise. Check inputs or upload/adjust the price guide in the sidebar.")
        else:
            st.success(f"Estimate: ${res.estimate:,.2f} (range ${res.low:,.2f}–${res.high:,.2f})")
            st.caption(res.explanation)

            # Optional: save defaults after a successful run if user requested
            if st.session_state.get("save_defaults_request"):
                ok = save_settings({
                    "country": country,
                    "denomination": denom,
                    "year": year,
                    "mint": mint,
                    "grade": grade,
                    "confidence": conf_guess,
                })
                st.session_state.pop("save_defaults_request", None)
                st.toast("Defaults saved" if ok else "Could not save defaults", icon="💾" if ok else "⚠️")

            # Export PDF
            from reportlab.pdfgen import canvas as _canvas
            from reportlab.lib.pagesizes import letter
            buf = io.BytesIO()
            c = _canvas.Canvas(buf, pagesize=letter)
            w, h = letter
            c.setFont("Helvetica-Bold", 16); c.drawString(72, h-72, "Coin Appraisal Report")
            c.setFont("Helvetica", 10); y = h-100
            for k, v in [["Country", country],["Denomination", denom],["Year", year],["Mint", mint],["Grade", grade]]:
                c.drawString(72, y, f"{k}: {v}"); y -= 14
            c.drawString(72, y, f"Estimate: ${res.estimate:,.2f} (range ${res.low:,.2f}–${res.high:,.2f})")
            c.showPage(); c.save(); pdf_bytes = buf.getvalue()
            st.download_button("Download PDF", data=pdf_bytes, file_name="coin_appraisal.pdf", mime="application/pdf")
            st.download_button("Download CSV", data=res.to_csv().encode(), file_name="coin_appraisal.csv", mime="text/csv")
