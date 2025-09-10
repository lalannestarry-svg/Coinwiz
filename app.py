if not exact.empty:
   cand = exact
if cand.empty: return None. return cand.iloc[0].to_dict()
 def verdict(value):
    if value is None: return "No match", "Add this coin to price_guide.csv"
    if value >= 100: return "Good 💎", "Potentially valuable!"
    if value >= 10: return "Maybe 🤔", "Worth a closer look."
    return "Common 💤", "Low resale value."

# ---------------- UI ----------------
st.title(APP_NAME)

st.header("1) Add a coin photo")
src = st.radio("Source", ["Upload", "Camera"], horizontal=True)
image = None
if src == "Upload":
    up = st.file_uploader("Upload coin photo", type=["jpg","jpeg","png","webp"])
    if up: image = Image.open(up).convert("RGB")
else:
    shot = st.camera_input("Take a photo")
    if shot: image = Image.open(shot).convert("RGB")

year_guess, mint_guess = "", ""
if image is not None:
    image = ImageOps.exif_transpose(image)
    st.image(image, caption="Your coin", use_column_width=True)

    # OCR
    text = pytesseract.image_to_string(image)
    st.caption(f"OCR text detected: `{text}`")
    # crude guesses
    import re
    years = re.findall(r"(18|19|20)\d{2}", text)
    if years: year_guess = years[0]
    if "d" in text.lower(): mint_guess = "d"
    elif "s" in text.lower(): mint_guess = "s"
    elif "p" in text.lower(): mint_guess = "p"

st.header("2) Coin details")
col1, col2 = st.columns(2)
with col1:
    country = st.text_input("Country", "united states")
    denom   = st.text_input("Denomination", "penny")
    year    = st.text_input("Year", year_guess or "2022")
with col2:
    mint    = st.text_input("Mint (optional)", mint_guess)
    grade   = st.selectbox("Grade", ["G","VG","F","VF","XF","AU","MS"], index=4)

st.header("3) Appraise")
if st.button("Appraise", use_container_width=True):
    df = load_guide()
    if df.empty:
        st.error("No price_guide.csv found. Add one to your repo.")
    else:
        row = match_row(df, country, denom, year, mint, grade)
        if row is None:
            st.error("Could not find this coin. Add it to price_guide.csv.")
        else:
            value = float(row.get("price", 0))
            tag, note = verdict(value)
            st.success(f"Estimate: ${value:,.2f}")
            st.info(f"Verdict: **{tag}** — {note}")
