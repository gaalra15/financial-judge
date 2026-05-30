import io
import json
import os
import random
import re
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic

st.set_page_config(
    page_title="AI Financial Spending Judge",
    page_icon="💰",
    layout="wide"
)

# ── SOLARPUNK THEME ──────────────────────────────────────────────────────────

SOLARPUNK_DARK = {
    "bg_primary":       "#1A0F0A",
    "bg_secondary":     "#241208",
    "bg_card":          "#2E1A0E",
    "accent_primary":   "#C17F3E",
    "accent_secondary": "#D4956A",
    "accent_gold":      "#E8B86D",
    "accent_copper":    "#B5541F",
    "text_primary":     "#F5E6D3",
    "text_secondary":   "#D4956A",
    "text_muted":       "#8B6347",
    "border":           "#4A2E1A",
    "success":          "#8B9E4A",
    "warning":          "#E8B86D",
    "danger":           "#C04A2A",
    "chart_colors": [
        "#C17F3E", "#E8B86D", "#D4956A",
        "#8B9E4A", "#B5541F", "#F5E6D3",
        "#6B4423", "#A0522D",
    ],
}

SOLARPUNK_LIGHT = {
    "bg_primary":       "#FBF5ED",
    "bg_secondary":     "#F2E8D9",
    "bg_card":          "#FFFFFF",
    "accent_primary":   "#8B4513",
    "accent_secondary": "#C17F3E",
    "accent_gold":      "#B8860B",
    "accent_copper":    "#A0522D",
    "text_primary":     "#2C1810",
    "text_secondary":   "#6B3A2A",
    "text_muted":       "#8B6347",
    "border":           "#D4B896",
    "success":          "#5B7A3A",
    "warning":          "#B8860B",
    "danger":           "#8B2000",
    "chart_colors": [
        "#8B4513", "#C17F3E", "#B8860B",
        "#5B7A3A", "#A0522D", "#2C1810",
        "#D4956A", "#6B3A2A",
    ],
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

C = SOLARPUNK_DARK if st.session_state.theme == "dark" else SOLARPUNK_LIGHT


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_meme(url: str) -> bytes | None:
    """Fetch meme image bytes server-side so browser hotlink blocking can't interfere."""
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return r.content if r.ok else None
    except Exception:
        return None


def _show_meme(url: str, fallback: str, **kwargs):
    img = _fetch_meme(url) or _fetch_meme(fallback)
    if img:
        st.image(img, **kwargs)


def _gordon_card(idx: int):
    """Render a styled Gordon Ramsay text meme card — 100% reliable, no external images."""
    card = _GORDON_CARDS[idx % len(_GORDON_CARDS)]
    st.markdown(f"""
<div style="background:linear-gradient(145deg,#1a0000,#2d0505);
border:3px solid #cc2200; border-radius:14px; padding:28px 16px;
text-align:center; min-height:220px;
display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;">
    <div style="font-size:2.8rem">{card['emoji']}</div>
    <div style="font-family:'Impact','Arial Black',sans-serif; font-size:1.5rem;
    font-weight:900; color:#ff3300; text-transform:uppercase; letter-spacing:3px;
    text-shadow:2px 2px 4px rgba(0,0,0,0.8); line-height:1.1">{card['top']}</div>
    <div style="color:#f0d0d0; font-size:0.82rem; font-weight:600;
    text-transform:uppercase; letter-spacing:1px; margin-top:4px;
    max-width:200px; line-height:1.5">{card['caption']}</div>
</div>""", unsafe_allow_html=True)


def _sp_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor=C["bg_card"],
        paper_bgcolor=C["bg_secondary"],
        font=dict(color=C["text_primary"], family="Nunito, sans-serif", size=12),
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis=dict(gridcolor=C["border"], showgrid=True, color=C["text_muted"]),
        yaxis=dict(gridcolor=C["border"], showgrid=True, color=C["text_muted"]),
        colorway=C["chart_colors"],
    )
    base.update(kwargs)
    return base


def get_api_key() -> str:
    # Method 1: Streamlit secrets
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
        if key and key.startswith("sk-ant-"):
            print("[startup] Key loaded from: streamlit secrets")
            return key
    except Exception:
        pass  # Silent — try next method

    # Method 2: Environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key.startswith("sk-ant-"):
        print(f"[startup] Key loaded from: environment variable")
        print(f"[startup] Key starts with: '{key[:10]}'")
        return key

    # Method 3: .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key and key.startswith("sk-ant-"):
            print("[startup] Key loaded from: .env file")
            return key
    except ImportError:
        pass

    # All methods failed
    print("[startup] CRITICAL: No API key found in any location")
    st.error(
        "**No API key found.** Add your Anthropic API key:\n\n"
        "- **Local dev:** set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml`\n"
        "- **Streamlit Cloud:** add it in the app's Secrets settings\n"
        "- Get a key at: https://console.anthropic.com/settings/keys"
    )
    st.stop()
    return ""  # unreachable, satisfies type checker


def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_api_key())


def generate_sample_csv() -> bytes:
    """Return a sample bank statement CSV as bytes for the demo download button."""
    rows = [
        # Jan income & expenses
        ("2025-01-03", "Salary Deposit - Employer Inc", 5000.00, "income"),
        ("2025-01-05", "Whole Foods Market", 94.37, "expense"),
        ("2025-01-06", "Starbucks", 6.75, "expense"),
        ("2025-01-07", "Netflix", 15.99, "expense"),
        ("2025-01-08", "Shell Gas Station", 62.40, "expense"),
        ("2025-01-10", "Amazon Purchase", 48.99, "expense"),
        ("2025-01-12", "Chipotle", 13.45, "expense"),
        ("2025-01-14", "Spotify", 9.99, "expense"),
        ("2025-01-15", "Rent Payment", 1400.00, "expense"),
        ("2025-01-17", "Target", 87.23, "expense"),
        ("2025-01-19", "Uber", 18.50, "expense"),
        ("2025-01-20", "Trader Joe's", 72.11, "expense"),
        ("2025-01-22", "Gym Membership", 45.00, "expense"),
        ("2025-01-24", "McDonalds", 11.30, "expense"),
        ("2025-01-26", "Electric Bill", 95.00, "expense"),
        ("2025-01-28", "Zara", 129.00, "expense"),
        ("2025-01-30", "DoorDash", 34.67, "expense"),
        # Feb income & expenses
        ("2025-02-03", "Salary Deposit - Employer Inc", 5000.00, "income"),
        ("2025-02-05", "Whole Foods Market", 101.22, "expense"),
        ("2025-02-07", "Starbucks", 8.10, "expense"),
        ("2025-02-08", "Netflix", 15.99, "expense"),
        ("2025-02-09", "BP Gas Station", 58.90, "expense"),
        ("2025-02-10", "Amazon Purchase", 214.50, "expense"),
        ("2025-02-13", "Taco Bell", 9.85, "expense"),
        ("2025-02-14", "Cinema Ticket", 28.00, "expense"),
        ("2025-02-15", "Rent Payment", 1400.00, "expense"),
        ("2025-02-16", "H&M", 76.40, "expense"),
        ("2025-02-18", "Lyft", 22.75, "expense"),
        ("2025-02-20", "Trader Joe's", 65.88, "expense"),
        ("2025-02-22", "Internet Bill Comcast", 79.99, "expense"),
        ("2025-02-24", "Burger King", 14.20, "expense"),
        ("2025-02-25", "Spotify", 9.99, "expense"),
        ("2025-02-26", "Water Bill", 42.50, "expense"),
        ("2025-02-28", "Steam Game Purchase", 59.99, "expense"),
        # Mar income & expenses
        ("2025-03-03", "Salary Deposit - Employer Inc", 5000.00, "income"),
        ("2025-03-04", "Whole Foods Market", 88.66, "expense"),
        ("2025-03-06", "Starbucks", 7.25, "expense"),
        ("2025-03-07", "Netflix", 15.99, "expense"),
        ("2025-03-08", "Shell Gas Station", 71.00, "expense"),
        ("2025-03-10", "Nike Store", 155.00, "expense"),
        ("2025-03-12", "Chipotle", 15.80, "expense"),
        ("2025-03-14", "Spotify", 9.99, "expense"),
        ("2025-03-15", "Rent Payment", 1400.00, "expense"),
        ("2025-03-17", "Amazon Purchase", 93.45, "expense"),
        ("2025-03-18", "Uber", 31.20, "expense"),
        ("2025-03-20", "Trader Joe's", 79.33, "expense"),
        ("2025-03-22", "Gym Membership", 45.00, "expense"),
        ("2025-03-23", "Dominos Pizza", 27.99, "expense"),
        ("2025-03-25", "Electric Bill", 88.00, "expense"),
        ("2025-03-26", "Disney+", 13.99, "expense"),
        ("2025-03-28", "DoorDash", 41.25, "expense"),
        ("2025-03-29", "Transfer to Savings", 300.00, "expense"),
    ]
    df_sample = pd.DataFrame(rows, columns=["Date", "Description", "Amount", "Type"])
    buf = io.StringIO()
    df_sample.to_csv(buf, index=False)
    return buf.getvalue().encode()


CATEGORIES = {
    "restaurants":  ["restaurant", "bistro", "steakhouse", "diner", "tapas", "sushi", "pizza place",
                     "burger joint", "mcdonalds", "burger king", "kfc", "subway", "five guys",
                     "dominos", "papa johns", "taco bell", "wendys", "popeyes", "chipotle",
                     "starbucks", "coffee", "cafe", "dunkin", "espresso", "bakery", "deli",
                     "doordash", "uber eats", "glovo", "just eat", "deliveroo", "grubhub",
                     "rappi", "postmates", "bar", "pub", "nightclub", "cocktail", "brewery"],
    "food":         ["grocery", "supermarket", "supermercado", "mercadona", "carrefour", "lidl",
                     "aldi", "whole foods", "trader joe", "tesco", "sainsburys", "walmart",
                     "costco", "market", "bodega"],
    "streaming":    ["netflix", "spotify", "hbo", "disney", "apple music", "youtube premium",
                     "twitch", "prime video", "dazn", "hulu", "paramount", "peacock",
                     "apple tv", "crunchyroll", "mubi"],
    "transport":    ["uber", "lyft", "cabify", "bolt", "taxi", "bus", "metro", "train", "transit",
                     "fuel", "gas station", "gasolinera", "petrol", "shell", "bp", "chevron",
                     "exxon", "repsol", "cepsa", "galp", "parking", "toll", "airline", "flight",
                     "ryanair", "vueling", "iberia", "easyjet", "lufthansa", "delta", "airport",
                     "car rental", "hertz", "enterprise"],
    "shopping":     ["amazon", "ebay", "etsy", "zara", "h&m", "nike", "adidas", "fashion",
                     "clothing", "apparel", "best buy", "apple store", "ikea", "target",
                     "home depot", "shop", "store", "mall"],
    "entertainment": ["cinema", "movie", "theater", "concert", "steam", "xbox", "playstation",
                      "nintendo", "game", "museum", "gym", "fitness", "yoga", "sport"],
    "bills":        ["electric", "water bill", "gas bill", "internet", "broadband", "phone bill",
                     "insurance", "rent", "mortgage", "subscription", "verizon", "t-mobile",
                     "att", "comcast", "xfinity", "spectrum", "utility"],
}

GORDON_RAMSAY_MEMES = [
    "https://i.imgflip.com/1wkd.jpg",    # IDIOT SANDWICH — always intro
    "https://i.imgflip.com/rfjua.jpg",    # RAW BEEF — always roast_0
    "https://i.imgflip.com/2g7hl0.jpg",
    "https://i.imgflip.com/1zkxx.jpg",
    "https://i.imgflip.com/1wpq7.jpg",
    "https://i.imgflip.com/3lmzyx.jpg",
    "https://i.imgflip.com/22bdq6.jpg",
    "https://i.imgflip.com/1otk96.jpg",
    "https://i.imgflip.com/2hgfw.jpg",
]

# Text-based Gordon Ramsay "meme cards" used when image URLs fail or are off-theme.
# Each card maps to a specific slide position.
_GORDON_CARDS = [
    # 0 — intro
    {"emoji": "👨‍🍳", "top": "IDIOT SANDWICH", "caption": "Look at yourself. Now look at your bank statement."},
    # 1-4 — roast slides
    {"emoji": "🔥", "top": "IT'S RAW!",          "caption": "This budget is so raw it's still breathing."},
    {"emoji": "😤", "top": "YOU DONKEY!",         "caption": "You spent HOW MUCH on that?!"},
    {"emoji": "🚫", "top": "SHUT IT DOWN",        "caption": "SHUT. IT. DOWN. Right now."},
    {"emoji": "💀", "top": "ABSOLUTE DISASTER",   "caption": "I've seen better financial plans from a pigeon."},
    # 5 — score slide
    {"emoji": "🌟", "top": "FINALLY...",          "caption": "...something passable. Don't get cocky."},
]

MILD_MEMES = [
    "https://i.imgflip.com/1bgw.jpg",
    "https://i.imgflip.com/22bdq6.jpg",
    "https://i.imgflip.com/26am.jpg",
    "https://i.imgflip.com/3si4.jpg",
    "https://i.imgflip.com/2doow.jpg",
    "https://i.imgflip.com/1trl6p.jpg",
    "https://i.imgflip.com/1jwhww.jpg",
]

MEDIUM_MEMES = [
    "https://i.imgflip.com/1bij.jpg",
    "https://i.imgflip.com/1o00in.jpg",
    "https://i.imgflip.com/9ehk.jpg",
    "https://i.imgflip.com/1g8my4.jpg",
    "https://i.imgflip.com/3lmzyx.jpg",
    "https://i.imgflip.com/2fm6x.jpg",
    "https://i.imgflip.com/1yxkcp.jpg",
    "https://i.imgflip.com/30b1gx.jpg",
    "https://i.imgflip.com/1h7in3.jpg",
    "https://i.imgflip.com/1ur9b0.jpg",
    "https://i.imgflip.com/1w7iy4.jpg",
    "https://i.imgflip.com/26am.jpg",
]

MEME_LIBRARY = {
    "money": [
        "https://i.imgflip.com/1bij.jpg",
        "https://i.imgflip.com/26am.jpg",
        "https://i.imgflip.com/9ehk.jpg",
        "https://i.imgflip.com/1g8my4.jpg",
        "https://i.imgflip.com/3si4.jpg",
    ],
    "food": [
        "https://i.imgflip.com/1otk96.jpg",
        "https://i.imgflip.com/1h7in3.jpg",
        "https://i.imgflip.com/1ur9b0.jpg",
        "https://i.imgflip.com/2fm6x.jpg",
        "https://i.imgflip.com/1bgw.jpg",
    ],
    "shopping": [
        "https://i.imgflip.com/1o00in.jpg",
        "https://i.imgflip.com/30b1gx.jpg",
        "https://i.imgflip.com/1jwhww.jpg",
        "https://i.imgflip.com/3lmzyx.jpg",
        "https://i.imgflip.com/1trl6p.jpg",
    ],
    "bills": [
        "https://i.imgflip.com/2hgfw.jpg",
        "https://i.imgflip.com/1e7ql7.jpg",
        "https://i.imgflip.com/22bdq6.jpg",
        "https://i.imgflip.com/1yxkcp.jpg",
        "https://i.imgflip.com/2xkr.jpg",
    ],
    "savings": [
        "https://i.imgflip.com/1w7iy4.jpg",
        "https://i.imgflip.com/2doow.jpg",
        "https://i.imgflip.com/3vfmsn.jpg",
        "https://i.imgflip.com/1ihzfe.jpg",
        "https://i.imgflip.com/24y43o.jpg",
    ],
    "transport": [
        "https://i.imgflip.com/1jwhww.jpg",
        "https://i.imgflip.com/1otk96.jpg",
        "https://i.imgflip.com/3lmzyx.jpg",
        "https://i.imgflip.com/2fm6x.jpg",
        "https://i.imgflip.com/1bij.jpg",
    ],
    "entertainment": [
        "https://i.imgflip.com/1o00in.jpg",
        "https://i.imgflip.com/30b1gx.jpg",
        "https://i.imgflip.com/1trl6p.jpg",
        "https://i.imgflip.com/1yxkcp.jpg",
        "https://i.imgflip.com/1h7in3.jpg",
    ],
    "intro": [
        "https://i.imgflip.com/9ehk.jpg",
        "https://i.imgflip.com/1g8my4.jpg",
        "https://i.imgflip.com/3lmzyx.jpg",
        "https://i.imgflip.com/2fm6x.jpg",
    ],
    "score_bad": [
        "https://i.imgflip.com/1jwhww.jpg",
        "https://i.imgflip.com/2doow.jpg",
        "https://i.imgflip.com/1w7iy4.jpg",
        "https://i.imgflip.com/9ehk.jpg",
    ],
    "score_good": [
        "https://i.imgflip.com/1bgw.jpg",
        "https://i.imgflip.com/3si4.jpg",
        "https://i.imgflip.com/22bdq6.jpg",
        "https://i.imgflip.com/1otk96.jpg",
    ],
    "default": [
        "https://i.imgflip.com/1bij.jpg",
        "https://i.imgflip.com/26am.jpg",
        "https://i.imgflip.com/1o00in.jpg",
        "https://i.imgflip.com/3lmzyx.jpg",
        "https://i.imgflip.com/1g8my4.jpg",
    ],
}

CATEGORY_COLORS = {
    "restaurants":   "#FF9F43",
    "food":          "#FF6B6B",
    "streaming":     "#9B59B6",
    "transport":     "#4ECDC4",
    "shopping":      "#45B7D1",
    "entertainment": "#FED766",
    "bills":         "#97C475",
    "other":         "#C7B8EA",
}


def categorize_transaction(description: str) -> str:
    if not isinstance(description, str):
        return "other"
    desc_lower = description.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "other"


CSV_CATEGORY_MAP = {
    "food & drink":     "food",
    "groceries":        "food",
    "supermarket":      "food",
    "restaurants":      "restaurants",
    "fast food":        "restaurants",
    "coffee":           "restaurants",
    "utilities":        "bills",
    "rent":             "bills",
    "bills":            "bills",
    "subscription":     "bills",
    "entertainment":    "entertainment",
    "health & fitness": "entertainment",
    "sports":           "entertainment",
    "streaming":        "streaming",
    "shopping":         "shopping",
    "travel":           "transport",
    "transport":        "transport",
    "auto & transport": "transport",
    "salary":           "other",
    "investment":       "other",
    "transfer":         "other",
    "other":            "other",
}


def _read_file(f) -> pd.DataFrame:
    """Read CSV or Excel, trying multiple separators and encodings automatically."""
    name = f.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    raw_bytes = f.read()
    f.seek(0)
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, encoding=encoding)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(raw_bytes))


def preprocess_dataframe(df: pd.DataFrame, account_label: str = ""):
    desc_candidates = ["description", "transaction description", "merchant", "name",
                       "payee", "details", "memo", "narration", "transaction",
                       "concepto", "descripción", "descripcion", "movimiento",
                       "comercio", "beneficiario", "referencia", "observaciones"]
    amount_candidates = ["amount", "debit", "credit", "value", "sum", "charge", "withdrawal",
                         "importe", "cargo", "abono", "debe", "haber", "cantidad", "cuantía"]
    df.columns = df.columns.str.lower().str.strip()

    desc_col = None
    for c in desc_candidates:
        if c in df.columns:
            desc_col = c
            break
    if not desc_col:
        for col in df.columns:
            if df[col].dtype == object:
                desc_col = col
                break

    amount_col = None
    for c in amount_candidates:
        if c in df.columns:
            amount_col = c
            break
    if not amount_col:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break

    if not desc_col or not amount_col:
        return None

    # Detect date column
    date_col = None
    for col in df.columns:
        if col in (desc_col, amount_col):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                break
        except Exception:
            pass

    signed_amt = pd.to_numeric(df[amount_col], errors="coerce")
    if signed_amt.isna().mean() > 0.5:
        # Likely European format (1.234,56) — swap separators and retry
        signed_amt = pd.to_numeric(
            df[amount_col].astype(str)
                .str.replace(r"\.", "", regex=True)
                .str.replace(",", ".", regex=False),
            errors="coerce",
        )
    result = pd.DataFrame({
        "description": df[desc_col].astype(str),
        "amount": signed_amt.abs(),
        "signed_amount": signed_amt,
    })
    if date_col:
        result["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if account_label:
        result["account"] = account_label

    # Preserve Type column if present (used for income/expense split)
    if "type" in df.columns:
        result["type"] = df["type"].astype(str).str.strip().str.lower().values

    result = result.dropna(subset=["amount"])
    result = result[result["amount"] > 0]

    # Use CSV's own Category column if coverage is good; else keyword-match
    if "category" in df.columns:
        raw_cats = df.loc[result.index, "category"].astype(str).str.strip() if hasattr(result.index, '__iter__') else df["category"].astype(str).str.strip()
        # Reindex to match result after dropna/filter
        raw_cats = df["category"].astype(str).str.strip().iloc[result.index] if len(result) == len(df) else df["category"].astype(str).str.strip().reindex(result.index)
        coverage = raw_cats.notna().mean()
        if coverage >= 0.8:
            result["category"] = raw_cats.str.lower().map(CSV_CATEGORY_MAP).fillna("other").values
        else:
            result["category"] = result["description"].apply(categorize_transaction)
    else:
        result["category"] = result["description"].apply(categorize_transaction)

    return result


def detect_transfers(df: pd.DataFrame) -> set:
    if "date" not in df.columns or "account" not in df.columns:
        return set()
    if df["account"].nunique() < 2:
        return set()
    work = df[["amount", "date", "account"]].copy()
    work["_idx"] = range(len(work))
    work["_rounded"] = work["amount"].round(0)
    work = work.dropna(subset=["date"])
    merged = work.merge(work, on="_rounded", suffixes=("_a", "_b"))
    merged = merged[merged["account_a"] != merged["account_b"]]
    merged["_diff"] = (merged["date_a"] - merged["date_b"]).abs().dt.days
    matched = merged[merged["_diff"] <= 3]
    return set(matched["_idx_a"].tolist()) | set(matched["_idx_b"].tolist())


def calculate_health_score(df: pd.DataFrame):
    total = df["amount"].sum()
    if total == 0:
        return 50, {}
    cat_totals = df.groupby("category")["amount"].sum()
    grp_pcts = (cat_totals / total * 100).to_dict()
    score = 50

    food = grp_pcts.get("food", 0) + grp_pcts.get("restaurants", 0)
    if food < 15:
        score += 5
    elif food <= 30:
        score += 10
    elif food > 45:
        score -= 10

    bills = grp_pcts.get("bills", 0)
    if bills >= 20:
        score += 15
    elif bills >= 10:
        score += 8
    elif bills >= 5:
        score += 3

    ent = grp_pcts.get("entertainment", 0) + grp_pcts.get("streaming", 0)
    if ent <= 10:
        score += 15
    elif ent <= 15:
        score += 8
    elif ent > 25:
        score -= 15

    shop = grp_pcts.get("shopping", 0)
    if shop <= 15:
        score += 10
    elif shop > 25:
        score -= 15

    transport = grp_pcts.get("transport", 0)
    if transport <= 15:
        score += 5
    elif transport > 25:
        score -= 5

    return max(0, min(100, int(score))), grp_pcts


def get_score_color(score: int) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "orange"
    return "red"


def get_score_label(score: int) -> str:
    if score >= 80:
        return "Excellent 🌟"
    elif score >= 65:
        return "Good 👍"
    elif score >= 50:
        return "Fair 😐"
    elif score >= 35:
        return "Poor 😬"
    return "Critical 🚨"


def category_stats(df: pd.DataFrame) -> dict:
    work = df.copy()
    # try to find a date column for month-over-month and largest-txn date
    date_col = None
    for c in work.columns:
        if c not in ("description", "amount", "category"):
            try:
                parsed = pd.to_datetime(work[c], errors="coerce")
                if parsed.notna().sum() > len(work) * 0.5:
                    work["_date"] = parsed
                    date_col = c
                    break
            except Exception:
                pass

    has_date = date_col is not None
    if has_date:
        work["_month"] = work["_date"].dt.to_period("M")
        months = sorted(work["_month"].dropna().unique())
        num_months = max(len(months), 1)
        prev_month = months[-2] if len(months) >= 2 else None
        curr_month = months[-1] if months else None
        monthly_by_cat = work.groupby(["category", "_month"])["amount"].sum()
    else:
        num_months = 1
        prev_month = curr_month = None

    total = df["amount"].sum()
    cat_totals = df.groupby("category")["amount"].sum()
    stats = {}

    for cat in cat_totals.index:
        cat_df = work[work["category"] == cat]
        cat_total = cat_totals[cat]
        monthly_avg = cat_total / num_months

        # month-over-month
        prev_amt = curr_amt = None
        delta_pct = 0.0
        prev_label = curr_label = ""
        if has_date and prev_month and curr_month:
            curr_amt = monthly_by_cat.get((cat, curr_month), 0)
            prev_amt = monthly_by_cat.get((cat, prev_month), 0)
            if prev_amt and prev_amt != 0:
                delta_pct = (curr_amt - prev_amt) / prev_amt * 100
            prev_label = str(prev_month)
            curr_label = str(curr_month)

        # largest single transaction
        if has_date and "_date" in cat_df.columns:
            idx = cat_df["amount"].idxmax()
            largest_amt = cat_df.loc[idx, "amount"]
            largest_date = cat_df.loc[idx, "_date"]
            largest_date_str = pd.Timestamp(largest_date).strftime("%b %d") if pd.notna(largest_date) else "—"
        else:
            largest_amt = cat_df["amount"].max()
            largest_date_str = "—"

        stats[cat] = {
            "total": cat_total,
            "pct": cat_total / total * 100,
            "monthly_avg": monthly_avg,
            "prev_amt": prev_amt,
            "curr_amt": curr_amt,
            "delta_pct": delta_pct,
            "prev_label": prev_label,
            "curr_label": curr_label,
            "largest_amt": largest_amt,
            "largest_date": largest_date_str,
            "has_mom": has_date and prev_month is not None,
        }
    return stats


def compute_score_components(df: pd.DataFrame) -> dict:
    work = df.copy()
    # attach month period if possible (reuse logic from category_stats)
    for c in work.columns:
        if c not in ("description", "amount", "category"):
            try:
                parsed = pd.to_datetime(work[c], errors="coerce")
                if parsed.notna().sum() > len(work) * 0.5:
                    work["_month"] = parsed.dt.to_period("M")
                    break
            except Exception:
                pass

    has_month = "_month" in work.columns
    monthly_totals = work.groupby("_month")["amount"].sum() if has_month else None

    # Component 1 — Spending Consistency
    if has_month and len(monthly_totals) >= 2:
        ratio = monthly_totals.max() / monthly_totals.min()
        consistency_score = 25 if ratio < 1.5 else (15 if ratio < 2.0 else 5)
    else:
        consistency_score = 15  # neutral when no monthly data

    # Component 2 — Discretionary Control
    disc_cats = ["food", "restaurants", "shopping", "entertainment", "streaming", "transport"]
    disc_total = work[work["category"].isin(disc_cats)]["amount"].sum()
    disc_pct = disc_total / work["amount"].sum() if work["amount"].sum() > 0 else 0
    disc_score = 25 if disc_pct < 0.20 else (18 if disc_pct < 0.30 else 10)

    # Component 3 — Savings Signal
    savings_keywords = ["savings", "investment", "transfer", "brokerage",
                        "retirement", "401k", "ira", "vanguard", "fidelity"]
    has_savings = work["description"].str.lower().str.contains(
        "|".join(savings_keywords), na=False).any()
    savings_score = 25 if has_savings else 0

    # Component 4 — Month-over-Month Volatility
    if has_month and len(monthly_totals) >= 2:
        cv = monthly_totals.std() / monthly_totals.mean()
        volatility_score = 25 if cv < 0.15 else (18 if cv < 0.30 else 8)
    else:
        volatility_score = 15  # neutral when no monthly data

    total = consistency_score + disc_score + savings_score + volatility_score
    return {
        "consistency": consistency_score,
        "discretionary": disc_score,
        "savings": savings_score,
        "volatility": volatility_score,
        "total": total,
        "disc_pct": disc_pct,
    }


def get_whatif_lines(df: pd.DataFrame, components: dict, score: int) -> list:
    client = get_anthropic_client()
    total = df["amount"].sum()
    system = (
        "Return ONLY a JSON array of 3 strings, no markdown, no backticks. "
        "Each string is one sentence under 15 words starting with 'If you...'. "
        "Be specific with dollar amounts from the data."
    )
    prompt = (
        f"Total spending: ${total:,.0f}. "
        f"Health score: {score}/100. "
        f"Component scores — Consistency: {components['consistency']}/25, "
        f"Discretionary Control: {components['discretionary']}/25 "
        f"(discretionary is {components['disc_pct']*100:.1f}% of spend), "
        f"Savings Signal: {components['savings']}/25, "
        f"Volatility: {components['volatility']}/25. "
        "Generate 3 'what if' improvement scenarios."
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return parse_claude_json(response.content[0].text)
    except Exception:
        return []


def get_budget_recommendations(
    monthly_income: float,
    budget_inputs: dict,
    cat_monthly_avgs: dict,
    currency: str,
) -> dict:
    client = get_anthropic_client()
    system = (
        "You are a financial advisor. "
        f"The user has a monthly income of {currency}{monthly_income:,.0f}. "
        f"Their current average monthly spending per category is: {cat_monthly_avgs}. "
        f"Their self-set monthly budget targets are: {budget_inputs}. "
        "Return ONLY a valid JSON object, no markdown, no backticks. "
        'Format: {"summary":"...","allocations":[{"category":"...","current_monthly_avg":0,'
        '"user_budget":0,"suggested_budget":0,"suggested_pct":0,"reasoning":"...","status":"over|under|on_track"}],'
        '"tips":[{"icon":"...","tip":"..."}]}'
    )
    prompt = (
        f"Monthly income: {currency}{monthly_income:,.0f}. "
        f"Current monthly averages by category: {cat_monthly_avgs}. "
        f"User budget targets: {budget_inputs}. "
        "Produce a full budget recommendation with summary, per-category allocations, and 4 tips. "
        "suggested_pct is the suggested_budget as a percentage of monthly income."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_claude_json(response.content[0].text)


def render_budget_recommendations(data: dict, currency: str, monthly_income: float):
    summary = data.get("summary", "")
    allocations = data.get("allocations", [])
    tips = data.get("tips", [])

    # Summary banner
    st.markdown(f"""
<div style="background:{C['bg_secondary']}; border:1px solid {C['accent_primary']};
border-radius:12px; padding:16px 20px; margin:12px 0; color:{C['text_secondary']}; font-size:1rem;">
    📋 <strong style="color:{C['text_primary']}">Summary:</strong> {summary}
</div>""", unsafe_allow_html=True)

    # Allocation rows
    st.markdown("#### 📊 Category Allocations")
    for item in allocations:
        status = item.get("status", "on_track")
        status_color = C["danger"] if status == "over" else C["success"] if status == "under" else C["accent_primary"]
        status_icon = "🔴" if status == "over" else "🟢" if status == "under" else "🔵"
        status_label = "Over budget" if status == "over" else "Under budget" if status == "under" else "On track"
        current = item.get("current_monthly_avg", 0)
        user_budget = item.get("user_budget", 0)
        suggested = item.get("suggested_budget", 0)
        suggested_pct = item.get("suggested_pct", 0)
        st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:10px; padding:14px; margin:6px 0;
border-left:4px solid {status_color}">
    <div style="display:flex; justify-content:space-between">
        <strong style="color:{C['text_primary']}">{item.get('category','')}</strong>
        <span style="color:{status_color}">{status_icon} {status_label}</span>
    </div>
    <div style="color:{C['text_muted']}; font-size:0.8rem; margin-top:6px">
        Current avg: {currency}{current:,.0f}/mo
        &nbsp;·&nbsp; Your target: {currency}{user_budget:,.0f}
        &nbsp;·&nbsp; AI suggests: {currency}{suggested:,.0f} ({suggested_pct}% of income)
    </div>
    <div style="color:{C['text_secondary']}; font-size:0.82rem; margin-top:4px; font-style:italic">
        {item.get('reasoning','')}
    </div>
</div>""", unsafe_allow_html=True)

    # Tips in 2-column cards
    if tips:
        st.markdown("#### 💡 Budget Tips")
        tip_colors = C["chart_colors"]
        for i in range(0, len(tips), 2):
            pair = tips[i:i+2]
            tip_cols = st.columns(len(pair))
            for col, item, color in zip(tip_cols, pair, tip_colors[i:i+2]):
                with col:
                    st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:12px; padding:16px; margin:8px 0;
border-top:3px solid {color}; display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.5rem">{item.get('icon','💡')}</span>
    <span style="color:{C['text_primary']}; font-size:0.95rem">{item.get('tip','')}</span>
</div>""", unsafe_allow_html=True)

    # Donut chart of suggested allocation vs income
    if allocations and monthly_income > 0:
        st.markdown("#### 🥧 Suggested Budget Allocation")
        chart_cats = [a["category"] for a in allocations]
        chart_vals = [a.get("suggested_budget", 0) for a in allocations]
        allocated = sum(chart_vals)
        unallocated = max(monthly_income - allocated, 0)
        if unallocated > 0:
            chart_cats.append("Savings / Unallocated")
            chart_vals.append(unallocated)

        budget_colors = {
            **{k.capitalize(): v for k, v in CATEGORY_COLORS.items()},
            "Savings / Unallocated": "#7FDBFF",
        }
        fig = px.pie(
            values=chart_vals,
            names=chart_cats,
            color=chart_cats,
            color_discrete_map=budget_colors,
            hole=0.5,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            showlegend=True,
            margin=dict(t=20, b=0, l=0, r=0),
            paper_bgcolor=C["bg_card"],
            plot_bgcolor=C["bg_card"],
            font_color=C["text_primary"],
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_1")


def build_roast_context(df: pd.DataFrame) -> dict:
    # Top 3 merchants by total spend
    top_merchants = df.groupby("description")["amount"].sum().nlargest(3)
    top3 = [{"merchant": m, "total": a} for m, a in top_merchants.items()]

    # Late night transactions (10pm–6am) — requires a parseable date column
    late_night = []
    largest_date = "unknown date"
    for c in df.columns:
        if c in ("description", "amount", "category"):
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                hours = parsed.dt.hour
                mask = (hours >= 22) | (hours < 6)
                ln_rows = df[mask][["description", "amount"]].head(5).to_dict("records")
                late_night = ln_rows
                # largest transaction date
                idx = df["amount"].idxmax()
                ts = parsed.loc[idx]
                if pd.notna(ts):
                    largest_date = pd.Timestamp(ts).strftime("%b %d, %Y")
                break
        except Exception:
            pass

    # Impulse buys: single transaction > 3x category average
    _rgrp = "category_group" if "category_group" in df.columns else "category"
    impulse = []
    for cat, group in df.groupby(_rgrp):
        avg = group["amount"].mean()
        for _, row in group[group["amount"] > avg * 3].iterrows():
            impulse.append({
                "category": cat,
                "description": row["description"],
                "amount": row["amount"],
                "avg": avg,
            })
    impulse = sorted(impulse, key=lambda x: x["amount"], reverse=True)[:3]

    # Largest single transaction
    idx = df["amount"].idxmax()
    largest = {
        "description": df.loc[idx, "description"],
        "amount": df.loc[idx, "amount"],
        "date": largest_date,
    }
    return {"top_merchants": top3, "late_night": late_night, "impulse": impulse, "largest": largest}


def get_roast(df: pd.DataFrame, score: int, roast_level: str) -> dict:
    client = get_anthropic_client()
    context = build_roast_context(df)

    if "Mild" in roast_level:
        tone = ("Be gently funny, like a supportive friend pointing out bad habits. "
                "Warm but honest.")
    elif "Medium" in roast_level:
        tone = ("Be a sharp, witty comedian roasting their finances. "
                "Funny but not cruel.")
    else:
        tone = (
            "You are Gordon Ramsay reviewing someone's finances instead of food. "
            "Channel his exact TV personality. "
            "Use his catchphrases adapted to finance: "
            "'This budget is so raw it's still breathing', "
            "'You DONKEY, you spent HOW MUCH on coffee?', "
            "'This is a financial DISASTER', "
            "'Shut it down. Shut. It. Down.' "
            "Be loud, specific, and brutally funny. "
            "Reference actual dollar amounts from the data. "
            "Use CAPS occasionally for emphasis like he shouts. "
            "End with one reluctant compliment like Gordon always does — put it in backhanded_compliment."
        )

    system = (
        f"{tone} "
        "Return ONLY valid JSON, no markdown, no backticks. "
        "Schema: "
        '{"opening_line":"one brutal opener line",'
        '"roasts":["first roast — specific with amounts","second roast — different topic",'
        '"third roast — different topic","fourth roast — different topic"],'
        '"backhanded_compliment":"one thing they do well, said backhanded",'
        '"final_score":5,'
        '"score_label":"exactly three funny words"}'
    )

    top3_str = ", ".join(
        f"{m['merchant']} (${m['total']:,.0f})" for m in context["top_merchants"]
    )
    late_str = (
        f"{len(context['late_night'])} late-night transactions detected"
        if context["late_night"] else "no late-night transactions detected"
    )
    impulse_str = (
        ", ".join(
            f"{i['description']} ${i['amount']:,.0f} (category avg ${i['avg']:,.0f})"
            for i in context["impulse"]
        ) if context["impulse"] else "none detected"
    )
    largest = context["largest"]

    prompt = (
        f"Health score: {score}/100. "
        f"Top 3 merchants by total spend: {top3_str}. "
        f"Late-night spending (10pm-6am): {late_str}. "
        f"Impulse purchases (>3x category avg): {impulse_str}. "
        f"Largest single transaction: {largest['description']} "
        f"${largest['amount']:,.0f} on {largest['date']}. "
        "Generate 4 distinct roasts covering different aspects of their spending. "
        "final_score is 1-10 (be fair). "
        "score_label is exactly 3 funny words describing their financial style."
    )

    roast_model = "claude-opus-4-7" if "Gordon" in roast_level else "claude-sonnet-4-6"
    response = client.messages.create(
        model=roast_model,
        max_tokens=1500,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_claude_json(response.content[0].text)


_GORDON_IDIOT_SANDWICH = "https://i.imgflip.com/1wkd.jpg"
_GORDON_RAW_BEEF = "https://i.imgflip.com/rfjua.jpg"


def pick_meme(roast_text: str, slide_type: str = "default") -> str:
    if "Gordon" in st.session_state.get("roast_level", ""):
        return random.choice(GORDON_RAMSAY_MEMES)
    text_lower = roast_text.lower()
    if any(w in text_lower for w in ["food", "eat", "restaurant", "coffee", "groceries", "delivery"]):
        category = "food"
    elif any(w in text_lower for w in ["shop", "amazon", "buy", "purchase", "spend", "clothes"]):
        category = "shopping"
    elif any(w in text_lower for w in ["bill", "subscription", "netflix", "spotify", "utility", "rent"]):
        category = "bills"
    elif any(w in text_lower for w in ["sav", "invest", "wealth", "future", "retire"]):
        category = "savings"
    elif any(w in text_lower for w in ["transport", "uber", "taxi", "car", "fuel", "gas", "travel"]):
        category = "transport"
    elif any(w in text_lower for w in ["entertain", "fun", "leisure", "game", "movie", "concert"]):
        category = "entertainment"
    elif any(w in text_lower for w in ["money", "cash", "dollar", "budget", "bank"]):
        category = "money"
    elif slide_type == "intro":
        category = "intro"
    else:
        category = "default"
    return random.choice(MEME_LIBRARY.get(category, MEME_LIBRARY["default"]))


def pick_score_meme(score: int) -> str:
    key = "score_good" if score >= 6 else "score_bad"
    return random.choice(MEME_LIBRARY[key])


def assign_memes(roast_data: dict) -> dict:
    # Read from both possible keys to be safe
    roast_level = (
        st.session_state.get("roast_level")
        or st.session_state.get("roast_level_slider")
        or "Medium 🔥"
    )

    is_gordon = "Gordon" in roast_level or "Ramsay" in roast_level
    is_mild   = "Mild" in roast_level

    print(f"[assign_memes] roast_level='{roast_level}' is_gordon={is_gordon} is_mild={is_mild}")

    memes: dict = {}
    num_roasts = len(roast_data.get("roasts", []))

    if is_gordon:
        # Guaranteed pins
        memes["intro"]   = GORDON_RAMSAY_MEMES[0]   # idiot sandwich
        memes["roast_0"] = GORDON_RAMSAY_MEMES[1]   # raw beef

        remaining = GORDON_RAMSAY_MEMES[2:]
        shuffled  = remaining.copy()
        random.shuffle(shuffled)
        for i in range(1, num_roasts):
            memes[f"roast_{i}"] = shuffled[(i - 1) % len(shuffled)]

        score = int(roast_data.get("final_score", 5))
        memes["score"] = GORDON_RAMSAY_MEMES[1] if score <= 4 else GORDON_RAMSAY_MEMES[0]

    elif is_mild:
        pool = MILD_MEMES.copy()
        random.shuffle(pool)
        memes["intro"] = pool[0]
        for i in range(num_roasts):
            memes[f"roast_{i}"] = pool[(i + 1) % len(pool)]
        memes["score"] = pool[-1]

    else:  # Medium
        pool = MEDIUM_MEMES.copy()
        random.shuffle(pool)
        memes["intro"] = pool[0]
        for i in range(num_roasts):
            memes[f"roast_{i}"] = pool[(i + 1) % len(pool)]
        memes["score"] = pool[-1]

    print(f"[assign_memes] result={memes}")
    return memes


def _render_roast_slideshow(roast: dict, roast_level: str):
    _FALLBACK = "https://i.imgflip.com/1bij.jpg"
    memes = st.session_state.get("roast_memes", {})

    slides = [{"type": "intro"}]
    for text in roast.get("roasts", []):
        slides.append({"type": "roast", "text": text})
    slides.append({"type": "score"})
    total = len(slides)

    slide_idx = st.session_state.roast_slide
    current = slides[slide_idx]
    is_gordon = "Gordon" in roast_level

    # --- Render slide ---
    if current["type"] == "intro":
        intro_meme = memes.get("intro", _FALLBACK)
        col_main, col_img = st.columns([3, 1])
        with col_main:
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:20px; padding:60px 40px; text-align:center;
border:2px solid {C['danger']}; min-height:300px">
    <div style="font-size:3rem; margin-bottom:16px">🔥</div>
    <div style="color:{C['danger']}; font-size:2.5rem; font-weight:900;
    letter-spacing:4px; text-transform:uppercase">ROAST IS ON</div>
    <div style="color:{C['text_muted']}; margin-top:16px; font-size:1rem">Intensity: {roast_level}</div>
    <div style="color:{C['text_muted']}; margin-top:8px; font-size:0.9rem; font-style:italic">
        "{roast.get('opening_line','')}"
    </div>
</div>""", unsafe_allow_html=True)
        with col_img:
            if is_gordon:
                _gordon_card(0)
            else:
                _show_meme(intro_meme, _FALLBACK, use_container_width=True)

    elif current["type"] == "roast":
        roast_idx = slide_idx - 1
        meme_key = f"roast_{roast_idx}"
        meme_url = memes.get(meme_key, _FALLBACK)
        col_text, col_meme = st.columns([3, 2])
        with col_text:
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:16px; padding:32px; min-height:280px;
border-left:5px solid {C['danger']}; display:flex; flex-direction:column; justify-content:center">
    <div style="color:{C['danger']}; font-size:0.8rem; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:16px">🔥 Roast #{slide_idx}</div>
    <div style="color:{C['text_primary']}; font-size:1.15rem; line-height:1.7">{current['text']}</div>
</div>""", unsafe_allow_html=True)
        with col_meme:
            if is_gordon:
                _gordon_card(1 + roast_idx)
            else:
                _show_meme(meme_url, _FALLBACK, use_container_width=True)

    elif current["type"] == "score":
        final_score = int(roast.get("final_score", 5))
        score_color = C["success"] if final_score >= 8 else C["warning"] if final_score >= 5 else C["danger"]
        score_meme = memes.get("score", _FALLBACK)
        st.markdown(f"""
<div style="background:{C['bg_secondary']}; border-radius:20px; padding:50px 40px; text-align:center;
border:2px solid {score_color}; min-height:300px">
    <div style="color:{C['text_muted']}; font-size:0.85rem; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:16px">Final Verdict</div>
    <div style="color:{score_color}; font-size:5rem; font-weight:900; line-height:1">
        {final_score}/10
    </div>
    <div style="color:{C['text_primary']}; font-size:1.4rem; font-weight:bold; margin:16px 0">
        "{roast.get('score_label','')}"
    </div>
    <div style="background:{C['bg_card']}; border-radius:10px; padding:16px; margin-top:20px;
    color:{C['text_secondary']}; font-style:italic; font-size:0.95rem">
        💚 {roast.get('backhanded_compliment','')}
    </div>
</div>""", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            if is_gordon:
                _gordon_card(5)
            else:
                _show_meme(score_meme, _FALLBACK, use_container_width=True)
        if final_score >= 8:
            st.balloons()
        elif final_score <= 3:
            st.snow()

    # --- Navigation ---
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([1, 3, 1])

    with nav1:
        if slide_idx > 0:
            if st.button("◀ Back", use_container_width=True, key="roast_back"):
                st.session_state.roast_slide -= 1
                st.rerun()

    with nav2:
        dots = "".join("🔴 " if i == slide_idx else "⚪ " for i in range(total))
        st.markdown(
            f"<div style='text-align:center; font-size:0.8rem'>"
            f"{dots}<br><span style='color:#666'>{slide_idx + 1} of {total}</span></div>",
            unsafe_allow_html=True,
        )

    with nav3:
        if slide_idx < total - 1:
            if st.button("Next ▶", use_container_width=True, type="primary", key="roast_next"):
                st.session_state.roast_slide += 1
                st.rerun()


def parse_claude_json(response_text: str):
    clean = re.sub(r'```json|```', '', response_text).strip()

    # Normal parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Truncated response recovery — find the last complete JSON object
    try:
        last_brace = clean.rfind('},')
        if last_brace == -1:
            last_brace = clean.rfind('}')
        if last_brace > 0:
            recovered = clean[:last_brace + 1] + ']'
            result = json.loads(recovered)
            st.warning(
                f"⚠️ Partial categorization recovered: {len(result)} transactions. "
                "Remaining will use keyword matching."
            )
            return result
    except json.JSONDecodeError:
        pass

    # Full fallback — return empty list so keyword matching takes over
    st.error("Claude categorization failed entirely — using keyword matching as fallback.")
    return []


def _call(client: anthropic.Anthropic, system: str, prompt: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _build_insights_context(df: pd.DataFrame, score: int) -> dict:
    """Pre-calculate rich transaction context to pass to Claude."""
    context: dict = {
        "category_totals": df.groupby("category")["amount"].sum().round(2).to_dict(),
        "total_expenses": round(float(df["amount"].sum()), 2),
        "health_score": score,
        "date_range": None,
        "worst_months": [],
        "month_over_month": [],
        "top_transactions": [],
        "recurring_patterns": [],
        "biggest_transaction": None,
    }

    has_date = "date" in df.columns and pd.to_datetime(df["date"], errors="coerce").notna().sum() > len(df) * 0.5

    if has_date:
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])
        work["month_str"] = work["date"].dt.strftime("%B %Y")
        work["month_period"] = work["date"].dt.to_period("M")

        cat_monthly = (
            work.groupby(["month_str", "month_period", "category"])["amount"]
            .sum().reset_index()
        )

        # Most expensive month per category
        cat_worst = cat_monthly.loc[
            cat_monthly.groupby("category")["amount"].idxmax()
        ][["category", "month_str", "amount"]].rename(
            columns={"month_str": "worst_month", "amount": "worst_amount"}
        )
        context["worst_months"] = cat_worst.round(2).to_dict("records")

        # Month-over-month change for last 2 months
        sorted_months = sorted(work["month_period"].unique())
        if len(sorted_months) >= 2:
            last_2 = sorted_months[-2:]
            prev_df = cat_monthly[cat_monthly["month_period"] == last_2[0]][["category", "amount"]].rename(columns={"amount": "prev"})
            curr_df = cat_monthly[cat_monthly["month_period"] == last_2[1]][["category", "amount"]].rename(columns={"amount": "curr"})
            mom = prev_df.merge(curr_df, on="category")
            mom["change_pct"] = ((mom["curr"] - mom["prev"]) / mom["prev"] * 100).round(1)
            mom["prev_month_name"] = last_2[0].strftime("%B %Y")
            mom["curr_month_name"] = last_2[1].strftime("%B %Y")
            context["month_over_month"] = mom.round(2).to_dict("records")

        # Top 3 transactions per category
        top_txns = (
            work.sort_values("amount", ascending=False)
            .groupby("category").head(3)
            [["category", "description", "amount", "date"]].copy()
        )
        top_txns["date_str"] = top_txns["date"].dt.strftime("%b %d %Y")
        context["top_transactions"] = top_txns.drop(columns=["date"]).round(2).to_dict("records")

        # Recurring patterns: same category + rounded amount appearing in 3+ months
        work["amount_rounded"] = (work["amount"] / 25).round() * 25
        recur = (
            work.groupby(["category", "amount_rounded"])
            .agg(count=("amount", "count"),
                 months=("month_str", lambda x: list(x.unique())[:4]),
                 avg_amount=("amount", "mean"))
            .reset_index()
        )
        context["recurring_patterns"] = (
            recur[recur["count"] >= 3]
            .sort_values("count", ascending=False)
            .head(5).round(2).to_dict("records")
        )

        # Biggest single transaction
        idx = work["amount"].idxmax()
        bt = work.loc[idx]
        context["biggest_transaction"] = {
            "amount": round(float(bt["amount"]), 2),
            "category": bt["category"],
            "date": bt["date"].strftime("%B %d %Y"),
            "description": str(bt["description"])[:50],
        }
        context["date_range"] = {
            "start": work["date"].min().strftime("%B %Y"),
            "end": work["date"].max().strftime("%B %Y"),
            "months_total": int(work["month_period"].nunique()),
        }
    else:
        # No date column — include top transactions and biggest without dates
        top_txns = (
            df.sort_values("amount", ascending=False)
            .groupby("category").head(3)
            [["category", "description", "amount"]].copy()
        )
        context["top_transactions"] = top_txns.round(2).to_dict("records")
        idx = df["amount"].idxmax()
        bt = df.loc[idx]
        context["biggest_transaction"] = {
            "amount": round(float(bt["amount"]), 2),
            "category": bt["category"],
            "date": "unknown",
            "description": str(bt["description"])[:50],
        }

    return context


def get_insights(df: pd.DataFrame, cat_pcts: dict, score: int) -> dict:
    client = get_anthropic_client()
    total = df["amount"].sum()
    context = _build_insights_context(df, score)
    has_dates = context["date_range"] is not None

    # --- Call 1: Key Stats ---
    stats_system = (
        "Return ONLY a valid JSON array, no other text, no markdown, no backticks. "
        "Format: ["
        '{"icon":"💰","label":"Total Spending","value":"$X","sublabel":"across all transactions"},'
        '{"icon":"📊","label":"Biggest Category","value":"Category X%","sublabel":"$X"},'
        '{"icon":"🏥","label":"Health Score","value":"X/100","sublabel":"label"},'
        '{"icon":"⚡","label":"Avg per Category","value":"$X","sublabel":"mean spend"}'
        "] Use the actual numbers from the data."
    )
    stats_prompt = (
        f"Total: ${total:.2f}. Health score: {score}/100 ({get_score_label(score)}). "
        f"Category totals: {context['category_totals']}."
    )

    # --- Call 2: Observations ---
    date_note = (
        "Month-by-month data is available — you MUST reference specific month names and compare months."
        if has_dates else
        "No date data is available — focus on category totals and specific transaction amounts instead of months."
    )
    obs_system = (
        "You are a sharp financial analyst with access to detailed transaction data. "
        "Generate exactly 4 observations.\n\n"
        f"{date_note}\n\n"
        "CRITICAL RULES:\n"
        "- Every observation MUST include specific dollar amounts from the data\n"
        "- Reference actual merchant names or transaction descriptions when relevant\n"
        "- If month data is available: compare specific months by name, note the worst month per category\n"
        "- If there are recurring patterns, mention which months they appeared\n"
        "- Mention the biggest transaction if notable\n"
        "- Be specific: NOT 'Entertainment is high' "
        "BUT 'Entertainment hit $X in [month], up $Y from the prior month'\n\n"
        "Return ONLY a JSON array, no markdown, no backticks:\n"
        '[{"type":"warning" or "good",'
        '"text":"specific observation with dollar amounts",'
        '"detail":"one extra sentence with more specifics — months, merchants, exact amounts"}]'
    )
    obs_prompt = f"Analyze this financial data and give specific observations: {json.dumps(context)}"

    # --- Call 3: Action Tips ---
    tips_system = (
        "You are a financial coach giving specific, data-backed advice. "
        "Generate exactly 4 action tips.\n\n"
        f"{date_note}\n\n"
        "CRITICAL RULES:\n"
        "- Each tip must include exact dollar amounts from the data\n"
        "- If month data is available: reference specific months and time periods\n"
        "- If there are recurring patterns, list the specific months they occurred\n"
        "- Reference the biggest transactions specifically\n"
        "- Give concrete next steps, not generic advice\n"
        "- NOT: 'Consider reducing food spending' "
        "BUT: 'Food spiked to $X in [month] — $Y above your $Z monthly average. "
        "Check if [pattern] repeats.'\n\n"
        "Return ONLY a JSON array, no markdown, no backticks:\n"
        '[{"icon":"one emoji","color":"hex color like #2ECC71 or #E74C3C or #3498DB or #9B59B6",'
        '"tip":"specific tip under 20 words with dollar amounts",'
        '"detail":"one sentence with the specific pattern — months or transactions involved"}]'
    )
    tips_prompt = f"Give specific action tips based on this data: {json.dumps(context)}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_stats = pool.submit(_call, client, stats_system, stats_prompt)
        f_obs   = pool.submit(_call, client, obs_system,   obs_prompt)
        f_tips  = pool.submit(_call, client, tips_system,  tips_prompt)

    stats        = parse_claude_json(f_stats.result())
    observations = parse_claude_json(f_obs.result())
    tips         = parse_claude_json(f_tips.result())

    return {"stats": stats, "observations": observations, "tips": tips}


def render_insights(data: dict):
    stats = data.get("stats", [])
    observations = data.get("observations", [])
    tips = data.get("tips", [])

    # Key Stats row
    st.markdown("#### 📈 Key Stats")
    for row_start in range(0, len(stats), 4):
        row_items = stats[row_start:row_start + 4]
        row_cols = st.columns(len(row_items))
        for col, item in zip(row_cols, row_items):
            with col:
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:12px; padding:20px; text-align:center;
border:1px solid {C['border']}; border-top:3px solid {C['accent_primary']}">
    <div style="font-size:2rem">{item['icon']}</div>
    <div style="color:{C['text_muted']}; font-size:0.8rem; margin-top:8px">{item['label']}</div>
    <div style="color:{C['text_primary']}; font-size:1.6rem; font-weight:bold; margin:4px 0">{item['value']}</div>
    <div style="color:{C['text_muted']}; font-size:0.75rem">{item['sublabel']}</div>
</div>""", unsafe_allow_html=True)

    # Observations — with detail sub-line
    st.markdown("#### 🔍 Key Observations")
    for item in observations:
        color = C["success"] if item.get("type") == "good" else C["danger"]
        icon = "✅" if item.get("type") == "good" else "⚠️"
        detail = item.get("detail", "")
        st.markdown(f"""
<div style="border-left:4px solid {color}; padding:12px 16px; margin:10px 0;
background:{C['bg_card']}; border-radius:0 10px 10px 0;">
    <div style="color:{C['text_primary']}; font-size:0.95rem; font-weight:500">{icon} {item['text']}</div>
    <div style="color:{C['text_muted']}; font-size:0.82rem; margin-top:6px; padding-top:6px;
    border-top:1px solid {C['border']}">{detail}</div>
</div>""", unsafe_allow_html=True)

    # Action Tips — 2-column grid with detail sub-line
    st.markdown("#### 💡 Action Tips")
    tip_cols = st.columns(2)
    for i, item in enumerate(tips):
        with tip_cols[i % 2]:
            detail = item.get("detail", "")
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:12px; padding:16px; margin:8px 0;
border-top:3px solid {item['color']};">
    <div style="display:flex; align-items:flex-start; gap:12px">
        <span style="font-size:1.4rem">{item['icon']}</span>
        <div>
            <div style="color:{C['text_primary']}; font-size:0.92rem; font-weight:500">{item['tip']}</div>
            <div style="color:{C['text_muted']}; font-size:0.8rem; margin-top:6px">{detail}</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)


def _compute_findings(monthly_totals: pd.Series, cat_monthly: pd.DataFrame, work: pd.DataFrame) -> list:
    findings = []
    if len(monthly_totals) < 2:
        return findings

    vals = monthly_totals.values
    dates = [d.strftime("%b %Y") for d in monthly_totals.index]
    mean_spend = float(monthly_totals.mean())

    # Spike: biggest month > 1.4x mean
    peak_idx = int(np.argmax(vals))
    if vals[peak_idx] > mean_spend * 1.4:
        findings.append({
            "type": "spike", "severity": "high",
            "badge": "📈 SPENDING SPIKE",
            "title": f"{dates[peak_idx]} was your highest month — {vals[peak_idx]/mean_spend:.1f}× your average",
            "body": (f"${vals[peak_idx]:,.0f} spent vs your ${mean_spend:,.0f} monthly average. "
                     "Dig into what drove that month."),
            "sparkline_data": vals.tolist(),
        })

    # Best month: lowest month < 0.7x mean
    low_idx = int(np.argmin(vals))
    if vals[low_idx] < mean_spend * 0.7:
        findings.append({
            "type": "drop", "severity": "medium", "positive": True,
            "badge": "🟢 BEST MONTH",
            "title": f"{dates[low_idx]} was your leanest month — only {vals[low_idx]/mean_spend:.1f}× your average",
            "body": f"${vals[low_idx]:,.0f} spent. Can you replicate what you did that month?",
        })

    # Upward streak: 3+ consecutive months of rising spend
    for i in range(len(vals) - 2):
        if vals[i] < vals[i + 1] < vals[i + 2]:
            findings.append({
                "type": "streak_up", "severity": "high",
                "badge": "🔴 UPWARD STREAK",
                "title": f"Spending rose 3+ months in a row: {dates[i]} → {dates[i+2]}",
                "body": (f"${vals[i]:,.0f} → ${vals[i+1]:,.0f} → ${vals[i+2]:,.0f}. "
                         f"That's a ${vals[i+2]-vals[i]:,.0f} cumulative increase."),
                "sparkline_data": vals.tolist(),
            })
            break

    # Downward streak: 3+ consecutive months of falling spend
    for i in range(len(vals) - 2):
        if vals[i] > vals[i + 1] > vals[i + 2]:
            findings.append({
                "type": "streak_down", "severity": "medium", "positive": True,
                "badge": "🟢 DOWNWARD STREAK",
                "title": f"Spending fell 3+ months in a row: {dates[i]} → {dates[i+2]}",
                "body": (f"${vals[i]:,.0f} → ${vals[i+1]:,.0f} → ${vals[i+2]:,.0f}. Great discipline!"),
            })
            break

    # Category dominance: one category > 40% of total
    total_all = float(work["amount"].sum())
    if total_all > 0:
        cat_totals = work.groupby("category")["amount"].sum()
        for cat, ctotal in cat_totals.items():
            if ctotal / total_all > 0.40:
                spark = cat_monthly[cat].values.tolist() if cat in cat_monthly.columns else []
                findings.append({
                    "type": "dominance", "severity": "high",
                    "badge": "⚠️ CATEGORY DOMINANCE",
                    "title": f"{cat.capitalize()} takes up {ctotal/total_all*100:.0f}% of your total spending",
                    "body": (f"${ctotal:,.0f} of ${total_all:,.0f} total went to {cat}. "
                             "A single category above 40% crowds out savings and other goals."),
                    "sparkline_data": spark,
                })
                break

    # Big outlier: transaction > 5x category average
    cat_means = work.groupby("category")["amount"].mean()
    for cat, grp in work.groupby("category"):
        avg = cat_means[cat]
        outliers = grp[grp["amount"] > avg * 5]
        if not outliers.empty:
            row = outliers.nlargest(1, "amount").iloc[0]
            findings.append({
                "type": "outlier", "severity": "medium",
                "badge": "🎯 BIG OUTLIER",
                "title": f"${row['amount']:,.0f} on {cat} — {row['amount']/avg:.1f}× your average",
                "body": f"'{str(row['description'])[:45]}' was far above your usual ${avg:,.0f} {cat} spend.",
            })
            break

    return findings


def render_deep_analytics(df: pd.DataFrame):
    # ── Setup ──────────────────────────────────────────────────────────────
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])

    has_dates = "date" in work.columns and len(work) > 0

    no_date_msg = "Add a date column to your file to unlock time series analysis."

    # ── Without dates: only show impulse flags ─────────────────────────────
    if not has_dates:
        st.info(no_date_msg)
        st.markdown("#### ⚡ Impulse Buy Flags")
        _render_impulse_flags(df, {})
        return

    work["month"] = work["date"].dt.to_period("M")
    work["month_str"] = work["date"].dt.strftime("%b %Y")
    work["year"] = work["date"].dt.year

    monthly_totals = work.groupby("month")["amount"].sum()
    monthly_totals.index = monthly_totals.index.to_timestamp()

    cat_monthly = (
        work.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0)
    )
    cat_monthly.index = cat_monthly.index.to_timestamp()

    # ── AUTO-DETECTED INSIGHTS ─────────────────────────────────────────────
    st.markdown("#### 🔍 Auto-Detected Insights")
    findings = _compute_findings(monthly_totals, cat_monthly, work)

    if findings:
        high_finds = [f for f in findings if f["severity"] == "high"]
        med_finds  = [f for f in findings if f["severity"] == "medium"]

        for _spark_i, f in enumerate(high_finds):
            fc1, fc2 = st.columns([3, 1])
            with fc1:
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:12px; padding:20px;
border-left:4px solid {C['accent_gold']}; margin:8px 0">
    <div style="color:{C['accent_gold']}; font-size:0.72rem; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:6px">{f['badge']}</div>
    <div style="color:{C['text_primary']}; font-size:1.05rem; font-weight:600">{f['title']}</div>
    <div style="color:{C['text_muted']}; font-size:0.86rem; margin-top:6px">{f['body']}</div>
</div>""", unsafe_allow_html=True)
            with fc2:
                spark = f.get("sparkline_data", [])
                if spark:
                    fig_s = go.Figure(go.Scatter(
                        y=spark, mode="lines",
                        line=dict(color=C["accent_gold"], width=2),
                        fill="tozeroy",
                        fillcolor=hex_to_rgba(C["accent_gold"], 0.18),
                    ))
                    fig_s.update_layout(
                        height=80, margin=dict(l=0, r=0, t=4, b=0),
                        plot_bgcolor=C["bg_card"], paper_bgcolor=C["bg_card"],
                        showlegend=False,
                        xaxis=dict(visible=False), yaxis=dict(visible=False),
                    )
                    st.plotly_chart(fig_s, use_container_width=True, key=f"chart_sparkline_{_spark_i}")

        if med_finds:
            mc1, mc2 = st.columns(2)
            for i, f in enumerate(med_finds):
                border = C["success"] if f.get("positive") else C["accent_copper"]
                with (mc1 if i % 2 == 0 else mc2):
                    st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:10px; padding:14px;
border-top:3px solid {border}; margin:8px 0">
    <div style="color:{border}; font-size:0.7rem; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:4px">{f['badge']}</div>
    <div style="color:{C['text_primary']}; font-size:0.92rem; font-weight:500">{f['title']}</div>
    <div style="color:{C['text_muted']}; font-size:0.82rem; margin-top:4px">{f['body']}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Not enough months to detect patterns yet (need 2+).")

    st.divider()

    # ── CHART 1: Monthly Expenses + Rolling Avg + Trend ────────────────────
    st.markdown("#### 📈 Monthly Spending Trend")
    mts = monthly_totals.reset_index()
    mts.columns = ["date", "amount"]
    mts = mts.sort_values("date")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=mts["date"], y=mts["amount"],
        name="Monthly Spend",
        marker_color=hex_to_rgba(C["accent_primary"], 0.8),
    ))
    if len(mts) >= 3:
        rolling = mts["amount"].rolling(3, min_periods=1).mean()
        fig1.add_trace(go.Scatter(
            x=mts["date"], y=rolling,
            name="3-Month Avg", mode="lines",
            line=dict(color=C["accent_gold"], width=2, dash="dot"),
        ))
        x_n = np.arange(len(mts))
        z = np.polyfit(x_n, mts["amount"].values, 1)
        trend = np.poly1d(z)(x_n)
        fig1.add_trace(go.Scatter(
            x=mts["date"], y=trend,
            name="Trend", mode="lines",
            line=dict(color=C["danger"], width=1.5),
        ))
    peak_i = int(mts["amount"].idxmax())
    fig1.add_annotation(
        x=mts.loc[peak_i, "date"], y=mts.loc[peak_i, "amount"],
        text=f"Peak: ${mts.loc[peak_i, 'amount']:,.0f}",
        showarrow=True, arrowhead=2,
        font=dict(color=C["accent_gold"]),
        arrowcolor=C["accent_gold"],
        bgcolor=C["bg_secondary"],
        bordercolor=C["border"],
    )
    fig1.update_layout(**_sp_layout(
        title="Total Monthly Expenses",
        xaxis_title="Month", yaxis_title="Amount ($)",
        legend=dict(bgcolor=C["bg_secondary"], bordercolor=C["border"]),
    ))
    st.plotly_chart(fig1, use_container_width=True, key="chart_2")

    st.divider()

    # ── CHART 2: Stacked Area by Category ─────────────────────────────────
    st.markdown("#### 🏔 Spending by Category Over Time")
    if not cat_monthly.empty:
        fig2 = go.Figure()
        for i, cat in enumerate(cat_monthly.columns):
            color = C["chart_colors"][i % len(C["chart_colors"])]
            fig2.add_trace(go.Scatter(
                x=cat_monthly.index, y=cat_monthly[cat],
                name=cat.capitalize(), stackgroup="one",
                mode="lines", line=dict(width=0.5, color=color),
                fillcolor=hex_to_rgba(color, 0.67),
            ))
        fig2.update_layout(**_sp_layout(
            title="Category Spending Over Time (Stacked)",
            xaxis_title="Month", yaxis_title="Amount ($)",
            legend=dict(bgcolor=C["bg_secondary"], bordercolor=C["border"]),
        ))
        st.plotly_chart(fig2, use_container_width=True, key="chart_3")

    st.divider()

    # ── CHART 3: Year-Over-Year ────────────────────────────────────────────
    years = sorted(work["year"].unique())
    if len(years) >= 2:
        st.markdown("#### 📅 Year-Over-Year Comparison")
        work["month_of_year"] = work["date"].dt.month
        yoy = work.groupby(["year", "month_of_year"])["amount"].sum().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig3 = go.Figure()
        for i, yr in enumerate(years):
            yd = yoy[yoy["year"] == yr].sort_values("month_of_year")
            color = C["chart_colors"][i % len(C["chart_colors"])]
            fig3.add_trace(go.Scatter(
                x=yd["month_of_year"].map(lambda m: month_names[m - 1]),
                y=yd["amount"], name=str(yr),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ))
        fig3.update_layout(**_sp_layout(
            title="Year-Over-Year Monthly Spending",
            xaxis_title="Month", yaxis_title="Amount ($)",
            legend=dict(bgcolor=C["bg_secondary"], bordercolor=C["border"]),
        ))
        st.plotly_chart(fig3, use_container_width=True, key="chart_4")
        st.divider()

    # ── CHART 4: Small Multiples ───────────────────────────────────────────
    st.markdown("#### 🔢 Category Trends — Small Multiples")
    all_cats = sorted(work["category"].unique().tolist())
    selected_cats = st.multiselect(
        "Select categories to display",
        options=all_cats,
        default=all_cats[:min(6, len(all_cats))],
        key="analytics_cat_filter",
    )
    if selected_cats:
        n_cols = 2
        n_rows = (len(selected_cats) + n_cols - 1) // n_cols
        fig4 = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[c.capitalize() for c in selected_cats],
        )
        for i, cat in enumerate(selected_cats):
            row, col = i // n_cols + 1, i % n_cols + 1
            cd = (
                work[work["category"] == cat]
                .groupby("month")["amount"].sum()
            )
            cd.index = cd.index.to_timestamp()
            cd = cd.sort_index().reset_index()
            cd.columns = ["date", "amount"]
            color = C["chart_colors"][i % len(C["chart_colors"])]
            fig4.add_trace(go.Scatter(
                x=cd["date"], y=cd["amount"],
                mode="lines+markers", name=cat.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=5), showlegend=False,
            ), row=row, col=col)
            if len(cd) >= 3:
                x_n = np.arange(len(cd))
                z = np.polyfit(x_n, cd["amount"].values, 1)
                fig4.add_trace(go.Scatter(
                    x=cd["date"], y=np.poly1d(z)(x_n),
                    mode="lines", showlegend=False,
                    line=dict(color=C["accent_gold"], width=1, dash="dot"),
                ), row=row, col=col)
        fig4.update_layout(
            height=300 * n_rows,
            plot_bgcolor=C["bg_card"], paper_bgcolor=C["bg_secondary"],
            font=dict(color=C["text_primary"], family="Nunito, sans-serif"),
        )
        fig4.update_xaxes(gridcolor=C["border"], showgrid=True, color=C["text_muted"])
        fig4.update_yaxes(gridcolor=C["border"], showgrid=True, color=C["text_muted"])
        fig4.update_annotations(font_color=C["text_secondary"])
        st.plotly_chart(fig4, use_container_width=True, key="chart_5")

    st.divider()

    # ── CHARTS 5 + 6: Quarterly Heatmap | Normalized % Stacked ───────────
    col_heat, col_norm = st.columns(2)

    with col_heat:
        st.markdown("##### 🌡 Quarterly Heatmap")
        work["quarter_label"] = (
            work["date"].dt.year.astype(str) + " Q"
            + work["date"].dt.quarter.astype(str)
        )
        heat_data = (
            work.groupby(["quarter_label", "category"])["amount"]
            .sum().unstack(fill_value=0)
        )
        if not heat_data.empty:
            fig5 = px.imshow(
                heat_data,
                color_continuous_scale=[C["bg_card"], C["accent_primary"]],
                aspect="auto", title="Spend by Quarter & Category",
            )
            fig5.update_layout(
                plot_bgcolor=C["bg_card"], paper_bgcolor=C["bg_secondary"],
                font=dict(color=C["text_primary"]),
                height=350, margin=dict(l=20, r=20, t=50, b=40),
                coloraxis_colorbar=dict(tickcolor=C["text_muted"], title="$"),
            )
            st.plotly_chart(fig5, use_container_width=True, key="chart_6")

    with col_norm:
        st.markdown("##### 📊 Normalized Category Mix (%)")
        if not cat_monthly.empty:
            cat_norm = cat_monthly.div(cat_monthly.sum(axis=1), axis=0) * 100
            fig6 = go.Figure()
            for i, cat in enumerate(cat_norm.columns):
                color = C["chart_colors"][i % len(C["chart_colors"])]
                fig6.add_trace(go.Scatter(
                    x=cat_norm.index, y=cat_norm[cat],
                    name=cat.capitalize(), stackgroup="one",
                    mode="lines", line=dict(width=0),
                    fillcolor=hex_to_rgba(color, 0.73),
                ))
            fig6.update_layout(**_sp_layout(
                title="Category Mix Over Time (%)",
                yaxis_title="% of Monthly Spend", height=350,
                legend=dict(bgcolor=C["bg_secondary"], bordercolor=C["border"]),
            ))
            st.plotly_chart(fig6, use_container_width=True, key="chart_7")

    st.divider()

    # ── SUBSCRIPTION DETECTOR ──────────────────────────────────────────────
    st.markdown("#### 🔄 Subscription Detector")
    work["month_period"] = work["date"].dt.to_period("M")
    work["amount_rounded"] = work["amount"].round(0)
    recurring = []
    for (desc, amt), grp in work.groupby(["description", "amount_rounded"]):
        months = grp["month_period"].nunique()
        if months >= 2:
            recurring.append({
                "Merchant": desc,
                "Amount": f"${amt:,.2f}",
                "Frequency": f"{months}x",
                "Annual Cost": f"${amt * 12:,.0f}",
                "_annual_raw": amt * 12,
            })
    if recurring:
        rec_df = pd.DataFrame(recurring).sort_values("_annual_raw", ascending=False)
        total_annual = rec_df["_annual_raw"].sum()
        st.dataframe(rec_df[["Merchant", "Amount", "Frequency", "Annual Cost"]],
                     use_container_width=True, hide_index=True)
        st.markdown(f"""
<div style="background:{C['bg_secondary']}; border-radius:8px; padding:12px 16px;
margin-top:8px; color:{C['text_primary']}; border:1px solid {C['border']}">
    🔄 <strong>Total recurring spend:</strong>
    <span style="color:{C['danger']}; font-size:1.1rem; font-weight:bold"> ${total_annual:,.0f}/year</span>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No recurring transactions detected.")

    st.divider()

    # ── IMPULSE BUY FLAGS ──────────────────────────────────────────────────
    st.markdown("#### ⚡ Impulse Buy Flags")
    date_map = work.set_index(work.index)["date"].dt.strftime("%b %d, %Y").to_dict()
    _render_impulse_flags(df, date_map)


def _render_impulse_flags(df: pd.DataFrame, date_map: dict):
    impulse_rows = []
    for cat, grp in df.groupby("category"):
        cat_mean = grp["amount"].mean()
        flagged = grp[grp["amount"] > cat_mean * 3].copy()
        flagged["_multiple"] = (flagged["amount"] / cat_mean).round(1)
        flagged["_category"] = cat
        impulse_rows.append(flagged)
    if impulse_rows:
        all_flags = pd.concat(impulse_rows).sort_values("amount", ascending=False)
        if all_flags.empty:
            st.info("No impulse purchases detected — impressive discipline!")
        else:
            for idx, row in all_flags.iterrows():
                date_str = date_map.get(idx, "")
                date_part = f" — {date_str}" if date_str else ""
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-left:4px solid {C['warning']};
border-radius:0 10px 10px 0; padding:12px 16px; margin:6px 0;">
    ⚠️ <strong style="color:{C['text_primary']}">{row['description']}</strong>
    <span style="color:{C['danger']}; font-weight:bold"> — ${row['amount']:,.2f}</span>
    <span style="color:{C['text_muted']}">{date_part} — {row['_multiple']}x your usual {row['_category']} spend</span>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No impulse purchases detected.")


def build_chat_context(expense_df: pd.DataFrame, income_df: pd.DataFrame) -> dict:
    df = expense_df.copy()
    if "date" not in df.columns or df["date"].isna().all():
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    df["month_str"] = df["date"].dt.strftime("%B %Y")
    df["day_name"] = df["date"].dt.day_name()
    df["day_of_month"] = df["date"].dt.day

    # Top 10 transactions with context vs category average
    _cat_means = df.groupby("category")["amount"].mean()
    top10 = df.nlargest(10, "amount")[["date", "description", "category", "amount"]].copy()
    top10["date"] = top10["date"].dt.strftime("%B %d %Y")
    top10["vs_cat_avg"] = top10.apply(
        lambda r: round(r["amount"] / _cat_means.get(r["category"], r["amount"]), 1), axis=1
    )

    # Per-category deep stats
    cat_stats = {}
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        monthly = cat_df.groupby("month_str")["amount"].sum()
        best_idx = cat_df["amount"].idxmax()
        cat_stats[cat] = {
            "total": round(float(cat_df["amount"].sum()), 2),
            "monthly_avg": round(float(monthly.mean()), 2),
            "monthly_max": round(float(monthly.max()), 2),
            "monthly_max_month": str(monthly.idxmax()),
            "monthly_min": round(float(monthly.min()), 2),
            "monthly_min_month": str(monthly.idxmin()),
            "transaction_count": int(len(cat_df)),
            "avg_transaction": round(float(cat_df["amount"].mean()), 2),
            "biggest_transaction": {
                "amount": round(float(cat_df.loc[best_idx, "amount"]), 2),
                "date": cat_df.loc[best_idx, "date"].strftime("%B %d %Y"),
                "description": str(cat_df.loc[best_idx, "description"])[:60],
                "day_of_week": cat_df.loc[best_idx, "day_name"],
            },
            "pct_of_total": round(
                float(cat_df["amount"].sum()) / float(df["amount"].sum()) * 100, 1
            ),
        }

    # Monthly overview
    monthly_totals = df.groupby("month_str")["amount"].sum()
    last_vs_prev = (
        round(
            (monthly_totals.iloc[-1] - monthly_totals.iloc[-2])
            / monthly_totals.iloc[-2] * 100, 1
        )
        if len(monthly_totals) > 1 else 0
    )
    monthly_overview = {
        "most_expensive_month": {
            "month": str(monthly_totals.idxmax()),
            "amount": round(float(monthly_totals.max()), 2),
            "vs_average": round(float(monthly_totals.max()) / float(monthly_totals.mean()), 1),
        },
        "cheapest_month": {
            "month": str(monthly_totals.idxmin()),
            "amount": round(float(monthly_totals.min()), 2),
        },
        "monthly_average": round(float(monthly_totals.mean()), 2),
        "monthly_trend": (
            "increasing" if monthly_totals.iloc[-1] > monthly_totals.iloc[0] else "decreasing"
        ),
        "last_month": {
            "month": str(monthly_totals.index[-1]),
            "amount": round(float(monthly_totals.iloc[-1]), 2),
            "vs_previous": last_vs_prev,
        },
    }

    # Day of week spend
    dow_spend = df.groupby("day_name")["amount"].agg(["sum", "mean", "count"])
    most_expensive_day = str(dow_spend["sum"].idxmax())

    # Impulse transactions (≥2.5× category mean)
    df["cat_mean"] = df["category"].map(_cat_means)
    df["impulse_ratio"] = df["amount"] / df["cat_mean"]
    impulse = df[df["impulse_ratio"] >= 2.5].nlargest(5, "impulse_ratio")[
        ["date", "description", "category", "amount", "impulse_ratio"]
    ].copy()
    impulse["date"] = impulse["date"].dt.strftime("%B %d %Y")
    impulse["impulse_ratio"] = impulse["impulse_ratio"].round(1)

    # Recurring patterns
    df["amount_bucket"] = (df["amount"] / 25).round() * 25
    recurring = (
        df.groupby(["category", "amount_bucket"])
        .agg(
            count=("amount", "count"),
            months=("month_str", lambda x: list(x.unique())[:6]),
            avg=("amount", "mean"),
        )
        .reset_index()
    )
    recurring = (
        recurring[recurring["count"] >= 3]
        .sort_values("count", ascending=False)
        .head(8)
    )
    recurring["avg"] = recurring["avg"].round(2)

    # Income / savings
    income_total = float(income_df["amount"].sum()) if income_df is not None and len(income_df) > 0 else 0
    expense_total = float(df["amount"].sum())
    savings_rate = (
        round((income_total - expense_total) / income_total * 100, 1)
        if income_total > 0 else 0
    )

    # Peak 7-day rolling spend
    peak_week_end = None
    peak_week_amount = 0.0
    try:
        df_idx = df.sort_values("date").set_index("date")
        rolling = df_idx["amount"].rolling("7D").sum()
        peak_week_end = rolling.idxmax().strftime("%B %d %Y")
        peak_week_amount = round(float(rolling.max()), 2)
    except Exception:
        pass

    return {
        "summary": {
            "total_expenses": round(expense_total, 2),
            "total_income": round(income_total, 2),
            "savings_rate": savings_rate,
            "total_transactions": int(len(df)),
            "date_range_start": df["date"].min().strftime("%B %d %Y"),
            "date_range_end": df["date"].max().strftime("%B %d %Y"),
            "months_analyzed": int(df["month_str"].nunique()),
            "most_expensive_day_of_week": most_expensive_day,
            "peak_spending_week": {"week_ending": peak_week_end, "total": peak_week_amount},
        },
        "top_10_transactions": top10.to_dict("records"),
        "category_stats": cat_stats,
        "monthly_overview": monthly_overview,
        "impulse_transactions": impulse.to_dict("records"),
        "recurring_patterns": recurring.to_dict("records"),
    }


def render_finance_chat(df: pd.DataFrame):
    st.divider()
    st.subheader("💬 Ask Your Finances")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    ctx = st.session_state.get("chat_context", {})

    system_prompt = (
        "You are a sharp personal finance analyst with complete access to the user's transaction data.\n\n"
        "CRITICAL RULES FOR EVERY ANSWER:\n"
        "- Always mention specific dates (e.g. 'March 14 2023')\n"
        "- Always mention specific dollar amounts\n"
        "- Always compare to averages: 'this is X times your usual spend'\n"
        "- Always mention the category context\n"
        "- If relevant, mention the day of the week\n"
        "- If relevant, mention how it compares to previous months\n"
        "- Keep answers to 3-5 sentences MAX\n"
        "- Be conversational but data-precise\n"
        "- Never say 'based on the data provided' — just answer directly\n"
        "- If the question is about the biggest expense, mention: the amount, the date, "
        "the day of week, the category, and how much higher it is than the category average\n"
        "- If the question is about a category, mention: the monthly average, "
        "the best month, the worst month, and the single biggest transaction in that category\n\n"
        "GOOD ANSWER EXAMPLE:\n"
        "'Your biggest expense was $4,996 on Travel on September 3 2022 (a Saturday) — "
        "that's 3.8x your average Travel transaction of $1,315. Your most expensive month "
        "overall was August 2021 at $18,420, which was 1.6x your monthly average of $11,450.'\n\n"
        "BAD ANSWER EXAMPLE:\n"
        "'Your biggest expense was in the Travel category which tends to be high for many people.'\n\n"
        f"Complete financial data:\n{json.dumps(ctx, indent=2)}"
    )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Styled message history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
<div style="background:{C['bg_secondary']}; border-radius:12px 12px 4px 12px;
padding:12px 16px; margin:8px 0 8px 20%; border:1px solid {C['border']}">
    <div style="color:{C['text_muted']}; font-size:0.75rem; margin-bottom:4px">You</div>
    <div style="color:{C['text_primary']}">{msg['content']}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:12px 12px 12px 4px;
padding:12px 16px; margin:8px 20% 8px 0;
border:1px solid {hex_to_rgba(C['accent_primary'], 0.25)}; border-left:3px solid {C['accent_primary']}">
    <div style="color:{C['accent_primary']}; font-size:0.75rem; margin-bottom:4px">AI Judge</div>
    <div style="color:{C['text_primary']}; line-height:1.6">{msg['content']}</div>
</div>""", unsafe_allow_html=True)

    # Suggestion chips — shown only when history is empty
    if not st.session_state.chat_history:
        suggestions = [
            "What was my single biggest expense?",
            "Which month did I overspend the most?",
            "What are my most suspicious transactions?",
            "Which category has the most consistent spending?",
            "What was my worst week of spending?",
            "How does last month compare to my average?",
            "What recurring charges am I paying?",
            "Which day of the week do I spend the most?",
        ]
        st.markdown("**Quick questions:**")
        chip_cols = st.columns(4)
        for i, suggestion in enumerate(suggestions):
            with chip_cols[i % 4]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    st.rerun()

    # If last message is from user (suggestion click or rerun), generate answer
    if (st.session_state.chat_history
            and st.session_state.chat_history[-1]["role"] == "user"):
        with st.spinner("Thinking..."):
            try:
                client = get_anthropic_client()
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=600,
                    system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ],
                )
                answer = response.content[0].text
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Could not get answer: {str(e)}")

    # Chat input for typed questions
    user_question = st.chat_input("Ask anything about your spending...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.rerun()


# --- UI ---

# ── SIDEBAR THEME TOGGLE ──────────────────────────────────────────────────
with st.sidebar:
    _tc1, _tc2 = st.columns([1, 3])
    with _tc1:
        st.markdown("🌙" if st.session_state.theme == "dark" else "☀️")
    with _tc2:
        _toggle_label = "Switch to Light" if st.session_state.theme == "dark" else "Switch to Dark"
        if st.button(_toggle_label, key="theme_toggle"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

# ── GLOBAL CSS ────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Nunito:wght@300;400;500;600&family=Orbitron:wght@400;500;700&display=swap');


.stApp {{
    background-color: {C['bg_primary']} !important;
    font-family: 'Nunito', sans-serif;
}}
.main .block-container {{
    background-color: {C['bg_primary']};
    padding: 2rem 3rem;
    max-width: 1200px;
}}
section[data-testid="stSidebar"] {{
    background: {C['bg_secondary']} !important;
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] * {{
    color: {C['text_primary']} !important;
}}

h1 {{
    font-family: 'Cinzel', serif !important;
    color: {C['accent_primary']} !important;
    font-size: 2.2rem !important;
    letter-spacing: 2px !important;
    border-bottom: 2px solid {C['accent_gold']};
    padding-bottom: 8px;
}}
h2 {{
    font-family: 'Cinzel', serif !important;
    color: {C['accent_primary']} !important;
    font-size: 1.6rem !important;
    letter-spacing: 1px !important;
}}
h3 {{
    font-family: 'Nunito', sans-serif !important;
    color: {C['text_secondary']} !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}}
p, li, span, div {{
    color: {C['text_primary']};
    font-family: 'Nunito', sans-serif;
}}
.stat-number {{
    font-family: 'Orbitron', monospace !important;
    color: {C['accent_primary']} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {C['bg_secondary']};
    border-radius: 12px;
    padding: 4px;
    border: 1px solid {C['border']};
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Nunito', sans-serif;
    font-weight: 600;
    color: {C['text_muted']} !important;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: {C['accent_primary']} !important;
    color: {C['bg_primary']} !important;
}}

.stButton button {{
    background: {C['accent_primary']} !important;
    color: {C['bg_primary']} !important;
    border: 1px solid {C['accent_gold']} !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
}}
.stButton button:hover {{
    background: {C['accent_gold']} !important;
    color: {C['bg_primary']} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px {hex_to_rgba(C['accent_gold'], 0.25)} !important;
}}

.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stTextArea textarea {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text_primary']} !important;
    font-family: 'Nunito', sans-serif !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {C['accent_primary']} !important;
    box-shadow: 0 0 0 2px {hex_to_rgba(C['accent_primary'], 0.18)} !important;
}}

[data-testid="stFileUploader"] {{
    background: {C['bg_primary']} !important;
    border: 2px dashed {C['accent_primary']} !important;
    border-radius: 12px !important;
}}
[data-testid="stFileUploader"] > div {{
    background: {C['bg_primary']} !important;
}}
[data-testid="stFileUploader"] section {{
    background: {C['bg_primary']} !important;
}}
[data-testid="stFileUploader"] section > div {{
    background: {C['bg_primary']} !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {C['bg_primary']} !important;
    border: none !important;
}}
[data-testid="stFileUploaderFile"] {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {{
    color: {C['text_muted']} !important;
}}
.streamlit-expanderHeader {{
    background: {C['bg_card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text_primary']} !important;
    font-family: 'Nunito', sans-serif !important;
}}

[data-testid="metric-container"] {{
    background: {C['bg_card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 16px;
    border-top: 3px solid {C['accent_primary']};
}}
[data-testid="metric-container"] label {{
    color: {C['text_muted']} !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-family: 'Orbitron', monospace !important;
    color: {C['accent_primary']} !important;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
}}
hr {{ border-color: {C['border']} !important; opacity: 0.5; }}

.stSuccess {{
    background: {hex_to_rgba(C['success'], 0.12)} !important;
    border-left: 4px solid {C['success']} !important;
    border-radius: 0 8px 8px 0 !important;
}}
.stWarning {{
    background: {hex_to_rgba(C['warning'], 0.12)} !important;
    border-left: 4px solid {C['warning']} !important;
    border-radius: 0 8px 8px 0 !important;
}}
.stError {{
    background: {hex_to_rgba(C['danger'], 0.12)} !important;
    border-left: 4px solid {C['danger']} !important;
    border-radius: 0 8px 8px 0 !important;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['bg_secondary']}; }}
::-webkit-scrollbar-thumb {{ background: {C['accent_primary']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C['accent_gold']}; }}

.stSpinner > div {{ border-top-color: {C['accent_gold']} !important; }}
.stProgress > div > div {{
    background: linear-gradient(90deg, {C['accent_primary']}, {C['accent_gold']}) !important;
}}
[data-baseweb="tag"] {{
    background: {C['accent_primary']} !important;
    border-radius: 6px !important;
}}
</style>

""", unsafe_allow_html=True)

# ── STYLED HEADER ─────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:24px 0 8px;
border-bottom:1px solid {C['border']}; margin-bottom:24px">
    <div style="font-family:'Orbitron',monospace; font-size:0.75rem;
    color:{C['accent_gold']}; letter-spacing:4px; text-transform:uppercase; margin-bottom:8px">
        ◆ Personal Finance Intelligence ◆
    </div>
    <div style="font-family:'Cinzel',serif; font-size:2.4rem; font-weight:700;
    color:{C['accent_primary']}; letter-spacing:3px">
        AI Financial Judge
    </div>
    <div style="font-family:'Nunito',sans-serif; color:{C['text_muted']};
    font-size:0.9rem; margin-top:8px; letter-spacing:1px">
        Upload your bank statement for AI-powered analysis
    </div>
</div>
""", unsafe_allow_html=True)

# One-time startup connection test
if "connection_tested" not in st.session_state:
    st.session_state.connection_tested = True
    try:
        _test_client = get_anthropic_client()
        _test_resp = _test_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}]
        )
        print("[startup] Claude API connected successfully")
    except SystemExit:
        pass  # st.stop() from get_api_key() — error already shown
    except Exception as _e:
        print(f"[startup] Claude API connection failed: {_e}")
        st.error(f"Claude API connection failed: {type(_e).__name__}: {_e}")


uploaded_files = st.file_uploader(
    "Upload one or more bank statements",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Account label inputs
    st.markdown("#### Account Labels")
    label_cols = st.columns(min(len(uploaded_files), 3))
    account_labels = {}
    for i, f in enumerate(uploaded_files):
        with label_cols[i % 3]:
            default = f.name.rsplit(".", 1)[0]
            account_labels[f.name] = st.text_input(
                f"Label for {f.name}", value=default, key=f"label_{f.name}"
            )

    try:
        dfs = []
        for f in uploaded_files:
            raw = _read_file(f)
            processed = preprocess_dataframe(raw, account_label=account_labels[f.name])
            if processed is not None and len(processed) > 0:
                dfs.append(processed)

        if not dfs:
            st.error("Could not parse any files. Please ensure they have description and amount columns.")
        else:
            df = pd.concat(dfs, ignore_index=True)

            with st.expander("🔎 Sample descriptions from your file (click to see)"):
                st.write(df["description"].dropna().unique()[:30].tolist())

            # Flag internal transfers
            transfer_idx = detect_transfers(df)
            if transfer_idx:
                df.loc[list(transfer_idx), "category"] = "internal_transfer"
                st.info(f"Detected {len(transfer_idx)} internal transfer(s) — excluded from spending totals.")

            # Income / expense split
            if "type" in df.columns:
                # CSV has an explicit Type column — use it directly
                income_df = df[df["type"] == "income"].copy()
                expense_df = df[(df["type"] == "expense") & (df["category"] != "internal_transfer")].copy()
                st.caption(f"Income/expense split from CSV Type column — {len(income_df)} income, {len(expense_df)} expense rows.")
            else:
                # Fallback: let user pick sign convention or use keywords
                amount_convention = st.radio(
                    "How does your bank export amounts?",
                    options=[
                        "All positive — use description to detect income",
                        "Expenses positive, income negative",
                        "Income positive, expenses negative",
                    ],
                    index=0,
                    horizontal=True,
                )
                _income_kws = [
                    "salary", "payroll", "paycheck", "direct dep", "direct deposit",
                    "employer", "wages", "income", "nómina", "nomina", "sueldo",
                    "zelle from", "venmo from", "cashapp from", "transfer from",
                    "refund", "reimbursement", "dividend", "interest earned", "tax refund",
                ]
                if amount_convention == "Expenses positive, income negative":
                    df["is_income"] = df["signed_amount"] < 0
                elif amount_convention == "Income positive, expenses negative":
                    df["is_income"] = df["signed_amount"] > 0
                else:
                    df["is_income"] = df["description"].str.lower().str.contains(
                        "|".join(_income_kws), na=False
                    )
                income_df = df[df["is_income"]].copy()
                expense_df = df[~df["is_income"] & (df["category"] != "internal_transfer")].copy()

            # Sidebar account filter
            if "account" in df.columns:
                all_accounts = sorted(df["account"].dropna().unique().tolist())
                selected_accounts = st.sidebar.multiselect(
                    "Filter by account", options=all_accounts, default=all_accounts
                )
                expense_df = expense_df[expense_df["account"].isin(selected_accounts)]

            st.success(f"Loaded {len(df)} transactions from {len(dfs)} file(s) — {len(expense_df)} expenses, {len(income_df)} income")

            # Income / savings rate summary
            income_total = income_df["amount"].sum()
            expense_total = expense_df["amount"].sum()
            savings_rate = (income_total - expense_total) / income_total if income_total > 0 else 0
            sr_color = C["success"] if savings_rate > 0.20 else C["warning"] if savings_rate > 0.10 else C["danger"]
            sr_label = "Great 🌟" if savings_rate > 0.20 else "Fair 😐" if savings_rate > 0.10 else "Low ⚠️"
            n_metric_cols = 3
            mc = st.columns(n_metric_cols)
            with mc[0]:
                st.metric("Total Income", f"${income_total:,.2f}")
            with mc[1]:
                st.metric("Total Expenses", f"${expense_total:,.2f}")
            with mc[2 % n_metric_cols]:
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:10px; padding:14px; text-align:center;
margin-top:4px; border:1px solid {C['border']}">
    <div style="color:{C['text_muted']}; font-size:0.8rem">Savings Rate</div>
    <div style="color:{sr_color}; font-size:1.8rem; font-weight:bold">{savings_rate*100:.1f}%</div>
    <div style="color:{sr_color}; font-size:0.85rem">{sr_label}</div>
</div>""", unsafe_allow_html=True)

            if "show_savings_explanation" not in st.session_state:
                st.session_state.show_savings_explanation = False

            if st.button(
                "❓ How is this calculated?"
                if not st.session_state.show_savings_explanation
                else "✕ Close explanation",
                key="savings_rate_toggle",
            ):
                st.session_state.show_savings_explanation = (
                    not st.session_state.show_savings_explanation
                )

            if st.session_state.show_savings_explanation:

                # Block 1: Formula
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:14px;
padding:20px; margin-bottom:12px; border:1px solid {C['border']}">
    <div style="font-family:'Cinzel',serif; color:{C['accent_primary']};
    font-size:1rem; margin-bottom:16px; letter-spacing:1px">
        ◆ The Formula
    </div>
    <div style="background:{C['bg_secondary']}; border-radius:10px;
    padding:16px; text-align:center">
        <div style="font-family:'Orbitron',monospace;
        font-size:1.1rem; color:{C['text_primary']}">
            Savings Rate =
            <span style="color:{C['accent_gold']}">
                ( Income − Expenses )
            </span>
            ÷
            <span style="color:{C['accent_primary']}">
                Income
            </span>
            × 100
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

                # Block 2: Your numbers
                st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:14px;
padding:20px; margin-bottom:12px; border:1px solid {C['border']}">
    <div style="color:{C['text_muted']}; font-size:0.8rem;
    margin-bottom:8px; text-align:center">Your calculation:</div>
    <div style="font-family:'Orbitron',monospace; font-size:1rem;
    color:{C['text_primary']}; text-align:center">
        (
        <span style="color:{C['success']}">${income_total:,.0f}</span>
        −
        <span style="color:{C['danger']}">${expense_total:,.0f}</span>
        ) ÷
        <span style="color:{C['success']}">${income_total:,.0f}</span>
        × 100 =
        <span style="color:{C['accent_gold']};
        font-size:1.4rem; font-weight:700">
            {savings_rate * 100:.1f}%
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

                # Block 3: Benchmark grid — st.columns avoids CSS grid rendering issues
                _b1, _b2, _b3, _b4 = st.columns(4)
                _benchmarks_grid = [
                    (_b1, "Below 0%",  "Spending more\nthan earning", C["danger"]),
                    (_b2, "0% – 10%",  "Very little\nsaved",          C["warning"]),
                    (_b3, "10% – 20%", "On the\nright track",         C["accent_gold"]),
                    (_b4, "Above 20%", "Excellent\ndiscipline",       C["success"]),
                ]
                for _bcol, _bpct, _bdesc, _bcolor in _benchmarks_grid:
                    with _bcol:
                        st.markdown(f"""
<div style="background:{C['bg_card']}; border:1px solid {_bcolor};
border-radius:8px; padding:10px; text-align:center">
    <div style="color:{_bcolor}; font-size:0.75rem; font-weight:600">
        {_bpct}
    </div>
    <div style="color:{C['text_muted']}; font-size:0.78rem; margin-top:4px">
        {_bdesc}
    </div>
</div>
""", unsafe_allow_html=True)

                # Block 4: Where you stand
                st.markdown(
                    "<div style='margin-top:16px'></div>",
                    unsafe_allow_html=True,
                )
                _sr_pct = savings_rate * 100
                _thresholds = [
                    ("🚨 Critical — below 0%",    -999,  0, C["danger"]),
                    ("😬 Concerning — 0% to 10%",    0, 10, C["warning"]),
                    ("⚠️ Okay — 10% to 20%",        10, 20, C["accent_gold"]),
                    ("✅ Good — 20% to 30%",         20, 30, C["accent_primary"]),
                    ("🚀 Excellent — above 30%",     30, 999, C["success"]),
                ]
                for _tlabel, _tlow, _thigh, _tcolor in _thresholds:
                    _is_here = _tlow <= _sr_pct < _thigh
                    if _is_here:
                        st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:8px;
padding:10px 14px; border-left:4px solid {_tcolor}; margin:4px 0;
display:flex; justify-content:space-between">
    <span style="color:{_tcolor}; font-weight:700">
        👉 YOU ARE HERE → {_tlabel}
    </span>
    <span style="color:{_tcolor}; font-family:'Orbitron',monospace;
    font-weight:700">{_sr_pct:.1f}%</span>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div style="padding:6px 14px; margin:2px 0;
color:{C['text_muted']}; font-size:0.85rem">
    {_tlabel}
</div>
""", unsafe_allow_html=True)

            if expense_df is None or len(expense_df) == 0:
                st.error("No expense transactions found after filtering.")
                st.markdown("#### 🔍 Diagnostic Info")
                diag1, diag2, diag3 = st.columns(3)
                with diag1:
                    st.metric("Total rows", len(df))
                    st.metric("Positive amounts", int((df["signed_amount"] > 0).sum()))
                with diag2:
                    st.metric("Negative amounts", int((df["signed_amount"] < 0).sum()))
                    st.metric("Zero amounts", int((df["signed_amount"] == 0).sum()))
                with diag3:
                    st.metric("Amount min", f"${df['amount'].min():,.2f}")
                    st.metric("Amount max", f"${df['amount'].max():,.2f}")
                st.write("**Sample descriptions:**", df["description"].head(10).tolist())
                st.info("Try a different option in the 'How does your bank export amounts?' selector above.")
            else:
                with st.expander("📋 Transaction Details (click to expand)"):
                    editor_cols = (
                        ["description", "amount", "category"]
                        + (["account"] if "account" in expense_df.columns else [])
                    )
                    edited = st.data_editor(
                        expense_df[editor_cols],
                        column_config={
                            "category": st.column_config.SelectboxColumn(
                                "Category", options=list(CATEGORIES.keys()) + ["other"],
                            ),
                            "amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )
                    expense_df = expense_df.copy()
                    expense_df["category"] = edited["category"].values

                score, cat_pcts = calculate_health_score(expense_df)
                df_for_analysis = expense_df

                # Build chat context once per file load; invalidate when row count changes
                _ctx_fingerprint = len(expense_df)
                if st.session_state.get("chat_context_key") != _ctx_fingerprint:
                    st.session_state["chat_context"] = build_chat_context(expense_df, income_df)
                    st.session_state["chat_context_key"] = _ctx_fingerprint

                tab_overview, tab_analytics, tab_budget, tab_insights, tab_roast = st.tabs(
                    ["📊 Overview", "🔬 Deep Analytics", "📋 Budget Planner", "💡 Insights", "🔥 Roast"]
                )

                # ===== OVERVIEW TAB =====
                with tab_overview:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("📊 Spending by Category")
                        cat_totals = df_for_analysis.groupby("category")["amount"].sum().reset_index()
                        fig = px.pie(
                            cat_totals, values="amount", names="category",
                            color="category", color_discrete_map=CATEGORY_COLORS, hole=0.4,
                        )
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                        fig.update_layout(
                            height=420,
                            showlegend=True,
                            margin=dict(t=20, b=60, l=0, r=0),
                            paper_bgcolor=C["bg_card"],
                            plot_bgcolor=C["bg_card"],
                            font_color=C["text_primary"],
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.28,
                                xanchor="center",
                                x=0.5,
                                bgcolor=C["bg_secondary"],
                                bordercolor=C["border"],
                                borderwidth=1,
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True, key="chart_8")

                    with col2:
                        st.subheader("🏥 Financial Health Score")
                        color = get_score_color(score)
                        label = get_score_label(score)
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div style="font-size: 80px; font-weight: bold; color: {color};">{score}</div>
                            <div style="font-size: 24px; color: {color};">/100</div>
                            <div style="font-size: 20px; margin-top: 10px;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(score / 100)

                        with st.expander("How is this calculated?"):
                            components = compute_score_components(df_for_analysis)
                            score_items = [
                                ("📊 Spending Consistency", components["consistency"], 25,
                                 "How stable your spending is month to month"),
                                ("🎯 Discretionary Control", components["discretionary"], 25,
                                 "% of spending on Food, Shopping, Entertainment, Transport"),
                                ("💰 Savings Signal", components["savings"], 25,
                                 "Detected transfers to savings or investment accounts"),
                                ("📉 Low Volatility", components["volatility"], 25,
                                 "Low variation in monthly totals"),
                            ]
                            for name, comp_score, max_score, description in score_items:
                                pct = comp_score / max_score
                                bar_color = C["success"] if pct >= 0.8 else C["warning"] if pct >= 0.5 else C["danger"]
                                status = "✅" if pct >= 0.8 else "⚠️" if pct >= 0.5 else "❌"
                                st.markdown(f"""
<div style="margin:10px 0;">
    <div style="display:flex; justify-content:space-between; color:{C['text_primary']}; font-size:0.9rem">
        <span>{status} {name}</span>
        <span style="color:{bar_color}">{comp_score}/{max_score}</span>
    </div>
    <div style="background:{C['border']}; border-radius:4px; height:6px; margin:4px 0;">
        <div style="background:{bar_color}; width:{pct*100:.0f}%; height:6px; border-radius:4px;"></div>
    </div>
    <div style="color:{C['text_muted']}; font-size:0.75rem">{description}</div>
</div>""", unsafe_allow_html=True)
                            st.markdown("---")
                            with st.spinner("Generating what-if scenarios..."):
                                whatif = get_whatif_lines(df_for_analysis, components, score)
                            for line in whatif:
                                st.markdown(f"""
<div style="background:{C['bg_secondary']}; border-left:3px solid {C['accent_primary']};
border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0;
color:{C['text_secondary']}; font-size:0.9rem;">
    💡 {line}
</div>""", unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                    st.markdown("### Category Breakdown")

                    _has_date = ("date" in df_for_analysis.columns
                                 and pd.to_datetime(df_for_analysis["date"], errors="coerce").notna().sum() > 0)

                    _cat_totals = df_for_analysis.groupby("category")["amount"].sum().sort_values(ascending=False)
                    _total_spend = _cat_totals.sum()

                    if _has_date:
                        _cat_monthly_avg = (
                            df_for_analysis.groupby("category")
                            .apply(lambda x: x.groupby(pd.to_datetime(x["date"], errors="coerce").dt.to_period("M"))["amount"].sum().mean())
                            .round(2)
                        )
                        _max_idx = df_for_analysis.groupby("category")["amount"].idxmax()
                        _cat_max_txn = df_for_analysis.loc[_max_idx][["category", "amount", "date"]].set_index("category")
                        df_for_analysis["_month_period"] = pd.to_datetime(df_for_analysis["date"], errors="coerce").dt.to_period("M")
                        _last_2 = sorted(df_for_analysis["_month_period"].dropna().unique())[-2:]
                        _num_months = df_for_analysis["_month_period"].nunique()
                    else:
                        _cat_monthly_avg = pd.Series(dtype=float)
                        _cat_max_txn = pd.DataFrame()
                        _last_2 = []
                        _num_months = 1

                    _mom_map = {}
                    if len(_last_2) == 2:
                        _prev = df_for_analysis[df_for_analysis["_month_period"] == _last_2[0]].groupby("category")["amount"].sum()
                        _curr = df_for_analysis[df_for_analysis["_month_period"] == _last_2[1]].groupby("category")["amount"].sum()
                        for _cat in _cat_totals.index:
                            _p, _c = _prev.get(_cat, 0), _curr.get(_cat, 0)
                            _mom_map[_cat] = (_c - _p) / _p * 100 if _p > 0 else 0

                    _rows_html = ""
                    _largest_pct = (_cat_totals.iloc[0] / _total_spend * 100) if _total_spend > 0 else 1
                    for _i, (_cat, _total) in enumerate(_cat_totals.items()):
                        _pct = _total / _total_spend * 100 if _total_spend > 0 else 0
                        _avg = _cat_monthly_avg.get(_cat, _total)
                        _mom = _mom_map.get(_cat, 0)
                        _mom_color = C["success"] if _mom <= 0 else C["danger"]
                        _mom_arrow = "↓" if _mom <= 0 else "↑"
                        _bar_w = min(_pct / _largest_pct * 100, 100)
                        _row_bg = C["bg_card"] if _i % 2 == 0 else C["bg_secondary"]

                        if not _cat_max_txn.empty and _cat in _cat_max_txn.index:
                            _mx = _cat_max_txn.loc[_cat]
                            _max_str = f"${_mx['amount']:,.0f} on {str(_mx['date'])[:10]}"
                        else:
                            _max_str = "—"

                        _rows_html += f"""
        <tr style="background:{_row_bg}">
            <td style="padding:10px 12px; color:{C['text_primary']};
            font-weight:600; white-space:nowrap">{_cat.capitalize()}</td>
            <td style="padding:10px 12px; font-family:'Orbitron',monospace;
            color:{C['accent_primary']}; font-weight:600;
            text-align:right; white-space:nowrap">${_total:,.0f}</td>
            <td style="padding:10px 12px; min-width:100px; white-space:nowrap">
                <div style="background:{C['bg_primary']};
                border-radius:4px; height:8px; width:100%">
                    <div style="background:{C['accent_primary']};
                    border-radius:4px; height:8px;
                    width:{_bar_w:.0f}%"></div>
                </div>
                <div style="color:{C['text_muted']};
                font-size:0.75rem; margin-top:2px">{_pct:.1f}%</div>
            </td>
            <td style="padding:10px 12px; color:{C['text_secondary']};
            text-align:right; white-space:nowrap">${_avg:,.0f}/mo</td>
            <td style="padding:10px 12px; color:{_mom_color};
            font-weight:600; text-align:right; white-space:nowrap">
                {_mom_arrow} {abs(_mom):.1f}%
            </td>
            <td style="padding:10px 12px; color:{C['text_muted']};
            font-size:0.82rem; white-space:nowrap">{_max_str}</td>
        </tr>"""

                    st.markdown(f"""
<div style="width:100%; border-radius:14px; border:1px solid {C['border']};
margin:16px 0; overflow-x:auto; -webkit-overflow-scrolling:touch;">
    <table style="width:100%; border-collapse:collapse;
    font-family:'Nunito',sans-serif">
        <thead>
            <tr style="background:{C['bg_secondary']};
            border-bottom:2px solid {C['accent_primary']}">
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:left; font-weight:600">Category</th>
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:right; font-weight:600">Total</th>
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:left; font-weight:600">Share %</th>
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:right; font-weight:600">Avg/Month</th>
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:right; font-weight:600">MoM</th>
                <th style="padding:10px 12px; white-space:nowrap;
                color:{C['text_muted']}; font-size:0.78rem;
                text-transform:uppercase; letter-spacing:1px;
                text-align:left; font-weight:600">Top Txn</th>
            </tr>
        </thead>
        <tbody>{_rows_html}</tbody>
        <tfoot>
            <tr style="background:{C['bg_secondary']};
            border-top:2px solid {C['accent_primary']}">
                <td style="padding:10px 12px; color:{C['text_primary']};
                font-weight:700; font-family:'Cinzel',serif; white-space:nowrap">TOTAL</td>
                <td style="padding:10px 12px; font-family:'Orbitron',monospace;
                color:{C['accent_gold']}; font-weight:700;
                text-align:right; white-space:nowrap">${_total_spend:,.0f}</td>
                <td colspan="4" style="padding:10px 12px;
                color:{C['text_muted']}; font-size:0.82rem; white-space:nowrap">
                    {len(_cat_totals)} categories · {_num_months} months analyzed
                </td>
            </tr>
        </tfoot>
    </table>
</div>
""", unsafe_allow_html=True)

                # ===== DEEP ANALYTICS TAB =====
                with tab_analytics:
                    render_deep_analytics(df_for_analysis)
                    render_finance_chat(df)

                # ===== BUDGET PLANNER TAB =====
                with tab_budget:
                    st.subheader("💼 Your Budget Setup")
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        monthly_income = st.number_input(
                            "Monthly Income (after tax)",
                            min_value=0, value=st.session_state.get("monthly_income", 5000),
                            step=100, help="Enter your average monthly take-home pay"
                        )
                    with b_col2:
                        currency = st.selectbox(
                            "Currency", ["$", "€", "£", "¥"],
                            index=["$", "€", "£", "¥"].index(st.session_state.get("currency", "$"))
                        )
                    st.session_state["monthly_income"] = monthly_income
                    st.session_state["currency"] = currency

                    st.markdown("#### Set your monthly budget targets per category")
                    budget_cats = ["Bills", "Food", "Shopping", "Transport", "Entertainment", "Other", "Savings"]
                    n_budget_cols = 3
                    budget_cols = st.columns(n_budget_cols)
                    budget_inputs = {}
                    for i, cat in enumerate(budget_cats):
                        with budget_cols[i % n_budget_cols]:
                            budget_inputs[cat] = st.number_input(
                                cat, min_value=0,
                                value=st.session_state.get(f"budget_{cat}", 0),
                                step=50, key=f"budget_{cat}"
                            )
                    st.session_state["budget_inputs"] = budget_inputs

                    total_budgeted = sum(budget_inputs.values())
                    remaining = monthly_income - total_budgeted
                    rem_color = C["success"] if remaining >= 0 else C["danger"]
                    st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:10px; padding:12px;
display:flex; justify-content:space-between; margin-top:8px;
border:1px solid {C['border']}">
    <span style="color:{C['text_muted']}">Total budgeted:
        <strong style="color:{C['text_primary']}">{currency}{total_budgeted:,}</strong>
    </span>
    <span style="color:{rem_color}">
        {"Remaining" if remaining >= 0 else "Over budget"}:
        {currency}{abs(remaining):,}
    </span>
</div>""", unsafe_allow_html=True)

                    st.divider()
                    if st.button("Get AI Budget Recommendations", type="primary", use_container_width=True):
                        bp_num_months = 1
                        if "date" in df_for_analysis.columns:
                            bp_num_months = max(df_for_analysis["date"].dt.to_period("M").nunique(), 1)
                        cat_monthly_avgs = {
                            cat.capitalize(): round(amt / bp_num_months, 0)
                            for cat, amt in df_for_analysis.groupby("category")["amount"].sum().items()
                        }
                        with st.spinner("Building your personalized budget plan..."):
                            try:
                                budget_data = get_budget_recommendations(
                                    monthly_income, budget_inputs, cat_monthly_avgs, currency
                                )
                                render_budget_recommendations(budget_data, currency, monthly_income)
                            except Exception as e:
                                st.error(f"Budget recommendation failed: {str(e)}")

                # ===== INSIGHTS TAB =====
                with tab_insights:
                    st.subheader("💡 Financial Insights")
                    st.markdown("Get a structured AI analysis of your spending patterns and actionable advice.")
                    if st.button("Analyze My Spending", type="primary", use_container_width=True,
                                 key="btn_insights"):
                        st.markdown("### 💡 Your Financial Insights")
                        with st.spinner("Analyzing your finances..."):
                            try:
                                data = get_insights(df_for_analysis, cat_pcts, score)
                                render_insights(data)
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")

                # ===== ROAST TAB =====
                with tab_roast:
                    if "roast_slide" not in st.session_state:
                        st.session_state.roast_slide = 0
                    if "roast_data" not in st.session_state:
                        st.session_state.roast_data = None

                    st.subheader("🔥 Roast Mode")
                    st.markdown("Let Claude brutally critique your spending habits.")

                    ctrl1, ctrl2 = st.columns([2, 1])
                    with ctrl1:
                        roast_level = st.select_slider(
                            "Roast Intensity",
                            options=["Mild 🌱", "Medium 🔥", "Gordon Ramsay 💀"],
                            value=st.session_state.get("roast_level", "Medium 🔥"),
                            key="roast_level_slider",
                        )
                    # Always sync slider value to session state
                    st.session_state["roast_level"] = roast_level

                    with ctrl2:
                        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
                        fire_btn = st.button(
                            "🔥 Roast Me", type="primary",
                            use_container_width=True, key="btn_roast",
                        )

                    if fire_btn:
                        with st.spinner("Sharpening the knives..."):
                            try:
                                print("=== ROAST DEBUG ===")
                                print(f"roast_level selected: '{roast_level}'")
                                print(f"roast_level in session: '{st.session_state.get('roast_level', 'NOT SET')}'")
                                print(f"'Gordon' in roast_level: {'Gordon' in roast_level}")

                                roast_data = get_roast(df_for_analysis, score, roast_level)

                                print(f"roast_data keys: {list(roast_data.keys())}")
                                print(f"number of roasts: {len(roast_data.get('roasts', []))}")

                                roast_memes = assign_memes(roast_data)

                                st.session_state["roast_data"]  = roast_data
                                st.session_state["roast_memes"] = roast_memes
                                st.session_state["roast_slide"] = 0

                                print(f"Stored roast_memes: {roast_memes}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Roast failed: {str(e)}")

                    if st.session_state.roast_data is None:
                        st.markdown(f"""
<div style="text-align:center; padding:60px 20px">
    <div style="font-size:4rem">🔥</div>
    <div style="color:{C['text_primary']}; font-size:1.5rem; font-weight:bold; margin:16px 0">
        Ready to get roasted?
    </div>
    <div style="color:{C['text_muted']}">Select your roast intensity above and hit the button</div>
</div>""", unsafe_allow_html=True)
                    else:
                        _render_roast_slideshow(
                            st.session_state.roast_data,
                            st.session_state.get("roast_level", "Medium 🔥"),
                        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Hero section
    st.markdown(f"""
<div style="text-align:center; padding:32px 0 16px;">
    <div style="color:{C['text_muted']}; font-size:1.1rem; max-width:600px; margin:0 auto; line-height:1.6">
        Upload your bank statement CSV or Excel file and get AI-powered analysis of your spending —
        interactive charts, a financial health score, and an optional brutal roast of your habits.
    </div>
</div>""", unsafe_allow_html=True)

    # Feature cards — 3 columns
    fc1, fc2, fc3 = st.columns(3)
    feature_cards = [
        (fc1, "📊", "Spending Breakdown",
         "Auto-categorizes every transaction into restaurants, food, transport, shopping, entertainment, bills, and more."),
        (fc2, "🏥", "Financial Health Score",
         "A 0–100 score based on spending consistency, discretionary control, savings signals, and month-over-month volatility."),
        (fc3, "🔬", "Deep Analytics",
         "Time series trends, category breakdowns, year-over-year comparisons, subscription detector, and impulse buy flags."),
    ]
    for col, icon, title, desc in feature_cards:
        with col:
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:14px; padding:22px; text-align:center; height:180px;
border:1px solid {C['border']}; border-top:3px solid {C['accent_primary']};
display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px;">
    <div style="font-size:2.2rem">{icon}</div>
    <div style="color:{C['text_primary']}; font-weight:bold; font-size:1rem">{title}</div>
    <div style="color:{C['text_muted']}; font-size:0.82rem; line-height:1.5">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fc4, fc5, fc6 = st.columns(3)
    feature_cards2 = [
        (fc4, "🔥", "Roast Mode",
         "Three intensities — Mild, Medium, or Gordon Ramsay. Claude will absolutely roast your financial choices."),
        (fc5, "📋", "Budget Planner",
         "Enter your income and targets per category. Claude recommends an optimized budget and flags where you're over."),
        (fc6, "💬", "Chat With Your Finances",
         "Ask natural-language questions about your spending — 'What was my biggest month?' or 'Top 5 merchants?'"),
    ]
    for col, icon, title, desc in feature_cards2:
        with col:
            st.markdown(f"""
<div style="background:{C['bg_card']}; border-radius:14px; padding:22px; text-align:center; height:180px;
border:1px solid {C['border']}; border-top:3px solid {C['accent_gold']};
display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px;">
    <div style="font-size:2.2rem">{icon}</div>
    <div style="color:{C['text_primary']}; font-weight:bold; font-size:1rem">{title}</div>
    <div style="color:{C['text_muted']}; font-size:0.82rem; line-height:1.5">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    # File format guidance + sample download
    hint_col, sample_col = st.columns([3, 1])
    with hint_col:
        st.markdown("#### What file format do I need?")
        st.markdown("""
Your CSV or Excel file just needs **a description column** and **an amount column** — column names are detected automatically.
Optionally include a **Date** column (unlocks heatmaps, subscription detection, and month-over-month stats)
and a **Type** column with `income`/`expense` labels.

Supported exports from: Chase, Bank of America, Wells Fargo, Revolut, N26, Monzo, BBVA, Santander, and most banks.
""")
    with sample_col:
        st.markdown("#### Try it now")
        st.markdown("Don't have your statement handy? Download a realistic 3-month sample:")
        st.download_button(
            label="⬇️ Download sample CSV",
            data=generate_sample_csv(),
            file_name="sample_bank_statement.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
