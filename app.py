import io
import json
import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import anthropic

st.set_page_config(
    page_title="AI Financial Spending Judge",
    page_icon="💰",
    layout="wide"
)


def get_anthropic_client() -> anthropic.Anthropic:
    """Return an Anthropic client, reading the API key from st.secrets or env."""
    key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not key:
        st.error(
            "**No API key found.** Add your Anthropic API key:\n\n"
            "- **Local dev:** set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml`\n"
            "- **Streamlit Cloud:** add it in the app's Secrets settings"
        )
        st.stop()
    return anthropic.Anthropic(api_key=key)


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


def preprocess_dataframe(df: pd.DataFrame, account_label: str = ""):
    desc_candidates = ["description", "transaction description", "merchant", "name",
                       "payee", "details", "memo", "narration", "transaction"]
    amount_candidates = ["amount", "debit", "credit", "value", "sum", "charge", "withdrawal"]
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
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_col = col
                break
        except Exception:
            pass

    signed_amt = pd.to_numeric(df[amount_col], errors="coerce")
    result = pd.DataFrame({
        "description": df[desc_col].astype(str),
        "amount": signed_amt.abs(),
        "signed_amount": signed_amt,
    })
    if date_col:
        result["date"] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
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
                parsed = pd.to_datetime(work[c], infer_datetime_format=True, errors="coerce")
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
    import numpy as np

    work = df.copy()
    # attach month period if possible (reuse logic from category_stats)
    for c in work.columns:
        if c not in ("description", "amount", "category"):
            try:
                parsed = pd.to_datetime(work[c], infer_datetime_format=True, errors="coerce")
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
        model="claude-opus-4-7",
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
        model="claude-opus-4-7",
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
<div style="background:#1a1a2e; border:1px solid #3498DB; border-radius:12px;
padding:16px 20px; margin:12px 0; color:#ccc; font-size:1rem;">
    📋 <strong style="color:white">Summary:</strong> {summary}
</div>""", unsafe_allow_html=True)

    # Allocation rows
    st.markdown("#### 📊 Category Allocations")
    for item in allocations:
        status = item.get("status", "on_track")
        status_color = "#E74C3C" if status == "over" else "#2ECC71" if status == "under" else "#3498DB"
        status_icon = "🔴" if status == "over" else "🟢" if status == "under" else "🔵"
        status_label = "Over budget" if status == "over" else "Under budget" if status == "under" else "On track"
        current = item.get("current_monthly_avg", 0)
        user_budget = item.get("user_budget", 0)
        suggested = item.get("suggested_budget", 0)
        suggested_pct = item.get("suggested_pct", 0)
        st.markdown(f"""
<div style="background:#111; border-radius:10px; padding:14px; margin:6px 0;
border-left:4px solid {status_color}">
    <div style="display:flex; justify-content:space-between">
        <strong style="color:white">{item.get('category','')}</strong>
        <span style="color:{status_color}">{status_icon} {status_label}</span>
    </div>
    <div style="color:#888; font-size:0.8rem; margin-top:6px">
        Current avg: {currency}{current:,.0f}/mo
        &nbsp;·&nbsp; Your target: {currency}{user_budget:,.0f}
        &nbsp;·&nbsp; AI suggests: {currency}{suggested:,.0f} ({suggested_pct}% of income)
    </div>
    <div style="color:#aaa; font-size:0.82rem; margin-top:4px; font-style:italic">
        {item.get('reasoning','')}
    </div>
</div>""", unsafe_allow_html=True)

    # Tips in 2-column cards
    if tips:
        st.markdown("#### 💡 Budget Tips")
        tip_colors = ["#2ECC71", "#E74C3C", "#3498DB", "#9B59B6"]
        for i in range(0, len(tips), 2):
            pair = tips[i:i+2]
            tip_cols = st.columns(len(pair))
            for col, item, color in zip(tip_cols, pair, tip_colors[i:i+2]):
                with col:
                    st.markdown(f"""
<div style="background:#111; border-radius:12px; padding:16px; margin:8px 0;
border-top:3px solid {color}; display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.5rem">{item.get('icon','💡')}</span>
    <span style="color:white; font-size:0.95rem">{item.get('tip','')}</span>
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)


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
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
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
        tone = ("Be Gordon Ramsay in Hell's Kitchen but for finances. "
                "Brutal, loud, specific, no mercy. Use caps for emphasis occasionally. "
                "Call them out hard.")

    system = (
        f"{tone} "
        "Structure your roast in exactly this order, return as JSON only, no markdown, no backticks: "
        '{"opening_line":"...","merchant_roast":"...","time_roast":"...", '
        '"impulse_roast":"...","backhanded_compliment":"...","final_score":5,"score_label":"..."}'
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
        "Roast them. final_score is 1-10 (be fair). "
        "score_label is exactly 3 funny words describing their financial style."
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_claude_json(response.content[0].text)


def render_roast(data: dict):
    opening = data.get("opening_line", "")
    final_score = int(data.get("final_score", 5))
    score_label = data.get("score_label", "")

    # Opening line — large centered
    st.markdown(f"""
<div style="text-align:center; padding:28px 16px;">
    <div style="color:white; font-size:1.6rem; font-weight:bold; line-height:1.5">
        {opening}
    </div>
</div>""", unsafe_allow_html=True)

    # Sequential roast cards
    cards = [
        ("🏪", "Top Merchant Offense",      data.get("merchant_roast", ""),          "#E74C3C"),
        ("🌙", "Night Owl Report",           data.get("time_roast", ""),              "#9B59B6"),
        ("💸", "Impulse Control Issues",     data.get("impulse_roast", ""),           "#E67E22"),
        ("🌹", "Backhanded Compliment",      data.get("backhanded_compliment", ""),   "#2ECC71"),
    ]
    for icon, title, text, color in cards:
        st.markdown(f"""
<div style="background:#111; border-radius:12px; padding:20px; margin:10px 0;
border-top:3px solid {color};">
    <div style="color:{color}; font-size:0.85rem; font-weight:bold; margin-bottom:8px">
        {icon} {title}
    </div>
    <div style="color:white; font-size:1rem; line-height:1.6">{text}</div>
</div>""", unsafe_allow_html=True)

    # Final score
    score_color = "#2ECC71" if final_score >= 7 else "#F39C12" if final_score >= 4 else "#E74C3C"
    st.markdown(f"""
<div style="text-align:center; padding:32px 0 16px;">
    <div style="color:#666; font-size:0.85rem; letter-spacing:2px">FINANCIAL ROAST SCORE</div>
    <div style="color:{score_color}; font-size:6rem; font-weight:bold; line-height:1">
        {final_score}
    </div>
    <div style="color:{score_color}; font-size:1.4rem">/10</div>
    <div style="color:white; font-size:1.15rem; margin-top:10px; font-style:italic">
        {score_label}
    </div>
</div>""", unsafe_allow_html=True)

    if final_score >= 8:
        st.balloons()
    elif final_score <= 3:
        st.snow()


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
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def get_insights(df: pd.DataFrame, cat_pcts: dict, score: int, roast: bool) -> dict:
    client = get_anthropic_client()
    total = df["amount"].sum()
    breakdown = "\n".join([
        f"- {cat}: ${cat_pcts[cat]/100*total:.2f} ({cat_pcts[cat]:.1f}%)"
        for cat in sorted(cat_pcts, key=lambda x: cat_pcts[x], reverse=True)
    ])
    biggest_cat = max(cat_pcts, key=cat_pcts.get)
    biggest_pct = cat_pcts[biggest_cat]
    biggest_amt = biggest_pct / 100 * total

    data_context = f"""Transaction data:
Total Spending: ${total:.2f}
Health Score: {score}/100 ({get_score_label(score)})
Spending Breakdown:
{breakdown}"""

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
    stats_prompt = f"Compute the 4 key stats for this data:\n{data_context}"

    # --- Call 2: Observations ---
    if roast:
        obs_system = (
            "Return ONLY a valid JSON array, no other text, no markdown, no backticks. "
            "Maximum 4 items. Each is a savage, funny roast of the spending. Under 15 words each. "
            'Format: [{"type":"warning","text":"..."},{"type":"good","text":"..."}] '
            "type is either 'good' or 'warning' only."
        )
    else:
        obs_system = (
            "Return ONLY a valid JSON array, no other text, no markdown, no backticks. "
            "Maximum 4 items. Each item is one short punchy observation under 15 words. "
            'Format: [{"type":"warning","text":"..."},{"type":"good","text":"..."}] '
            "type is either 'good' or 'warning' only."
        )
    obs_prompt = f"Generate observations for this spending data:\n{data_context}"

    # --- Call 3: Action Tips ---
    if roast:
        tips_system = (
            "Return ONLY a valid JSON array, no other text, no markdown, no backticks. "
            "Maximum 4 items. Each tip is a savage but actionable burn under 12 words. Be specific with numbers. "
            'Format: [{"icon":"🔍","color":"#2ECC71","tip":"..."}] '
            "Use colors: #2ECC71 green, #E74C3C red, #3498DB blue, #9B59B6 purple."
        )
    else:
        tips_system = (
            "Return ONLY a valid JSON array, no other text, no markdown, no backticks. "
            "Maximum 4 items. Each tip must be under 12 words. Be specific with numbers. "
            'Format: [{"icon":"🔍","color":"#2ECC71","tip":"..."}] '
            "Use colors: #2ECC71 green, #E74C3C red, #3498DB blue, #9B59B6 purple."
        )
    tips_prompt = f"Generate action tips for this spending data:\n{data_context}"

    stats = parse_claude_json(_call(client, stats_system, stats_prompt))
    observations = parse_claude_json(_call(client, obs_system, obs_prompt))
    tips = parse_claude_json(_call(client, tips_system, tips_prompt))

    return {"stats": stats, "observations": observations, "tips": tips}


def render_insights(data: dict, roast: bool):
    stats = data.get("stats", [])
    observations = data.get("observations", [])
    tips = data.get("tips", [])

    # Key Stats row
    st.markdown("#### 📈 Key Stats")
    n_stat_cols = 2 if st.session_state.get("mobile_mode", False) else 4
    for row_start in range(0, len(stats), n_stat_cols):
        row_items = stats[row_start:row_start + n_stat_cols]
        row_cols = st.columns(len(row_items))
        for col, item in zip(row_cols, row_items):
            with col:
                st.markdown(f"""
<div style="background:#1a1a2e; border-radius:12px; padding:20px; text-align:center;
border:1px solid #333;">
    <div style="font-size:2rem">{item['icon']}</div>
    <div style="color:#888; font-size:0.8rem; margin-top:8px">{item['label']}</div>
    <div style="color:white; font-size:1.6rem; font-weight:bold; margin:4px 0">{item['value']}</div>
    <div style="color:#666; font-size:0.75rem">{item['sublabel']}</div>
</div>""", unsafe_allow_html=True)

    # Observations
    st.markdown("#### 🔍 Key Observations")
    for item in observations:
        color = "#2ECC71" if item.get("type") == "good" else "#E74C3C"
        icon = "✅" if item.get("type") == "good" else "⚠️"
        st.markdown(f"""
<div style="border-left:4px solid {color}; padding:12px 16px; margin:8px 0;
background:#111; border-radius:0 8px 8px 0;">
    {icon} {item['text']}
</div>""", unsafe_allow_html=True)

    # Action Tips in 2x2 grid
    st.markdown("#### 💡 Action Tips")
    for i in range(0, len(tips), 2):
        pair = tips[i:i+2]
        tip_cols = st.columns(len(pair))
        for col, item in zip(tip_cols, pair):
            with col:
                st.markdown(f"""
<div style="background:#111; border-radius:12px; padding:16px; margin:8px 0;
border-top:3px solid {item['color']}; display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.5rem">{item['icon']}</span>
    <span style="color:white; font-size:0.95rem">{item['tip']}</span>
</div>""", unsafe_allow_html=True)


def render_deep_analytics(df: pd.DataFrame):
    DARK = dict(plot_bgcolor="#0e0e0e", paper_bgcolor="#0e0e0e", font_color="white")

    # Detect date column once
    work = df.copy()
    date_col_found = False
    for c in work.columns:
        if c in ("description", "amount", "category"):
            continue
        try:
            parsed = pd.to_datetime(work[c], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(work) * 0.5:
                work["_date"] = parsed
                date_col_found = True
                break
        except Exception:
            pass

    no_date_msg = "No date column detected in your file — add a date column to unlock this chart."

    # ---- CHART 1: Day of Week Heatmap ----
    st.markdown("#### 📅 Which day of the week costs you most")
    if date_col_found:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        work["_dow"] = work["_date"].dt.day_name()
        _hgrp = "category_group" if "category_group" in work.columns else "category"
        heat_df = work.groupby(["_dow", _hgrp])["amount"].sum().reset_index()
        heat_pivot = heat_df.pivot(index="_dow", columns=_hgrp, values="amount").fillna(0)
        heat_pivot = heat_pivot.reindex([d for d in dow_order if d in heat_pivot.index])
        fig1 = px.imshow(
            heat_pivot,
            color_continuous_scale="Oranges",
            title="Which day of the week costs you most",
            labels={"x": "Category", "y": "Day", "color": "Amount ($)"},
            aspect="auto",
        )
        fig1.update_layout(**DARK, title_x=0)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info(no_date_msg)

    st.divider()

    # ---- CHART 2: Merchant Bubble Chart ----
    st.markdown("#### 🫧 Your top merchants — frequency vs total spend")
    merchant_df = (
        df.groupby("description")
        .agg(frequency=("amount", "count"), total=("amount", "sum"),
             avg_amount=("amount", "mean"), category=("category", "first"))
        .reset_index()
    )
    merchant_df = merchant_df[merchant_df["frequency"] >= 2].nlargest(40, "total")
    if not merchant_df.empty:
        fig2 = px.scatter(
            merchant_df,
            x="frequency", y="total",
            size="avg_amount",
            color="category",
            hover_name="description",
            color_discrete_map=CATEGORY_COLORS,
            size_max=60,
            title="Your top merchants — frequency vs total spend",
            labels={"frequency": "Number of Transactions",
                    "total": "Total Amount ($)", "avg_amount": "Avg Transaction"},
        )
        fig2.update_layout(**DARK, title_x=0)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough repeat merchants to display (need at least 2 transactions per merchant).")

    st.divider()

    # ---- CHART 3: Subscription Detector ----
    st.markdown("#### 🔄 Subscription Detector")
    if date_col_found:
        work["_month"] = work["_date"].dt.to_period("M")
        work["_rounded"] = work["amount"].round(0)
        recurring = []
        for (desc, amt), group in work.groupby(["description", "_rounded"]):
            months = group["_month"].nunique()
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
<div style="background:#1a1a2e; border-radius:8px; padding:12px 16px; margin-top:8px; color:white;">
    🔄 <strong>Total recurring spend:</strong>
    <span style="color:#E74C3C; font-size:1.1rem; font-weight:bold"> ${total_annual:,.0f}/year</span>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No recurring transactions detected.")
    else:
        st.info(no_date_msg)

    st.divider()

    # ---- CHART 4: Impulse Buy Flagging ----
    st.markdown("#### ⚡ Impulse Buy Flags")
    _igrp = "category_group" if "category_group" in df.columns else "category"
    impulse_rows = []
    for cat, group in df.groupby(_igrp):
        cat_mean = group["amount"].mean()
        flagged = group[group["amount"] > cat_mean * 3].copy()
        flagged["_multiple"] = (flagged["amount"] / cat_mean).round(1)
        flagged["_category"] = cat
        impulse_rows.append(flagged)

    if impulse_rows:
        all_flags = pd.concat(impulse_rows).sort_values("amount", ascending=False)
        date_map = (work.set_index(work.index)["_date"].dt.strftime("%b %d, %Y").to_dict()
                    if date_col_found else {})
        if all_flags.empty:
            st.info("No impulse purchases detected — impressive discipline!")
        else:
            for idx, row in all_flags.iterrows():
                date_str = date_map.get(idx, "")
                date_part = f" — {date_str}" if date_str else ""
                st.markdown(f"""
<div style="background:#111; border-left:4px solid #E67E22; border-radius:0 10px 10px 0;
padding:12px 16px; margin:6px 0;">
    ⚠️ <strong style="color:white">{row['description']}</strong>
    <span style="color:#E74C3C; font-weight:bold"> — ${row['amount']:,.2f}</span>
    <span style="color:#888">{date_part} — {row['_multiple']}x your usual {row['_category']} spend</span>
</div>""", unsafe_allow_html=True)
    else:
        st.info("No impulse purchases detected.")

    st.divider()

    # ---- CHART 5: Spending Velocity ----
    st.markdown("#### 📈 Spending velocity — when in the month do you spend?")
    if date_col_found:
        work["_dom"] = work["_date"].dt.day
        work["_month_label"] = work["_date"].dt.strftime("%b %Y")
        velocity = (work.groupby(["_month_label", "_dom"])["amount"]
                    .sum().reset_index().sort_values(["_month_label", "_dom"]))
        velocity["_cumulative"] = velocity.groupby("_month_label")["amount"].cumsum()
        fig5 = px.line(
            velocity,
            x="_dom", y="_cumulative",
            color="_month_label",
            title="Spending velocity — when in the month do you spend?",
            labels={"_dom": "Day of Month", "_cumulative": "Cumulative Spend ($)",
                    "_month_label": "Month"},
            markers=False,
        )
        fig5.update_layout(**DARK, title_x=0)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info(no_date_msg)


def render_finance_chat(df: pd.DataFrame):
    st.divider()
    st.subheader("💬 Ask Your Finances")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # Build transaction context for the system prompt
    _cgrp = "category_group" if "category_group" in df.columns else "category"
    cat_totals = df.groupby(_cgrp)["amount"].sum().round(2).to_dict()
    top_merchants = df.groupby("description")["amount"].sum().nlargest(10).round(2).to_dict()
    top_txns = df.nlargest(20, "amount")[["description", "amount", "category"]].to_dict("records")
    start_date = end_date = "unknown"
    monthly_totals: dict = {}
    for c in df.columns:
        if c in ("description", "amount", "category"):
            continue
        try:
            parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                start_date = parsed.min().strftime("%Y-%m-%d")
                end_date = parsed.max().strftime("%Y-%m-%d")
                _tmp = df.copy()
                _tmp["_month"] = parsed.dt.to_period("M").astype(str)
                monthly_totals = _tmp.groupby("_month")["amount"].sum().round(2).to_dict()
                break
        except Exception:
            pass

    system_prompt = (
        "You are a financial data analyst. You have access to the user's complete transaction data as a JSON summary below. "
        "Answer questions conversationally but precisely. "
        "Always include specific dollar amounts and dates from the data. "
        "Never make up numbers. If you cannot answer from the data, say so. "
        "Keep answers under 4 sentences unless a list is genuinely needed. "
        "Do not use markdown bold or asterisks.\n\n"
        f"Transaction summary (top 20 by amount): {json.dumps(top_txns)}\n"
        f"Category totals: {json.dumps(cat_totals)}\n"
        f"Monthly totals: {json.dumps(monthly_totals)}\n"
        f"Date range: {start_date} to {end_date}\n"
        f"Top 10 merchants by spend: {json.dumps(top_merchants)}"
    )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Suggestion chips — only shown before first message and when no pending question
    if not st.session_state.chat_history and not st.session_state.pending_question:
        suggestions = [
            "How much did I spend on food last month?",
            "What is my biggest single transaction?",
            "Which month was my most expensive?",
            "What are my top 5 merchants?",
        ]
        n_chip_cols = 2 if st.session_state.get("mobile_mode", False) else 4
        chip_cols = st.columns(n_chip_cols)
        for i, suggestion in enumerate(suggestions):
            with chip_cols[i % n_chip_cols]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Determine question to process: suggestion click or typed input
    user_question = st.chat_input("Ask anything about your spending...")
    question = None
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    elif user_question:
        question = user_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    client = get_anthropic_client()
                    response = client.messages.create(
                        model="claude-opus-4-7",
                        max_tokens=512,
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_history
                        ],
                    )
                    answer = response.content[0].text
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Could not get answer: {str(e)}")


# --- UI ---
st.title("💰 AI Financial Spending Judge")
st.markdown("Upload your bank statement to get AI-powered spending analysis and personalized insights.")

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
        print(f"[startup] Claude API OK — model=claude-haiku-4-5-20251001 response={_test_resp.content[0].text!r}")
    except SystemExit:
        pass  # st.stop() raises SystemExit — handled by st.error above
    except Exception as _e:
        _key = os.getenv("ANTHROPIC_API_KEY", "")
        print(f"[startup] Claude API FAILED: {type(_e).__name__}: {_e}")
        print(f"[startup] Key loaded: {'yes' if _key else 'NO — key is empty'}, starts with: {_key[:10]!r}")
        st.error(f"Claude API connection failed at startup: {type(_e).__name__}: {_e}")

# Sidebar controls
mobile_mode = st.sidebar.checkbox("📱 Mobile layout", value=False)
st.session_state["mobile_mode"] = mobile_mode

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
            raw = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
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
            sr_color = "#2ECC71" if savings_rate > 0.20 else "#F39C12" if savings_rate > 0.10 else "#E74C3C"
            sr_label = "Great 🌟" if savings_rate > 0.20 else "Fair 😐" if savings_rate > 0.10 else "Low ⚠️"
            n_metric_cols = 2 if mobile_mode else 3
            mc = st.columns(n_metric_cols)
            with mc[0]:
                st.metric("Total Income", f"${income_total:,.2f}")
            with mc[1]:
                st.metric("Total Expenses", f"${expense_total:,.2f}")
            with mc[2 % n_metric_cols]:
                st.markdown(f"""
<div style="background:#111; border-radius:10px; padding:14px; text-align:center; margin-top:4px;">
    <div style="color:#888; font-size:0.8rem">Savings Rate</div>
    <div style="color:{sr_color}; font-size:1.8rem; font-weight:bold">{savings_rate*100:.1f}%</div>
    <div style="color:{sr_color}; font-size:0.85rem">{sr_label}</div>
</div>""", unsafe_allow_html=True)

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
                        fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)

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
                                bar_color = "#2ECC71" if pct >= 0.8 else "#F39C12" if pct >= 0.5 else "#E74C3C"
                                status = "✅" if pct >= 0.8 else "⚠️" if pct >= 0.5 else "❌"
                                st.markdown(f"""
<div style="margin:10px 0;">
    <div style="display:flex; justify-content:space-between; color:white; font-size:0.9rem">
        <span>{status} {name}</span>
        <span style="color:{bar_color}">{comp_score}/{max_score}</span>
    </div>
    <div style="background:#333; border-radius:4px; height:6px; margin:4px 0;">
        <div style="background:{bar_color}; width:{pct*100:.0f}%; height:6px; border-radius:4px;"></div>
    </div>
    <div style="color:#666; font-size:0.75rem">{description}</div>
</div>""", unsafe_allow_html=True)
                            st.markdown("---")
                            with st.spinner("Generating what-if scenarios..."):
                                whatif = get_whatif_lines(df_for_analysis, components, score)
                            for line in whatif:
                                st.markdown(f"""
<div style="background:#1a1a2e; border-left:3px solid #3498DB; border-radius:0 8px 8px 0;
padding:10px 14px; margin:6px 0; color:#ccc; font-size:0.9rem;">
    💡 {line}
</div>""", unsafe_allow_html=True)

                        st.markdown("### Category Breakdown")
                        cat_stats = category_stats(df_for_analysis)
                        for cat, s in sorted(cat_stats.items(), key=lambda x: x[1]["total"], reverse=True):
                            cat_color = CATEGORY_COLORS.get(cat, "#C7B8EA")
                            if s["has_mom"]:
                                delta_color = "#2ECC71" if s["delta_pct"] >= 0 else "#E74C3C"
                                arrow = "↑" if s["delta_pct"] >= 0 else "↓"
                                mom_line = (
                                    f'<div style="color:{delta_color}; font-size:0.85rem; margin-top:4px">'
                                    f'{arrow} {abs(s["delta_pct"]):.1f}% vs {s["prev_label"]} '
                                    f'(${s["prev_amt"]:,.0f} → ${s["curr_amt"]:,.0f})</div>'
                                )
                            else:
                                mom_line = (
                                    f'<div style="color:#666; font-size:0.85rem; margin-top:4px">'
                                    f'{s["pct"]:.1f}% of total spending</div>'
                                )
                            st.markdown(f"""
<div style="background:#111; border-radius:12px; padding:16px; margin:8px 0;
border-left:4px solid {cat_color};">
    <div style="color:#888; font-size:0.85rem">{cat.capitalize()}</div>
    <div style="color:white; font-size:1.8rem; font-weight:bold">${s['total']:,.2f}</div>
    {mom_line}
    <div style="color:#666; font-size:0.8rem; margin-top:4px">
        Monthly avg: ${s['monthly_avg']:,.0f}
        &nbsp;|&nbsp;
        Largest: ${s['largest_amt']:,.0f} on {s['largest_date']}
    </div>
</div>""", unsafe_allow_html=True)

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
                    n_budget_cols = 2 if mobile_mode else 3
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
                    rem_color = "#2ECC71" if remaining >= 0 else "#E74C3C"
                    st.markdown(f"""
<div style="background:#111; border-radius:10px; padding:12px;
display:flex; justify-content:space-between; margin-top:8px">
    <span style="color:#888">Total budgeted:
        <strong style="color:white">{currency}{total_budgeted:,}</strong>
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
                                data = get_insights(df_for_analysis, cat_pcts, score, False)
                                render_insights(data, False)
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")

                # ===== ROAST TAB =====
                with tab_roast:
                    st.subheader("🔥 Roast Mode")
                    st.markdown("Let Claude brutally critique your spending habits.")
                    roast_level = st.select_slider(
                        "Roast Intensity",
                        options=["Mild 🌱", "Medium 🔥", "Gordon Ramsay 💀"],
                        value="Medium 🔥"
                    )
                    if st.button("Roast My Finances", type="primary", use_container_width=True,
                                 key="btn_roast"):
                        intensity_labels = {
                            "Mild 🌱": "Warming Up...",
                            "Medium 🔥": "The Roast is On...",
                            "Gordon Ramsay 💀": "BLOODY HELL, HERE WE GO...",
                        }
                        st.markdown(f"### 🔥 {intensity_labels.get(roast_level, 'The Roast is On...')}")
                        with st.spinner("Sharpening the knives..."):
                            try:
                                roast_data = get_roast(df_for_analysis, score, roast_level)
                                render_roast(roast_data)
                            except Exception as e:
                                st.error(f"Roast failed: {str(e)}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Hero section
    st.markdown("""
<div style="text-align:center; padding:32px 0 16px;">
    <div style="color:#888; font-size:1.1rem; max-width:600px; margin:0 auto; line-height:1.6">
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
         "Day-of-week heatmaps, merchant bubble charts, subscription detector, impulse buy flags, and spending velocity curves."),
    ]
    for col, icon, title, desc in feature_cards:
        with col:
            st.markdown(f"""
<div style="background:#111; border-radius:14px; padding:22px; text-align:center; height:180px;
border:1px solid #222; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px;">
    <div style="font-size:2.2rem">{icon}</div>
    <div style="color:white; font-weight:bold; font-size:1rem">{title}</div>
    <div style="color:#888; font-size:0.82rem; line-height:1.5">{desc}</div>
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
<div style="background:#111; border-radius:14px; padding:22px; text-align:center; height:180px;
border:1px solid #222; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px;">
    <div style="font-size:2.2rem">{icon}</div>
    <div style="color:white; font-weight:bold; font-size:1rem">{title}</div>
    <div style="color:#888; font-size:0.82rem; line-height:1.5">{desc}</div>
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
