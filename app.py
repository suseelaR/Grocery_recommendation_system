"""
🛒 Smart Grocery Helper — Customer-Friendly Version
Wishlist bug fixed: recommendations stored in session_state so
clicking Save does not wipe the results off screen.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from difflib import get_close_matches
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grocery Helper",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    [data-testid="metric-container"] {
        background: #f8fdf3;
        border: 1px solid #d4edba;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label        { color: #3B6D11 !important; font-size: 13px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 26px !important; }

    [data-testid="stSidebar"] { background: #f3faea; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PRODUCT ICONS
# ─────────────────────────────────────────────
PRODUCT_ICONS = {
    "whole milk": "🥛",      "yogurt": "🍶",           "butter": "🧈",
    "cream cheese": "🧀",    "cheese": "🧀",            "curd": "🥛",
    "eggs": "🥚",            "domestic eggs": "🥚",
    "bread": "🍞",           "rolls/buns": "🥖",        "white bread": "🍞",
    "brown bread": "🍞",     "pastry": "🥐",            "waffles": "🧇",
    "cake bar": "🎂",
    "beef": "🥩",            "pork": "🥩",              "chicken": "🍗",
    "sausage": "🌭",         "frankfurter": "🌭",       "ham": "🥩",
    "citrus fruit": "🍊",    "tropical fruit": "🍍",   "pip fruit": "🍎",
    "root vegetables": "🥕", "other vegetables": "🥦", "herbs": "🌿",
    "onions": "🧅",          "frozen vegetables": "❄️",
    "soda": "🥤",            "bottled water": "💧",     "coffee": "☕",
    "bottled beer": "🍺",    "canned beer": "🍺",
    "chocolate": "🍫",       "candy": "🍬",             "sugar": "🍬",
    "pasta": "🍝",           "rice": "🍚",              "oil": "🫙",
    "salt": "🧂",            "shopping bags": "🛍️",    "newspapers": "📰",
}

def get_icon(product: str) -> str:
    p = product.lower()
    for key, icon in PRODUCT_ICONS.items():
        if key in p or p in key:
            return icon
    return "🛒"

# ─────────────────────────────────────────────
# CUSTOMER-FRIENDLY LANGUAGE HELPERS
# ─────────────────────────────────────────────
def match_strength_label(lift: float) -> str:
    if lift >= 2.5:
        return "⭐⭐⭐ Perfect"
    elif lift >= 1.7:
        return "⭐⭐ Great"
    return "⭐ Okay"

def shopper_stat(confidence: float) -> str:
    return f"👥 {int(confidence * 100)}% picked this"

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading products...")
def load_data() -> pd.DataFrame:
    try:
        data = pd.read_csv("dataset/Groceries_dataset.csv")
    except FileNotFoundError:
        st.error("❌ Dataset not found at `dataset/Groceries_dataset.csv`.")
        st.stop()
    data.columns = data.columns.str.strip()
    data["Transaction"] = data["Member_number"].astype(str) + "_" + data["Date"].astype(str)
    data["Date"]      = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    data["Month"]     = data["Date"].dt.month_name()
    data["Month_num"] = data["Date"].dt.month
    return data

# ─────────────────────────────────────────────
# TRANSACTION ENCODING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Studying shopping patterns...")
def encode_transactions(data: pd.DataFrame):
    transactions = data.groupby("Transaction")["itemDescription"].apply(list).tolist()
    te       = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df       = pd.DataFrame(te_array, columns=te.columns_)
    return df, list(te.columns_)

# ─────────────────────────────────────────────
# RULE GENERATION
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Finding best product pairs...")
def generate_rules(df: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def recommend(products: list, rules: pd.DataFrame, top_n: int = 5) -> list:
    products_lower = [p.lower() for p in products]
    mask = (
        rules["antecedents"].apply(lambda x: any(p in [i.lower() for i in x] for p in products_lower)) |
        rules["consequents"].apply(lambda x: any(p in [i.lower() for i in x] for p in products_lower))
    )
    best = {}
    for _, row in rules[mask].iterrows():
        for item in list(row["antecedents"]) + list(row["consequents"]):
            if item.lower() not in products_lower:
                if item not in best or best[item]["lift"] < row["lift"]:
                    best[item] = {
                        "name":       item,
                        "confidence": float(row["confidence"]),
                        "lift":       float(row["lift"]),
                    }
    return sorted(best.values(), key=lambda x: x["lift"], reverse=True)[:top_n]

# ─────────────────────────────────────────────
# SESSION STATE  ← all persistent data lives here
# ─────────────────────────────────────────────
defaults = {
    "wishlist":        [],   # items the customer saved
    "search_history":  [],   # past searches
    "last_recs":       [],   # ← KEY FIX: last recommendation results
    "last_selected":   [],   # cart items used for last search
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data        = load_data()
df, items   = encode_transactions(data)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Smart Grocery Helper")
    st.markdown("*We suggest what other shoppers usually grab with your picks.*")
    st.markdown("---")

    page = st.radio(
        "Go to",
        ["🏠 Home", "🛍️ Get Suggestions", "📈 Popular Items", "❤️ My List", "ℹ️ How It Works"],
        key="nav",
    )

    st.markdown("---")
    st.markdown("**Settings**")

    popularity_level = st.select_slider(
        "Popularity",
        options=["All", "Popular", "Top picks"],
        value="Popular",
        help="'Top picks' shows only the most commonly bought items.",
    )
    suggestion_count = st.slider("Results", 3, 10, 5)

    support_map  = {"All": 0.003, "Popular": 0.005, "Top picks": 0.015}
    min_support  = support_map[popularity_level]
    min_confidence = 0.1

    st.markdown("---")
    wl_count = len(st.session_state.wishlist)
    st.markdown(f"❤️ **My list:** {wl_count} item{'s' if wl_count != 1 else ''}")

rules = generate_rules(df, min_support, min_confidence)

# ─────────────────────────────────────────────
# HELPER — save item to wishlist
# Called by every "❤️ Save" button via a callback
# so the state update happens BEFORE the rerun.
# ─────────────────────────────────────────────
def save_to_wishlist(item_name: str):
    if item_name not in st.session_state.wishlist:
        st.session_state.wishlist.append(item_name)

def remove_from_wishlist(item_name: str):
    if item_name in st.session_state.wishlist:
        st.session_state.wishlist.remove(item_name)

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 🛒 Welcome to Smart Grocery Helper!")
    st.markdown(
        "Tell us what's already in your cart and we'll suggest what other shoppers "
        "usually grab along with it — so you never forget anything!"
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧺 Products",  f"{len(items):,}")
    c2.metric("👥 Shoppers",  f"{data['Member_number'].nunique():,}")
    c3.metric("🛒 Trips",     f"{data['Transaction'].nunique():,}")
    c4.metric("🔗 Pairings",  f"{len(rules):,}")

    st.markdown("---")
    st.markdown("### How to use this app")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 1️⃣ Pick your items")
        st.markdown("Go to **Get Suggestions** and choose what's already in your cart.")
    with b:
        st.markdown("#### 2️⃣ See what goes with it")
        st.markdown("We'll show items other shoppers usually buy along with your picks.")
    with c:
        st.markdown("#### 3️⃣ Save what you need")
        st.markdown("Tap ❤️ on any item to add it to **My List** for easy reference.")

# ─────────────────────────────────────────────
# GET SUGGESTIONS  (wishlist bug fixed here)
# ─────────────────────────────────────────────
elif page == "🛍️ Get Suggestions":
    st.markdown("## 🛍️ What should I add to my cart?")
    st.markdown("Choose the items already in your cart and we'll suggest what goes well with them.")

    selected = st.multiselect(
        "Items already in my cart:",
        options=sorted(items),
        default=st.session_state.last_selected,   # restore previous selection
        placeholder="Start typing to search for a product...",
        key="cart_select",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        go = st.button("🔍 Show me suggestions", type="primary", use_container_width=True)

    # ── When the button is pressed: compute & store results ──
    if go:
        if not selected:
            st.warning("⚠️ Please choose at least one product from your cart first.")
            st.session_state.last_recs     = []
            st.session_state.last_selected = []
        elif rules.empty:
            st.info("We couldn't find suggestions. Try 'All' in the sidebar settings.")
            st.session_state.last_recs     = []
            st.session_state.last_selected = []
        else:
            recs = recommend(selected, rules, suggestion_count)
            # ← Store results so they survive the rerun caused by Save buttons
            st.session_state.last_recs     = recs
            st.session_state.last_selected = selected

            # Log search history
            st.session_state.search_history.insert(0, {
                "query":   selected,
                "results": [r["name"] for r in recs],
                "time":    datetime.now().strftime("%H:%M"),
            })

    # ── Always render whatever is stored — survives every rerun ──
    recs = st.session_state.last_recs

    if recs:
        st.success(
            f"🎉 We found **{len(recs)} item{'s' if len(recs) > 1 else ''}** "
            f"that shoppers usually buy with your picks!"
        )

        top_products = set(df.sum().sort_values(ascending=False).head(10).index.str.lower())

        for rec in recs:
            name       = rec["name"]
            icon       = get_icon(name)
            is_popular = name.lower() in top_products
            in_list    = name in st.session_state.wishlist

            c1, c2, c3 = st.columns([1, 7, 2])

            with c1:
                st.markdown(
                    f"<div style='font-size:42px;text-align:center;padding-top:8px'>{icon}</div>",
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(f"### {name}")
                if is_popular:
                    st.markdown(
                        "<span style='background:#EAF3DE;color:#27500A;font-size:12px;"
                        "padding:2px 9px;border-radius:10px;'>🔥 Trending</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                st.markdown(f"**{match_strength_label(rec['lift'])}**")
                st.caption(shopper_stat(rec["confidence"]))

            with c3:
                st.markdown("<div style='padding-top:14px'></div>", unsafe_allow_html=True)

                if in_list:
                    # Already saved — show a disabled confirmation instead of the button
                    st.success("✅ Saved!")
                else:
                    # Use on_click callback so the state updates BEFORE the rerun
                    # This keeps the recommendation list visible after clicking Save
                    st.button(
                        "❤️ Save",
                        key=f"save_{name}",
                        on_click=save_to_wishlist,
                        args=(name,),
                    )

            st.divider()

    elif st.session_state.last_selected and not recs:
        # Had a search but no results
        all_lower   = [i.lower() for i in items]
        suggestions = set()
        for p in st.session_state.last_selected:
            suggestions.update(get_close_matches(p.lower(), all_lower, n=2, cutoff=0.6))
        if suggestions:
            st.warning(f"😕 No matches found. Did you mean: **{', '.join(suggestions)}**?")
        else:
            st.warning(
                "😕 No matches found. Try different products, "
                "or set **Popularity** to **All** in the sidebar."
            )

# ─────────────────────────────────────────────
# POPULAR ITEMS
# ─────────────────────────────────────────────
elif page == "📈 Popular Items":
    st.markdown("## 📈 What do most shoppers buy?")
    st.markdown("These are the items that appear most often in shoppers' carts.")

    item_counts = df.sum().sort_values(ascending=False).head(15).reset_index()
    item_counts.columns = ["Product", "Number of shoppers who bought it"]
    item_counts["Label"] = item_counts["Product"].apply(lambda p: f"{get_icon(p)} {p}")

    fig = px.bar(
        item_counts,
        x="Number of shoppers who bought it",
        y="Label",
        orientation="h",
        title="Top 15 most popular products",
        color="Number of shoppers who bought it",
        color_continuous_scale=["#C0DD97", "#3B6D11"],
        text="Number of shoppers who bought it",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed", title=""),
        xaxis_title="Number of shoppers who bought it",
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=20, t=40, b=0),
        height=460,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗓️ Shopping activity by month")
    monthly = (
        data.dropna(subset=["Month_num"])
        .groupby(["Month_num", "Month"])
        .size()
        .reset_index(name="Shopping trips")
        .sort_values("Month_num")
    )
    fig2 = px.line(
        monthly, x="Month", y="Shopping trips",
        title="How busy each month is",
        markers=True,
        color_discrete_sequence=["#3B6D11"],
    )
    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of shopping trips",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MY LIST  (wishlist + history)
# ─────────────────────────────────────────────
elif page == "❤️ My List":
    st.markdown("## ❤️ My Shopping List")

    if not st.session_state.wishlist:
        st.info("Your list is empty. Go to **Get Suggestions**, find items you like, and tap ❤️ Save to my list.")
    else:
        col_info, col_clear = st.columns([4, 1])
        col_info.markdown(f"You have **{len(st.session_state.wishlist)} item(s)** saved:")

        if col_clear.button("🗑️ Clear all"):
            st.session_state.wishlist = []
            st.rerun()

        for item in list(st.session_state.wishlist):   # iterate a copy so removal is safe
            c1, c2, c3 = st.columns([1, 7, 1])
            c1.markdown(
                f"<div style='font-size:24px;text-align:center'>{get_icon(item)}</div>",
                unsafe_allow_html=True,
            )
            c2.markdown(f"**{item}**")
            # Use on_click callback for removal too — avoids double-rerun
            c3.button(
                "✕",
                key=f"del_{item}",
                help=f"Remove {item}",
                on_click=remove_from_wishlist,
                args=(item,),
            )

    st.divider()
    st.markdown("### 🕐 Your recent searches")

    if not st.session_state.search_history:
        st.info("You haven't searched for anything yet.")
    else:
        for entry in st.session_state.search_history[:5]:
            picked    = ", ".join(entry["query"])
            suggested = ", ".join(entry["results"][:3])
            more      = "..." if len(entry["results"]) > 3 else ""
            st.markdown(
                f"🛒 **You had:** {picked} &nbsp;→&nbsp; "
                f"**We suggested:** {suggested}{more} &nbsp;"
                f"<span style='color:#aaa;font-size:12px;'>{entry['time']}</span>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────
elif page == "ℹ️ How It Works":
    st.markdown("## ℹ️ How does this app know what to suggest?")
    st.markdown("""
We studied the shopping habits of thousands of real customers.

By looking at what people buy **together** in the same shopping trip, we found patterns —
like *"people who buy herbs almost always also buy root vegetables."*

We use those patterns to suggest items you might have forgotten, based on what's already in your cart.
""")

    st.markdown("---")
    st.markdown("### What do the labels mean?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### ⭐⭐⭐ Perfect")
        st.markdown("Shoppers who bought your item almost *always* bought this too. Highly recommended!")
    with col2:
        st.markdown("#### ⭐⭐ Great")
        st.markdown("Many shoppers bought this along with your item. Worth adding to your cart.")
    with col3:
        st.markdown("#### ⭐ Okay")
        st.markdown("Some shoppers bought this with your item. Could be useful to consider.")

    st.markdown("---")
    st.info(
        "🌿 If you pick **herbs**, a very large number of shoppers also bought "
        "**🥕 root vegetables** in the same trip — so we label it **⭐⭐⭐ Perfect**.\n\n"
        "It's like having a friend who remembers every shopping trip ever made "
        "and gives you a heads-up on what you might need!"
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("🛒 Trips",    f"{data['Transaction'].nunique():,}")
    c2.metric("🔗 Pairings", f"{len(rules):,}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:12px;'>"
    "🛒 Smart Grocery Helper &nbsp;·&nbsp; "
    f"{data['Transaction'].nunique():,} trips studied &nbsp;·&nbsp; "
    f"{len(items)} products"
    "</div>",
    unsafe_allow_html=True,
)
