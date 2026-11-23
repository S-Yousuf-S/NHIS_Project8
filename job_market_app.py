"""
================================================================================
 JOB_MARKET_APP.PY

 Project      : NHIS Project 8 — Job Market Analysis and Recommendation System
 Author       : Yousuf S. R. Sakkaf
 Description  : JobLens — a job portal built on Upwork postings. A job seeker
                types what they are looking for, picks a country and gets the best
                matching current listings, with filters for contract type, job
                category, remote work and stated pay. A second page, Market
                Insights, holds the monthly-updating market dashboard, pay insights,
                demand trends, forecasts and scenarios for recruiters and analysts.
                Everything is loaded from the two files saved by the notebook
                (recommender and dashboard tables); nothing is retrained here.
 Run          : streamlit run job_market_app.py
================================================================================
"""

import base64
import glob
import html
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
ASSETS_DIR = "deploy_assets"
MODEL_DIRS = ["Assets/models", os.path.join(ASSETS_DIR, "models"), ASSETS_DIR]    # first folder holding a saved file wins
LOGO_FILE = os.path.join(ASSETS_DIR, "logo_wordmark.png")
LOGO_WHITE_FILE = os.path.join(ASSETS_DIR, "logo_wordmark_white.png")
ICON_FILE = os.path.join(ASSETS_DIR, "logo_icon.png")
STATUS_WORDS = {"Within earlier range": "normal", "Above earlier range": "higher than usual", "Below earlier range": "lower than usual"}
PAGE_SIZE = 10                  # jobs shown at first and added by each "Show more"
CANDIDATES = 100                # best matches fetched before the filters are applied
NEW_DAYS = 3                    # listings from the last days of the data get a "New" badge

MAP_SCALE = ["#2563EB", "#06B6D4", "#FDE047", "#F59E0B", "#DC2626"]    # blue (low) → cyan → yellow → amber → red (high)

# Same palette as the notebook: teal for structure, amber as the accent
TEAL, TEAL_DARK, TEAL_DEEP, TEAL_LIGHT, MINT = "#008080", "#006D6D", "#134E4A", "#14B8A6", "#E0F2F1"
AMBER, AMBER_DARK, AMBER_LIGHT = "#F59E0B", "#D97706", "#FEF3C7"
SUCCESS_TEXT, SUCCESS_BG, WARNING, WARNING_TEXT, WARNING_BG = "#047857", "#ECFDF5", "#EF4444", "#B91C1C", "#FEF2F2"
SKY, SKY_LIGHT = "#1E40AF", "#DBEAFE"
NEUTRAL, INK, LINE = "#78716C", "#1C2B2A", "#CDE7E4"
CONTRACT_COLORS = {"Freelancer": TEAL_DARK, "Permanent": AMBER}

POPULAR_SEARCHES = ["Website developer", "Virtual assistant", "Graphic designer", "Content writer", "Video editor", "Data entry"]

st.set_page_config(page_title="JobLens — find jobs and track the job market", page_icon=ICON_FILE if os.path.exists(ICON_FILE) else "💼",
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"], button, textarea, input {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #F6FBFA; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 4rem; padding-bottom: 2rem; max-width: 1180px; }   /* leaves room for Streamlit's own header so the page switch stays clickable */
footer { visibility: hidden; }
[data-testid="stAppDeployButton"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, %%TEAL_DEEP%% 0%, %%TEAL_DARK%% 100%); }
[data-testid="stSidebar"] * { color: #E6F7F5; }
.sb-brand img { height: 34px; }
.sb-sec { margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,.18); }
.sb-sec h4 { margin: 0 0 8px 0; font-size: .9rem; letter-spacing: .04em; color:#FFFFFF; }
.sb-sec p, .sb-sec li { font-size: .88rem; line-height: 1.5; margin: 0 0 6px 0; color:#CFEDEA; }
.sb-sec ol { padding-left: 18px; margin: 0; }
.sb-stat { background: rgba(255,255,255,.10); border-radius: 12px; padding: 10px 12px; font-size:.88rem; color:#FFFFFF; }

/* Top bar */
.brand img { height: 40px; display:block; }
.brand span { display:block; font-size:.78rem; color:%%NEUTRAL%%; margin-top:2px; }

/* Hero */
.hero { position:relative; overflow:hidden; border-radius: 22px; padding: 38px 40px 92px 40px; color:#fff; margin-top: 6px;
        background: linear-gradient(120deg, %%TEAL_DEEP%% 0%, %%TEAL%% 62%, %%TEAL_LIGHT%% 100%); }
.hero::before { content:""; position:absolute; right:-70px; top:-90px; width:300px; height:300px; border-radius:50%;
        background: radial-gradient(circle, rgba(245,158,11,.95) 0%, rgba(245,158,11,.55) 45%, rgba(245,158,11,0) 70%); }
.hero::after { content:""; position:absolute; right:150px; bottom:-60px; width:190px; height:190px; border-radius:50%; border:26px solid rgba(255,255,255,.10); }
.hero h1 { position:relative; margin:0; font-size: 2.5rem; font-weight:800; color:#fff; letter-spacing:-.02em; line-height:1.12; }
.hero h1 b { color: %%AMBER%%; }
.hero p { position:relative; margin:10px 0 0 0; font-size:1.08rem; color:#DDF6F3; max-width: 640px; }
.hero .stats { position:relative; margin-top:16px; display:flex; gap:22px; flex-wrap:wrap; }
.hero .stats div { font-size:.88rem; color:#CFF3EE; } .hero .stats b { font-size:1.15rem; color:#fff; margin-right:4px; }

/* Search bar: the form sits over the hero edge like a portal search box */
[data-testid="stForm"] { position:relative; z-index:5; margin: -64px 22px 0 22px; padding: 14px 16px !important; background:#fff !important;
        border: 1px solid %%LINE%% !important; border-radius: 18px !important; box-shadow: 0 14px 34px rgba(19,78,74,.16); }
[data-testid="stForm"] input { font-size: 1.02rem !important; }
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 12px !important; }

/* Buttons */
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, %%AMBER%%, #FBBF24); border: none; color: %%TEAL_DEEP%%; font-weight: 800; border-radius: 12px; font-size: 1.02rem; }
[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover { filter: brightness(.96); color: %%TEAL_DEEP%%; }
[data-testid="stBaseButton-secondary"] { border-radius: 999px; border: 1px solid #A7D8D3; background: #fff; color: %%TEAL_DARK%%; font-weight: 600; font-size: .85rem; padding: .2rem .8rem; }
[data-testid="stBaseButton-secondary"]:hover { border-color: %%AMBER%%; background: %%AMBER_LIGHT%%; color: %%TEAL_DEEP%%; }

/* Cards and notes */
.section-title { font-size:1.12rem; font-weight:800; color:%%TEAL_DEEP%%; margin: 0 0 4px 0; }
.sub { font-size:.9rem; color:%%NEUTRAL%%; margin: 0 0 10px 0; }
.st-key-filters, [class*="st-key-card_"] { background:#fff; border:1px solid %%LINE%%; border-radius:16px; padding: 0; }
.st-key-filters { padding: 10px 14px 8px; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] { padding: 0 14px 12px; }
[class*="st-key-card_"] { margin-bottom: 2px; transition: box-shadow .15s, border-color .15s; }
[class*="st-key-card_"]:hover { border-color:%%TEAL_LIGHT%%; box-shadow: 0 8px 22px rgba(0,128,128,.12); }
[data-testid="stBaseLinkButton-secondary"] { background:%%AMBER%%; border:none; color:%%TEAL_DEEP%% !important; font-weight:800; border-radius:10px; }
[data-testid="stBaseLinkButton-secondary"]:hover { background:%%AMBER_DARK%%; color:#fff !important; }

/* Page switch (top right) */
.st-key-page { display:flex; justify-content:flex-end; }
.st-key-page [role="radiogroup"] { width:fit-content; gap:4px; background:#fff; border:1px solid %%LINE%%; border-radius:999px; padding:4px; }
.st-key-page [role="radiogroup"] label { padding:7px 20px; border-radius:999px; margin:0; cursor:pointer; }
.st-key-page [role="radiogroup"] label > div:first-child { display:none; }
.st-key-page [role="radiogroup"] label p { font-weight:700; font-size:.95rem; color:%%NEUTRAL%%; }
.st-key-page [role="radiogroup"] label:has(input:checked) { background:%%TEAL%%; }
.st-key-page [role="radiogroup"] label:has(input:checked) p { color:#fff; }
.note { font-size:.9rem; color:%%TEAL_DARK%%; background:#F0FDFA; border-left:5px solid %%TEAL%%; border-radius:8px; padding:8px 12px; margin:8px 0; }
.warn { font-size:.9rem; color:%%WARNING_TEXT%%; background:%%WARNING_BG%%; border-left:5px solid %%WARNING%%; border-radius:8px; padding:8px 12px; margin:8px 0; }
.kpi { background:#fff; border:1px solid %%LINE%%; border-left:6px solid %%TEAL%%; border-radius:14px; padding:12px 16px; height:100%; }
.kpi.amber { border-left-color:%%AMBER%%; } .kpi.ok { border-left-color:#10B981; } .kpi.watch { border-left-color:%%AMBER%%; }
.kpi .l { font-size:.82rem; font-weight:600; color:%%NEUTRAL%%; } .kpi .v { font-size:1.6rem; font-weight:800; color:%%TEAL_DEEP%%; line-height:1.2; }
.kpi .s { font-size:.8rem; color:%%NEUTRAL%%; margin-top:2px; }
.up { color:%%SUCCESS_TEXT%%; font-weight:700; } .down { color:%%WARNING_TEXT%%; font-weight:700; }

/* Job cards */
.job { padding: 14px 16px 0; }
.job .top { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.job .name { font-size:1.1rem; font-weight:700; color:%%TEAL_DARK%%; line-height:1.3; }
.job .badges { display:flex; gap:6px; align-items:center; white-space:nowrap; }
.badge-new { background:%%AMBER%%; color:%%TEAL_DEEP%%; font-size:.7rem; font-weight:800; border-radius:6px; padding:2px 8px; letter-spacing:.04em; }
.badge-match { font-size:.8rem; font-weight:700; color:%%SUCCESS_TEXT%%; background:%%SUCCESS_BG%%; border-radius:999px; padding:2px 10px; }
.job .meta { margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.tag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }
.tag.cat { background:%%MINT%%; color:%%TEAL_DARK%%; } .tag.free { background:#CCFBF1; color:#115E59; }
.tag.perm { background:%%AMBER_LIGHT%%; color:#92400E; } .tag.remote { background:%%SKY_LIGHT%%; color:%%SKY%%; }
.job .pay { margin-top:10px; font-size:1.02rem; font-weight:800; color:%%TEAL_DEEP%%; } .job .pay small { font-weight:600; color:%%NEUTRAL%%; font-size:.8rem; }
.job .pay.none { font-weight:500; color:%%NEUTRAL%%; font-size:.92rem; }
.job .foot { margin-top:12px; display:flex; justify-content:space-between; align-items:center; font-size:.82rem; color:%%NEUTRAL%%; }
.job a.apply { background:%%AMBER%%; color:%%TEAL_DEEP%%; font-weight:800; text-decoration:none; border-radius:10px; padding:7px 16px; font-size:.88rem; }
.job a.apply:hover { background:%%AMBER_DARK%%; color:#fff; }
.empty { text-align:center; padding: 36px 12px; color:%%NEUTRAL%%; }
.foot-note { text-align:center; color:%%NEUTRAL%%; font-size:.85rem; margin-top: 26px; }
button[data-baseweb="tab"] { font-weight:600; font-size:1rem; }
</style>
"""
for token, value in {"TEAL": TEAL, "TEAL_DARK": TEAL_DARK, "TEAL_DEEP": TEAL_DEEP, "TEAL_LIGHT": TEAL_LIGHT, "MINT": MINT, "AMBER": AMBER, "AMBER_DARK": AMBER_DARK,
                     "AMBER_LIGHT": AMBER_LIGHT, "SUCCESS_TEXT": SUCCESS_TEXT, "SUCCESS_BG": SUCCESS_BG, "WARNING": WARNING, "WARNING_TEXT": WARNING_TEXT,
                     "WARNING_BG": WARNING_BG, "SKY": SKY, "SKY_LIGHT": SKY_LIGHT, "NEUTRAL": NEUTRAL, "LINE": LINE}.items():
    CSS = CSS.replace(f"%%{token}%%", value)
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def flat(markup):
    """Remove line indentation so Markdown never mistakes indented HTML for a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def image_b64(path):
    try:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode()
    except OSError:
        return ""


def logo_tag(path, alt="JobLens"):
    encoded = image_b64(path)
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}">' if encoded else f"<b>{alt}</b>"


def latest_file(name):
    """Newest saved file for `name` (the notebook stamps each save with the time)."""
    for folder in MODEL_DIRS:
        files = sorted(glob.glob(os.path.join(folder, f"{name}_*.pkl")))
        if files:
            return files[-1]
    return None


@st.cache_resource(show_spinner="Loading saved results…")
def load_pickle(path):
    return joblib.load(path)


def note(message, kind="note"):
    st.markdown(f'<div class="{kind}">{message}</div>', unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(flat(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'),
                unsafe_allow_html=True)


def style_fig(fig, title, kind="line", height=420):
    """Chart house style: deep-teal bold title, dashed light grid (bars one direction only, lines both)."""
    fig.update_layout(title=dict(text=title, font=dict(size=20, color=TEAL_DEEP), x=0.0), height=height, margin=dict(l=10, r=10, t=60, b=10),
                      paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=-0.15), font=dict(color=INK))
    grid = dict(showgrid=True, gridcolor="rgba(120,113,108,0.25)", griddash="dash", zeroline=False)
    off = dict(showgrid=False)
    if kind == "bar":
        fig.update_xaxes(**off)
        fig.update_yaxes(**grid)
    elif kind == "hbar":
        fig.update_xaxes(**grid)
        fig.update_yaxes(**off)
    else:
        fig.update_xaxes(**grid)
        fig.update_yaxes(**grid)
    return fig


def show_table(frame, formats=None, height=None):
    """Interactive table with numbers formatted per column ({column: printf format})."""
    config = {column: st.column_config.NumberColumn(format=pattern) for column, pattern in (formats or {}).items() if column in frame.columns}
    st.dataframe(frame, hide_index=True, use_container_width=True, column_config=config, height=height)


# --- Recommender: identical to the notebook's recommend() --------------------------
RESULT_COLUMNS = ["Rank", "Title", "Job Category", "Contract Type", "Country", "Remote", "Pay Type", "Pay (units)", "Similar Postings", "Match (%)", "Posted", "Link"]


def recommend(query, country, contract_type=None, top_n=5, model=None):
    """Best-matching current listings for a job seeker's text, from one country and, optionally, one contract type."""
    listings = model["listings"]
    allowed = (listings["country"] == country).to_numpy()
    if contract_type:
        allowed = allowed & (listings["worker_type"] == contract_type).to_numpy()
    candidates = np.flatnonzero(allowed)
    if len(candidates) == 0 or not query.strip():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    scores = sum((model["vectorizers"][part].transform([query]) @ model["matrices"][part][candidates].T).toarray()[0] for part in model["parts"]) / len(model["parts"])
    order = np.argsort(-scores)[:top_n]
    order = order[scores[order] > 0]                                                  # drop listings with nothing in common with the text
    found = listings.iloc[candidates[order]]
    return pd.DataFrame({"Rank": np.arange(1, len(found) + 1), "Title": found["title"].to_numpy(), "Job Category": found["job_category"].to_numpy(),
                         "Contract Type": found["worker_type"].to_numpy(), "Country": found["country"].to_numpy(), "Remote": np.where(found["remote_posting"], "Yes", "No"),
                         "Pay Type": np.where(found["worker_type"] == "Freelancer", "Hourly rate", "Salary"), "Pay (units)": found["pay"].to_numpy(),
                         "Similar Postings": found["similar_postings"].to_numpy(), "Match (%)": scores[order] * 100,
                         "Posted": found["posted_day"].to_numpy(), "Link": found["link"].to_numpy()})


def latest_listings(model, country, contract_type=None):
    """Newest listings of a country, in the same table layout as recommend() (used before anything is searched)."""
    listings = model["listings"]
    mask = (listings["country"] == country).to_numpy()
    if contract_type:
        mask = mask & (listings["worker_type"] == contract_type).to_numpy()
    found = listings.loc[mask].sort_values(["posted_day", "similar_postings"], ascending=False)
    return pd.DataFrame({"Rank": np.arange(1, len(found) + 1), "Title": found["title"].to_numpy(), "Job Category": found["job_category"].to_numpy(),
                         "Contract Type": found["worker_type"].to_numpy(), "Country": found["country"].to_numpy(), "Remote": np.where(found["remote_posting"], "Yes", "No"),
                         "Pay Type": np.where(found["worker_type"] == "Freelancer", "Hourly rate", "Salary"), "Pay (units)": found["pay"].to_numpy(),
                         "Similar Postings": found["similar_postings"].to_numpy(), "Match (%)": np.nan,
                         "Posted": found["posted_day"].to_numpy(), "Link": found["link"].to_numpy()})


def pay_text(row):
    if pd.isna(row["Pay (units)"]):
        return "Pay not stated"
    return f'{row["Pay (units)"]:,.2f} units per hour' if row["Pay Type"] == "Hourly rate" else f'{row["Pay (units)"]:,.2f} units salary'


def toggle_saved(item):
    """Add the job to the saved list of this visit, or remove it if it is already there."""
    saved = st.session_state.setdefault("saved", {})
    if item["Link"] in saved:
        del saved[item["Link"]]
    else:
        saved[item["Link"]] = item


def render_card(row, position, newest_day, show_match):
    """One listing as a portal-style card: details on top, Save and View buttons underneath."""
    if pd.isna(row["Pay (units)"]):
        pay = '<div class="pay none">Pay not stated</div>'
    elif row["Pay Type"] == "Hourly rate":
        pay = f'<div class="pay">{row["Pay (units)"]:,.2f} units <small>per hour</small></div>'
    else:
        pay = f'<div class="pay">{row["Pay (units)"]:,.2f} units <small>salary</small></div>'
    contract_class = "free" if row["Contract Type"] == "Freelancer" else "perm"
    remote = '<span class="tag remote">Remote</span>' if row["Remote"] == "Yes" else ""
    posted = pd.Timestamp(row["Posted"])
    is_new = newest_day is not None and posted >= newest_day - pd.Timedelta(days=NEW_DAYS - 1)
    badges = ('<span class="badge-new">NEW</span>' if is_new else "") + (f'<span class="badge-match">{row["Match (%)"]:.0f}% match</span>' if show_match else "")
    similar = int(row["Similar Postings"])
    repeats = f" · {similar} similar postings" if similar > 1 else ""
    link = str(row["Link"])
    item = {"Title": str(row["Title"]), "Job Category": str(row["Job Category"]), "Contract Type": str(row["Contract Type"]), "Pay": pay_text(row), "Link": link}
    with st.container(key=f"card_{position}"):
        st.markdown(flat(f"""
        <div class="job">
          <div class="top"><div class="name">{html.escape(str(row['Title']))}</div><div class="badges">{badges}</div></div>
          <div class="meta"><span class="tag cat">{html.escape(str(row['Job Category']))}</span><span class="tag {contract_class}">{row['Contract Type']}</span>{remote}</div>
          {pay}
        </div>"""), unsafe_allow_html=True)
        info_col, save_col, view_col = st.columns([2.6, 1, 1.5], vertical_alignment="center")
        info_col.markdown(f'<span style="font-size:.82rem;color:{NEUTRAL};">Posted {posted:%d %b %Y}{repeats}</span>', unsafe_allow_html=True)
        is_saved = link in st.session_state.get("saved", {})
        save_col.button("★ Saved" if is_saved else "☆ Save", key=f"save_{position}", on_click=toggle_saved, args=(item,), use_container_width=True)
        if link.startswith("http"):
            view_col.link_button("View on Upwork ↗", link, use_container_width=True)


# ------------------------------------------------------------------------------
# Load the saved files
# ------------------------------------------------------------------------------
recommender_path, dashboard_path = latest_file("recommender"), latest_file("dashboard_tables")
if not recommender_path or not dashboard_path:
    st.markdown(f'<div class="brand">{logo_tag(LOGO_FILE)}</div>', unsafe_allow_html=True)
    st.error("The saved files were not found. Run the notebook to create `recommender_*.pkl` and `dashboard_tables_*.pkl`, "
             "then place both in `Assets/models` next to this app.")
    st.stop()

recommender = load_pickle(recommender_path)
bundle = load_pickle(dashboard_path)
T, META = bundle["tables"], bundle["meta"]
listings = recommender["listings"]
PERIOD = META.get("period", "Week")
NEWEST_DAY = pd.Timestamp(listings["posted_day"].max()) if len(listings) else None
COUNTRY_COUNTS = listings["country"].value_counts()
COUNTRIES = [name for name in COUNTRY_COUNTS.index if name != "Others"] + (["Others"] if "Others" in COUNTRY_COUNTS.index else [])


def country_label(name):
    return "Others (country not stated)" if name == "Others" else name


# ------------------------------------------------------------------------------
# Sidebar (project information; collapsed by default so the portal stays clean)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(flat(f"""
    <div class="sb-brand">{logo_tag(LOGO_WHITE_FILE)}</div>

    <div class="sb-sec"><h4>ABOUT THE PROJECT</h4>
      <p>JobLens turns thousands of Upwork postings into a job finder and a market tracker: search listings by country,
      then see what is in demand, what it pays and where the market is heading.</p></div>

    <div class="sb-sec"><h4>HOW TO USE IT</h4>
      <ol><li><b>Find jobs:</b> type what you are looking for, choose your country and press <b>Find jobs</b>.</li>
          <li>Narrow the list with the filters on the left.</li>
          <li><b>Market insights:</b> switch at the top for pay, demand, forecasts and the weekly dashboard.</li></ol></div>

    <div class="sb-sec"><h4>GOOD TO KNOW</h4>
      <p>• Pay is in <b>units</b> and is never converted.<br>• Hourly rates (Freelancer) and salaries (Permanent) are never mixed.<br>
      • Pay is shown only when the client stated it.<br>• Country is the <b>client's</b> country.<br>• Remote is detected from the title only.</p></div>

    <div class="sb-sec"><h4>DATA</h4>
      <div class="sb-stat">Postings up to <b>{META.get('data_through', 'n/a')}</b><br><b>{len(listings):,}</b> current listings</div></div>

    <div class="sb-sec"><h4>PROJECT</h4>
      <p>NHIS Project 8 — Job Market Analysis and Recommendation System<br>Built by Yousuf S. R. Sakkaf<br>Made with Python and Streamlit.</p></div>
    """), unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Top bar: brand on the left, page switch on the right
# ------------------------------------------------------------------------------
bar_left, bar_right = st.columns([1.2, 1])
with bar_left:
    st.markdown(f'<div class="brand">{logo_tag(LOGO_FILE)}<span>Find jobs. Understand the market.</span></div>', unsafe_allow_html=True)
with bar_right:
    page = st.radio("Page", ["Find jobs", "Market insights"], horizontal=True, key="page", label_visibility="collapsed")


# ------------------------------------------------------------------------------
# Page 1 — Job portal (Task 5)
# ------------------------------------------------------------------------------
def start_search(query, country):
    st.session_state["search"] = {"query": query.strip(), "country": country}
    st.session_state["show_n"] = PAGE_SIZE


def pick_popular(text):
    """Popular-search chip: fill the search box and run the search."""
    st.session_state["q_input"] = text
    start_search(text, st.session_state.get("country_input", COUNTRIES[0]))


def render_portal():
    st.markdown(flat(f"""
    <div class="hero"><h1>Find your next <b>job</b></h1>
      <p>Search current Upwork listings in your country — freelance and permanent — and see the pay when the client stated it.</p>
      <div class="stats"><div><b>{len(listings):,}</b>current listings</div><div><b>{listings['country'].nunique()}</b>countries</div>
      <div><b>{(listings['worker_type'] == 'Freelancer').sum():,}</b>freelance</div><div><b>{(listings['worker_type'] == 'Permanent').sum():,}</b>permanent</div></div>
    </div>"""), unsafe_allow_html=True)

    with st.form("search_form", border=False):
        what_col, where_col, go_col = st.columns([3.2, 2, 1.1], vertical_alignment="bottom")
        with what_col:
            query_input = st.text_input("What", key="q_input", max_chars=120, placeholder="Job title or keywords, e.g. wordpress developer")
        with where_col:
            country_input = st.selectbox("Where", COUNTRIES, key="country_input", format_func=country_label)
        with go_col:
            submitted = st.form_submit_button("Find jobs", type="primary", use_container_width=True)
    if submitted:
        start_search(query_input, country_input)

    chip_cols = st.columns([0.9] + [1] * len(POPULAR_SEARCHES), vertical_alignment="center")
    chip_cols[0].markdown('<p class="sub" style="margin:0;text-align:right;">Popular:</p>', unsafe_allow_html=True)
    for column, text in zip(chip_cols[1:], POPULAR_SEARCHES):
        column.button(text, key=f"chip_{text}", on_click=pick_popular, args=(text,), use_container_width=True)

    active = st.session_state.get("search", {"query": "", "country": COUNTRIES[0]})
    query, country = active["query"], active["country"]
    in_country = listings[listings["country"] == country]

    filter_col, result_col = st.columns([1, 2.6], gap="large")
    with filter_col:
        with st.container(key="filters"):
            st.markdown('<p class="section-title">Filters</p>', unsafe_allow_html=True)
            contract = st.radio("Contract type", ["All", "Freelancer", "Permanent"], key="f_contract")
            category_counts = in_country["job_category"].value_counts()
            category = st.selectbox("Job category", ["All categories"] + list(category_counts.index), key=f"f_category_{country}",
                                    format_func=lambda name: name if name == "All categories" else f"{name} ({category_counts[name]:,})")
            remote_only = st.checkbox("Remote only", key="f_remote")
            pay_only = st.checkbox("Pay stated only", key="f_pay")

    contract_filter = None if contract == "All" else contract
    if query:
        view = recommend(query, country, contract_filter, CANDIDATES, recommender)
    else:
        view = latest_listings(recommender, country, contract_filter)
    if category != "All categories":
        view = view[view["Job Category"] == category]
    if remote_only:
        view = view[view["Remote"] == "Yes"]
    if pay_only:
        view = view[view["Pay (units)"].notna()]

    with result_col:
        if query:
            st.markdown(f'<p class="section-title">{len(view):,} best matches for “{html.escape(query)}” in {html.escape(country_label(country))}</p>'
                        f'<p class="sub">Ranked by how closely the title matches your words. Only listings from this country are searched.</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="section-title">Latest jobs in {html.escape(country_label(country))}</p>'
                        f'<p class="sub">Type what you are looking for above to see the best matches first.</p>', unsafe_allow_html=True)
        if view.empty:
            note("No listings match. Try other words, remove a filter or choose another country." if query or category != "All categories" or remote_only or pay_only
                 else f"There are no current listings for {html.escape(country_label(country))}.", "warn")
        else:
            shown = st.session_state.get("show_n", PAGE_SIZE)
            saved = st.session_state.get("saved", {})
            if saved:
                with st.expander(f"★ Saved jobs ({len(saved)})"):
                    for item in saved.values():
                        st.markdown(f"[{item['Title']}]({item['Link']}) · {item['Job Category']} · {item['Contract Type']} · {item['Pay']}")
                    st.download_button("Download saved jobs (CSV)", pd.DataFrame(saved.values()).to_csv(index=False), "joblens_saved_jobs.csv", "text/csv")
                    st.button("Clear saved jobs", key="clear_saved", on_click=lambda: st.session_state.update(saved={}))
            for position, (_, row) in enumerate(view.head(shown).iterrows()):
                render_card(row, position, NEWEST_DAY, show_match=bool(query))
            if len(view) > shown:
                st.button(f"Show more ({len(view) - shown:,} more)", key="show_more", on_click=lambda: st.session_state.update(show_n=shown + PAGE_SIZE))
            st.caption(f"Listings come from postings up to {META.get('data_through', 'the latest data')}. Upwork may send a closed posting to its home page, so a link that no longer opens the job means it has been filled or removed.")
            st.download_button("Download these results (CSV)", view.head(max(shown, PAGE_SIZE)).drop(columns=["Rank"]).to_csv(index=False), "joblens_results.csv", "text/csv")

    with st.expander("How the matching works"):
        st.markdown("Titles of the current listings (the last 14 days) are turned into word and letter patterns (TF-IDF) and compared with your text by cosine "
                    f"similarity, inside the chosen country. Three methods were tested by hiding one word of 2,000 real titles and checking that the original came back; "
                    f"**{recommender.get('retriever', 'Word + Character')}** is used.")
        if "retriever_comparison" in T:
            show_table(T["retriever_comparison"], {"Recall@10": "%.3f", "MRR": "%.3f", "Category Match@10": "%.3f"})


# ------------------------------------------------------------------------------
# Page 2 — Market insights (Tasks 1 to 4, 6 to 8)
# ------------------------------------------------------------------------------
def render_insights():
    st.markdown(flat("""
    <div class="hero" style="padding:28px 36px 30px 36px;"><h1 style="font-size:2rem;">Market <b>insights</b></h1>
    <p>For recruiters, analysts and curious job seekers: how the market moved, what pays more and which roles are most in demand.</p></div>"""), unsafe_allow_html=True)
    st.write("")
    summary = T["period_summary"]
    latest_row = summary.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Current listings", f"{len(listings):,}", "unique titles from the last 14 days")
    with k2:
        kpi("Countries", f"{listings['country'].nunique()}", "with current listings", "amber")
    with k3:
        kpi(f"Postings per day ({PERIOD.lower()})", f"{latest_row['Postings per Day']:,.0f}", f"latest {PERIOD.lower()} in the data")
    with k4:
        kpi("Data through", META.get("data_through", "n/a"), "updates when the notebook is refreshed", "amber")
    st.write("")

    tab_dash, tab_pay, tab_demand = st.tabs(["Market dashboard", "Pay", "High-demand roles"])

    # ---- Market dashboard (Task 6) ----
    with tab_dash:
        st.markdown(f'<p class="section-title">How the latest {PERIOD.lower()} compares</p>'
                    f'<p class="sub">Each card shows the latest {PERIOD.lower()}, its change from the {PERIOD.lower()} before, and whether it is unusual compared with all earlier {PERIOD.lower()}s. '
                    'Amber cards are worth a closer look.</p>', unsafe_allow_html=True)
        latest = T["latest_view"]
        for column, (_, row) in zip(st.columns(len(latest)), latest.iterrows()):
            flagged = row["Status"] != "Within earlier range"
            change = 0.0 if pd.isna(row["Change (%)"]) else row["Change (%)"]
            arrow = "▲" if change > 0 else ("▼" if change < 0 else "•")
            with column:
                kpi(row["Measure"], f"{row['Latest']:,.2f}" if row["Latest"] < 1000 else f"{row['Latest']:,.0f}",
                    f'<span class="{"up" if change > 0 else "down"}">{arrow} {abs(change):.1f}%</span> vs previous {PERIOD.lower()} · {STATUS_WORDS.get(row["Status"], row["Status"])}', "watch" if flagged else "ok")
        st.write("")
        periods = summary.reset_index().rename(columns={summary.index.name or "index": f"{PERIOD} Starting"})
        first_column = periods.columns[0]
        periods[first_column] = pd.to_datetime(periods[first_column]).dt.strftime("%Y-%m-%d")
        panels = [("Postings per Day", "bar", TEAL), ("Freelancer Share (%)", "line", TEAL_DARK), ("Remote Share (%)", "line", AMBER), ("Average Hourly Rate (units)", "line", TEAL_DEEP)]
        grid = make_subplots(rows=2, cols=2, subplot_titles=[name for name, _, _ in panels], vertical_spacing=0.18, horizontal_spacing=0.1)
        for position, (name, kind, color) in enumerate(panels):
            row_no, col_no = position // 2 + 1, position % 2 + 1
            if kind == "bar":
                grid.add_trace(go.Bar(x=periods[first_column], y=periods[name], marker_color=color, showlegend=False), row=row_no, col=col_no)
            else:
                grid.add_trace(go.Scatter(x=periods[first_column], y=periods[name], mode="lines+markers", line=dict(color=color, width=3),
                                          marker=dict(size=8), showlegend=False), row=row_no, col=col_no)
            grid.update_yaxes(rangemode="tozero", row=row_no, col=col_no)
            grid.update_xaxes(type="category", row=row_no, col=col_no)
        style_fig(grid, f"{PERIOD}ly Market Dashboard", "line", height=620)
        grid.update_xaxes(showgrid=False)
        st.plotly_chart(grid, use_container_width=True)
        with st.expander(f"{PERIOD}ly summary table"):
            show_table(periods, {"Postings per Day": "%.0f", "Freelancer Share (%)": "%.2f", "Remote Share (%)": "%.2f", "Average Hourly Rate (units)": "%.2f",
                                 "Median Salary (units)": "%.2f", "Average Salary (units)": "%.2f"})
            st.download_button("Download the summary (CSV)", periods.to_csv(index=False), "market_summary.csv", "text/csv")
        note(f"Periods are weeks until three calendar months of data exist, then months. Only full days count, so a {PERIOD.lower()} with fewer full days is compared per day.")

    # ---- Pay (Tasks 1 and 4) ----
    with tab_pay:
        sub_country, sub_words, sub_drivers = st.tabs(["Hourly rate by country", "Title keywords and pay", "What drives pay"])
        with sub_country:
            rates = T["country_rates"]
            low, high = rates["Average"].quantile([0.05, 0.95])                                  # clip the scale so a few extreme countries do not wash out the rest
            world = px.choropleth(rates, locations="iso3", color="Average", hover_name="country", hover_data={"iso3": False, "Average": ":.2f", "Median": ":.2f", "Postings": ":,"},
                                  color_continuous_scale=MAP_SCALE, range_color=(low, high), labels={"Average": "Average rate (units)"})
            world.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="white", landcolor="#EEF2F1")
            world.update_layout(coloraxis_colorbar=dict(title="Units / hour", thickness=14))
            style_fig(world, "Average Hourly Rate by Client Country (units per hour)", "line", height=520)
            st.plotly_chart(world, use_container_width=True)
            note(f"{len(rates)} countries with at least 30 stated hourly postings. Others (country not stated) is left out because it is not a country.")
            side = st.radio("Show", ["Highest", "Lowest"], horizontal=True)
            ranked = rates.sort_values("Average", ascending=(side == "Lowest")).head(10).iloc[::-1]
            bars = go.Figure(go.Bar(x=ranked["Average"], y=ranked["country"], orientation="h", marker_color=TEAL if side == "Highest" else AMBER,
                                    text=[f"{value:.2f}" for value in ranked["Average"]], textposition="outside", customdata=ranked["Postings"],
                                    hovertemplate="%{y}: %{x:.2f} units<br>%{customdata:,} postings<extra></extra>"))
            bars.update_xaxes(title="Average hourly rate (units)", range=[0, ranked["Average"].max() * 1.15])
            st.plotly_chart(style_fig(bars, f"{side} Paying Countries", "hbar", height=440), use_container_width=True)
            note("Averages from countries with few postings swing more by chance, so read each rate with its posting count (shown on hover).")

        with sub_words:
            contract_pick = st.radio("Contract type", ["Freelancer", "Permanent"], horizontal=True, key="keyword_contract")
            words = T["keyword_pay"]
            words = words[(words["Contract"] == contract_pick) & words["Significant"]]
            count_pick = st.slider("Words to show on each side", 3, 10, 6)
            ranked_words = pd.concat([words.nlargest(count_pick, "Difference (%)"), words.nsmallest(count_pick, "Difference (%)")]).drop_duplicates("Word").sort_values("Difference (%)")
            bars = go.Figure(go.Bar(x=ranked_words["Difference (%)"], y=ranked_words["Word"], orientation="h",
                                    marker_color=[TEAL if value > 0 else AMBER for value in ranked_words["Difference (%)"]],
                                    text=[f"{value:+.0f}%" for value in ranked_words["Difference (%)"]], textposition="outside",
                                    customdata=np.stack([ranked_words["Median With"], ranked_words["Median Without"], ranked_words["Postings"]], axis=-1),
                                    hovertemplate="%{y}<br>median with: %{customdata[0]:.2f}<br>median without: %{customdata[1]:.2f}<br>%{customdata[2]:,} postings<extra></extra>"))
            bars.update_xaxes(title="Median pay difference with the word vs without (%)")
            bars.update_yaxes(tickfont=dict(size=14), automargin=True)
            st.plotly_chart(style_fig(bars, f"Title Words with the Largest Pay Premium and Discount — {contract_pick}", "hbar", height=150 + 34 * len(ranked_words)), use_container_width=True)
            note("Only words with a statistically significant difference (corrected for many tests) are shown. A word in a title can reflect one repeated client, "
                 "so use it as a signal, not a rule.")
            with st.expander("All significant words"):
                show_table(words.sort_values("Difference (%)", ascending=False)[["Word", "Postings", "Median With", "Median Without", "Difference (%)", "q_value"]],
                           {"Median With": "%.2f", "Median Without": "%.2f", "Difference (%)": "%.1f", "q_value": "%.2g"})

        with sub_drivers:
            contract_drive = st.radio("Contract type", ["Freelancer", "Permanent"], horizontal=True, key="driver_contract")
            top = T["pay_importance"]
            top = top[top["Contract"] == contract_drive].nlargest(12, "Importance").iloc[::-1]
            model_name = top["Model"].iloc[-1] if len(top) else ""
            bars = go.Figure(go.Bar(x=top["Importance"], y=top["Feature"], orientation="h", marker_color=CONTRACT_COLORS[contract_drive],
                                    text=[f"{value:.3f}" for value in top["Importance"]], textposition="outside"))
            bars.update_xaxes(title="Drop in R² when the feature is shuffled", range=[0, max(top["Importance"].max() * 1.2, 0.01)])
            st.plotly_chart(style_fig(bars, f"What Drives Pay — {contract_drive} ({model_name})", "hbar", height=480), use_container_width=True)
            st.markdown("**How well can pay be predicted from the posting?**")
            comparison = T["pay_model_comparison"]
            show_table(comparison[comparison["Contract"] == contract_drive], {"Test R² (log pay)": "%.3f", "MAE (units)": "%.2f", "RMSE (units)": "%.2f",
                                                                              "Median Error (units)": "%.2f", "CV R² (5-fold)": "%.3f"})
            note("R² is measured on log pay. Titles, category and country explain only part of what sets pay, so treat any estimate as a rough guide.")

    # ---- High-demand roles (Task 3) ----
    with tab_demand:
        demand = T["high_demand"]
        st.markdown('<p class="section-title">Roles ranked by forecast demand</p>'
                    '<p class="sub">The top five roles by forecast postings per day are marked High demand. Pick roles below to see their history and the 14-day forecast.</p>', unsafe_allow_html=True)
        show_table(demand, {"Last 7 Days (avg/day)": "%.1f", "Forecast (avg/day)": "%.1f", "Change (%)": "%.1f"})
        chosen = st.multiselect("Show demand history and forecast for", list(demand["Category"]), default=list(demand["Category"].head(3)))
        panel, forecast = T["demand_panel"], T["demand_forecast"]
        palette = [TEAL, AMBER, "#7C3AED", "#2563EB", "#DB2777", "#65A30D"]
        trend_fig = go.Figure()
        for position, category in enumerate(chosen):
            color = palette[position % len(palette)]
            trend_fig.add_trace(go.Scatter(x=panel.index, y=panel[category], mode="lines", line=dict(color=color, width=2.5), name=category, legendgroup=category))
            part = forecast[category]
            trend_fig.add_trace(go.Scatter(x=list(part.index) + list(part.index[::-1]), y=list(part["high"]) + list(part["low"][::-1]), fill="toself",
                                           fillcolor="rgba(120,113,108,0.18)", line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=category))
            trend_fig.add_trace(go.Scatter(x=part.index, y=part["forecast"], mode="lines", line=dict(color=color, width=2.5, dash="dash"),
                                           name=f"{category} (forecast)", legendgroup=category, showlegend=False))
        trend_fig.update_yaxes(title="Postings per day")
        st.plotly_chart(style_fig(trend_fig, "Daily Postings and 14-Day Forecast", "line", height=480), use_container_width=True)
        note("Solid lines are daily postings, dashed lines the forecast and the shaded band an 80% interval. "
             "The intervals come from the model's own errors, so real days can fall outside them.")
        with st.expander("How the forecasting model was chosen"):
            comparison = T["forecast_comparison"].reset_index()
            comparison = comparison.rename(columns={comparison.columns[0]: "Forecasting Model"})
            show_table(comparison, {column: "%.2f" for column in comparison.columns[1:]})


if page == "Find jobs":
    render_portal()
else:
    render_insights()

st.markdown('<div class="foot-note">JobLens is a learning project built on Upwork posting data. Pay is in units and listings are not hiring advice.</div>', unsafe_allow_html=True)
