"""
================================================================================
 PREREQUISITES.PY

 Project      : NHIS Project 8 - Job Market Analysis and Recommendation System
                (Upwork job postings)

 Author       : Yousuf S. R. Sakkaf

 Description  : Shared utility module imported by the single project notebook:
                job_market_analysis.ipynb

 Scope        : General, reusable helpers plus the one dataset-specific piece
                this project cannot do without - the robust Upwork CSV loader
                (pd.read_csv fails on the raw file because of stray quotes).
                Analytical decisions (which strategy on which column, which
                features, which model) are made visibly inside the notebook,
                never hidden in a function here.

 Contents     : - Colour theme (teal / amber) and chart colour scales
                - Logging configuration
                - HTML section headers, status boxes, centred tables
                - Visualization defaults
                - Upwork loader: Local -> Google Drive -> GitHub raw
                - Model artifact save/load (check-before-retrain)
                - Generic audits: missing values, duplicates
                - Generic statistics: IQR, Z-score, skew, kurtosis, outlier summary

 Usage        : from prerequisites import *
================================================================================
"""

import os
import re
import glob
import json
import shutil
import logging
import joblib
import urllib.request
import urllib.error
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skew, kurtosis, zscore
from IPython.display import display, HTML

# =========================================================
# PROJECT LOCATIONS
# =========================================================
GITHUB_REPO = "S-Yousuf-S/NHIS_Project8"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/Assets"
GITHUB_MODELS_API = f"https://api.github.com/repos/{GITHUB_REPO}/contents/Assets/models"

LOCAL_ASSETS_DIR = "Assets"
DRIVE_ASSETS_DIR = "/content/drive/MyDrive/NHIS_Project/Project 8/Assets"
LOCAL_MODELS_DIR = os.path.join(LOCAL_ASSETS_DIR, "models")
DRIVE_MODELS_DIR = os.path.join(DRIVE_ASSETS_DIR, "models")

RAW_DATA_FILENAME = "all_upwork_jobs.csv"

# =========================================================
# COLOR THEME - teal family with an amber accent
#
# Status colours (success / warning / info) are fixed semantic colours, the same in
# every project. Amber is never used as a fill under white text (about 2.2:1 contrast).
# =========================================================
PRIMARY_COLOR = "#008080"        # teal - section header background (white text, 4.8:1)
SECONDARY_COLOR = "#E0F2F1"      # pale teal - subsection background
TABLE_COLOR = "#006D6D"          # table header background (white text, 6.2:1)
SUBSECTION_TEXT_COLOR = "#006D6D"
PLOT_COLOR = "#134E4A"           # chart titles / text
INFO_COLOR = "#1F77B4"           # info box border and label (fixed semantic colour) 
INFO_BG = "#F1F3F6"                                                                   
ACCENT_COLOR = "#F59E0B"         # amber - highlights, winner bars, rules
SUCCESS_COLOR = "#2E8B57"        # fixed semantic colour - borders / label / good marks 
SUCCESS_TEXT = "#2E8B57"                                                              
SUCCESS_BG = "#EAF7EA"                                                                 
WARNING_COLOR = "#C0392B"        # fixed semantic colour - borders / label / bad marks  
WARNING_TEXT = "#C0392B"                                                             
WARNING_BG = "#FDEDEC"                                              
NEUTRAL_COLOR = "#78716C"        # warm grey - third categorical colour, gridlines

# Categorical order: teal, amber, warm grey, deep teal. Beyond four series use
# small multiples or direct labels rather than more hues.
CATEGORICAL_PALETTE = [PRIMARY_COLOR, ACCENT_COLOR, NEUTRAL_COLOR, PLOT_COLOR]

# Sequential (light -> dark teal) and diverging (amber <-> teal, colour-blind safe;
# red/green are reserved for status boxes only).
SEQUENTIAL_COLORS = [SECONDARY_COLOR, PRIMARY_COLOR, PLOT_COLOR]
DIVERGING_COLORS = [ACCENT_COLOR, "#F5F5F4", PRIMARY_COLOR]
TEAL_SEQ_CMAP = LinearSegmentedColormap.from_list("teal_seq", SEQUENTIAL_COLORS)
AMBER_TEAL_DIV_CMAP = LinearSegmentedColormap.from_list("amber_teal_div", DIVERGING_COLORS)
PLOTLY_TEAL_SCALE = SEQUENTIAL_COLORS


# =========================================================
# LOGGING
# =========================================================
LOGGER_NAME = "job_market_analysis"


def setup_logging(log_dir="logs", name=LOGGER_NAME):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"YS_{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_filename, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized -> {log_filename}")
    return logger


# =========================================================
# HTML STATUS BOXES / SECTION HEADERS
# Centered text, inline styles (not a <style> block, which can leak across
# Colab cell outputs), bold-label + black-message box format.
# =========================================================
def section_header(title):
    display(HTML(f"""
    <div style="background:{PRIMARY_COLOR};color:white;padding:16px;border-radius:8px;font-size:28px;font-weight:bold;margin-top:20px;margin-bottom:12px;text-align:center;">
        {title}
    </div>"""))


def subsection_header(title):
    display(HTML(f"""
    <div style="background:{SECONDARY_COLOR};border-left:6px solid {PRIMARY_COLOR};color:{SUBSECTION_TEXT_COLOR};padding:10px;border-radius:6px;font-size:20px;font-weight:bold;margin-top:15px;margin-bottom:10px;text-align:center;">
        <i>{title}</i>
    </div>"""))


def success_box(message):
    display(HTML(f"""
    <div style="background:{SUCCESS_BG};border-left:6px solid {SUCCESS_COLOR};color:{SUCCESS_TEXT};padding:12px;border-radius:6px;margin:10px 0;">
        <b>✅ Success:</b><br><span style="color:black;">{message}</span>
    </div>"""))


def warning_box(message):
    display(HTML(f"""
    <div style="background:{WARNING_BG};border-left:6px solid {WARNING_COLOR};color:{WARNING_TEXT};padding:12px;border-radius:6px;margin:10px 0;">
        <b>⚠️ Note:</b><br><span style="color:black;">{message}</span>
    </div>"""))


def info_box(message):
    display(HTML(f"""
    <div style="background:{INFO_BG};border-left:6px solid {INFO_COLOR};color:{INFO_COLOR};padding:12px;border-radius:6px;margin:10px 0;">
        <b>ℹ️ Information:</b><br><span style="color:black;">{message}</span>
    </div>"""))


def centered_table(df, index=False, float_format='{:.4f}'):
    """Renders a DataFrame as a centered, well-spaced HTML table matching the theme."""
    html = df.to_html(index=index, border=0, escape=False, float_format=float_format.format)
    html = html.replace('class="dataframe"', '')
    html = html.replace(
        '<table', '<table style="margin:auto;border-collapse:separate;border-spacing:0;'
        'border:1px solid #ccc;border-radius:6px;overflow:hidden;"'
    ).replace(
        '<th>', f'<th style="background:{TABLE_COLOR};color:white;padding:10px 24px;'
                f'text-align:center !important;border-bottom:1px solid #ccc;">'
    ).replace(
        '<td>', '<td style="padding:8px 24px;text-align:center !important;border-bottom:1px solid #eee;">'
    )
    display(HTML(f'<div style="overflow-x:auto;margin:15px 0;">{html}</div>'))


# =========================================================
# VISUALIZATION DEFAULTS
# =========================================================
def set_global_visualization_settings(palette=None):
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.width", 1000)
    sns.set_theme(style="whitegrid", palette=palette or CATEGORICAL_PALETTE)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlecolor"] = PLOT_COLOR
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.labelcolor"] = PLOT_COLOR
    plt.rcParams["grid.color"] = "#E7E5E4"


# =========================================================
# UPWORK CSV - ROBUST LINE-BASED LOADER (Local -> Drive -> GitHub)
#
# Why not pd.read_csv: a stray, unbalanced quote inside a title makes the
# C parser treat the rest of the file as one string ("EOF inside string").
# Every record has the same anchor instead: the closing quote of the title is
# followed by  ","https://www.upwork.com/  - so the title is everything before
# that anchor (it may span several physical lines) and the remaining seven
# fields are split from the right-hand side.
# =========================================================
UPWORK_LINK_MARKER = '","https://www.upwork.com/'
UPWORK_EMPTY_LINK_MARKER = '","","'     # posting whose link field is empty
# Every posting carries exactly one "timestamp + is_hourly flag" sequence, whatever its title or link look like
UPWORK_RECORD_TAIL = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"?,(?:TRUE|FALSE|True|False),')


def _find_record_anchor(line):
    """Index where the title ends and the link field starts (earliest of the two markers), or -1."""
    hits = [i for i in (line.find(UPWORK_LINK_MARKER), line.find(UPWORK_EMPTY_LINK_MARKER)) if i != -1]
    return min(hits) if hits else -1


UPWORK_COLUMNS = [
    "title", "link", "published_date", "is_hourly",
    "hourly_low", "hourly_high", "budget", "country",
]
UPWORK_NUMERIC_COLUMNS = ["hourly_low", "hourly_high", "budget"]


def parse_upwork_lines(lines, expected_rows=None, dataset_name="upwork_jobs"):
    """
    Parse the raw Upwork CSV line by line.

    Parameters
    ----------
    lines : iterable of str
        Physical lines of the file, header first.
    expected_rows : int | None
        Optional record count to reconcile against (load_upwork_jobs derives it from the file).

    Returns
    -------
    (DataFrame, dict)
        Raw-typed frame (text, bool, float) and a load report. Dates stay as
        text on purpose - parsing and sorting happen in Data Cleaning.
    """
    logger = logging.getLogger(LOGGER_NAME)

    iterator = iter(lines)
    header = next(iterator, "")
    if not header.lstrip('\ufeff"').lower().startswith("title"):
        logger.warning(f"[{dataset_name}] First line does not look like a header: {header[:80]!r}")

    rows, bad_lines, pending = [], [], []
    data_lines = blank_lines = joined_titles = 0

    for line_no, raw_line in enumerate(iterator, start=2):
        data_lines += 1
        line = raw_line.rstrip("\r\n")

        if not line.strip() and not pending:
            blank_lines += 1
            continue

        anchor = _find_record_anchor(line)
        if anchor == -1:                       # title continues on the next line
            pending.append(line)
            logger.debug(f"[{dataset_name}] Line {line_no}: title continues (no link anchor)")
            continue

        if pending:
            joined_titles += 1
        title = " ".join(pending + [line[:anchor]])
        pending = []

        if title.startswith('"'):
            title = title[1:]
        title = " ".join(title.replace('""', '"').split())

        parts = line[anchor + 3:].split(",", 6)
        if len(parts) != 7:
            bad_lines.append((line_no, line[:200]))
            logger.debug(f"[{dataset_name}] Line {line_no}: expected 7 fields after title, got {len(parts)}")
            continue

        link, pub, is_hourly, low, high, budget, country = [p.strip().strip('"') for p in parts]
        rows.append((title, link, pub, is_hourly, low, high, budget, country))

    df = pd.DataFrame(rows, columns=UPWORK_COLUMNS)

    # --- type conversion (counted, never silent) ---
    hourly_map = df["is_hourly"].str.upper().map({"TRUE": True, "FALSE": False})
    unmapped_bool = int(hourly_map.isna().sum())
    df["is_hourly"] = hourly_map.astype(bool) if unmapped_bool == 0 else hourly_map.astype("boolean")

    coercion_failures = {}
    for col in UPWORK_NUMERIC_COLUMNS:
        raw_text = df[col]
        converted = pd.to_numeric(raw_text, errors="coerce")
        coercion_failures[col] = int((converted.isna() & raw_text.str.strip().ne("")).sum())
        df[col] = converted

    df["country"] = df["country"].replace("", pd.NA)

    report = {
        "data_lines_read": data_lines,
        "records_parsed": len(df),
        "titles_joined_across_lines": joined_titles,
        "bad_lines": len(bad_lines),
        "bad_line_examples": bad_lines[:3],
        "leftover_fragments": len(pending),
        "blank_lines": blank_lines,
        "titles_with_replacement_char": int(df["title"].str.contains("�", regex=False).sum()),
        "unmapped_is_hourly": unmapped_bool,
        "numeric_coercion_failures": coercion_failures,
        "expected_rows": expected_rows,
        "row_check": (
            "not checked" if expected_rows is None
            else ("match" if len(df) == expected_rows else "mismatch")
        ),
    }

    logger.info(
        f"[{dataset_name}] Parsed {report['records_parsed']} records from {data_lines} data lines "
        f"({joined_titles} multi-line titles, {len(bad_lines)} bad lines, {len(pending)} leftover fragments)"
    )
    if report["row_check"] == "mismatch":
        logger.warning(f"[{dataset_name}] Row count {len(df)} differs from expected {expected_rows}")
    return df, report


def load_upwork_jobs(
    filename=RAW_DATA_FILENAME,
    local_dir=LOCAL_ASSETS_DIR,
    drive_path=DRIVE_ASSETS_DIR,
    github_raw_base=None,
    expected_rows=None,
    dataset_name="upwork_jobs",
):
    """
    Locate the raw file via Local -> Google Drive -> GitHub raw, cache it
    locally, then parse it with parse_upwork_lines().

    Returns (DataFrame, report) - report["source"] says which tier supplied it.
    """
    logger = logging.getLogger(LOGGER_NAME)
    github_raw_base = github_raw_base or GITHUB_RAW_BASE
    local_path = os.path.join(local_dir, filename)
    source = None

    if os.path.exists(local_path):
        source = "local"
        logger.info(f"[{dataset_name}] Found locally: {local_path}")

    if source is None and drive_path:
        drive_file = os.path.join(drive_path, filename)
        if os.path.exists(drive_file):
            os.makedirs(local_dir, exist_ok=True)
            shutil.copy2(drive_file, local_path)
            source = "drive"
            logger.info(f"[{dataset_name}] Copied from Drive: {drive_file} -> {local_path}")
        else:
            logger.debug(f"[{dataset_name}] Not on Drive (or Drive not mounted): {drive_file}")

    if source is None:
        github_url = f"{github_raw_base}/{filename}"
        logger.info(f"[{dataset_name}] Falling back to GitHub: {github_url}")
        os.makedirs(local_dir, exist_ok=True)
        urllib.request.urlretrieve(github_url, local_path)
        source = "github"
        logger.debug(f"[{dataset_name}] Cached local copy: {local_path}")
    
    # Independent record count taken from the file itself: every posting contains exactly one
    # timestamp + is_hourly sequence, and counting it on the whole text does not depend on how lines are split.
    
    if expected_rows is None:
        with open(local_path, encoding="utf-8", errors="replace") as handle:
            expected_rows = len(UPWORK_RECORD_TAIL.findall(handle.read()))
        logger.info(f"[{dataset_name}] Independent record count: {expected_rows}")

    with open(local_path, encoding="utf-8", errors="replace") as handle:
        df, report = parse_upwork_lines(handle, expected_rows=expected_rows, dataset_name=dataset_name)

    report["source"] = source
    return df, report


def show_load_report(report):
    """Render the load report with the project's HTML helpers (no raw print)."""
    rows = [
        ("Data lines read", f"{report['data_lines_read']:,}"),
        ("Records parsed", f"{report['records_parsed']:,}"),
        ("Titles joined across lines", f"{report['titles_joined_across_lines']:,}"),
        ("Bad lines", f"{report['bad_lines']:,}"),
        ("Leftover fragments", f"{report['leftover_fragments']:,}"),
        ("Titles with replacement character", f"{report['titles_with_replacement_char']:,}"),
        ("Unmapped is_hourly values", f"{report['unmapped_is_hourly']:,}"),
        ("Numeric coercion failures", ", ".join(f"{k}: {v}" for k, v in report["numeric_coercion_failures"].items())),
        ("Record count check", report["row_check"] if report["expected_rows"] is None else
            f"{report['row_check']} (parsed {report['records_parsed']:,} vs {report['expected_rows']:,} records counted in the file)")
    ]
    centered_table(pd.DataFrame(rows, columns=["check", "result"]))

    problems = (
        report["bad_lines"] + report["leftover_fragments"] + report["unmapped_is_hourly"]
        + sum(report["numeric_coercion_failures"].values())
    )
    if report["row_check"] == "mismatch" or problems:
        warning_box(
            f"Load finished with {problems} unparsed or unconverted values and a record count check of "
            f"'{report['row_check']}'. Inspect bad_line_examples before cleaning."
        )
    else:
        success_box(f"All {report['records_parsed']:,} records parsed cleanly.")


# =========================================================
# MODEL ARTIFACT SAVE/LOAD (check-before-retrain)
#
#   Local -> Google Drive -> GitHub (Assets/models) -> None
#
# A missing artifact returns None. Training is decided by the calling notebook
# cell and is NEVER triggered implicitly here. Artifacts are joblib .pkl files
# (dicts holding results, best-model name and the fitted object).
# =========================================================
def save_model_artifact(model, model_name, local_dir=LOCAL_MODELS_DIR, drive_dir=DRIVE_MODELS_DIR):
    """
    Save a fitted object locally (timestamped .pkl) and mirror it to Drive.

    Drive folder is created when its parent folder exists (i.e. Drive is
    mounted and the project folder is there); otherwise the skip is logged
    loudly with the reason. GitHub upload is intentionally manual.
    Returns the artifact file name.
    """
    logger = logging.getLogger(LOGGER_NAME)
    os.makedirs(local_dir, exist_ok=True)

    filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    local_path = os.path.join(local_dir, filename)
    joblib.dump(model, local_path)
    logger.info(f"Saved model artifact locally: {local_path}")

    if drive_dir:
        parent = os.path.dirname(os.path.normpath(drive_dir))
        if os.path.isdir(parent):
            os.makedirs(drive_dir, exist_ok=True)
            drive_path = os.path.join(drive_dir, filename)
            shutil.copy2(local_path, drive_path)
            logger.info(f"Backed up model artifact to Drive: {drive_path}")
        else:
            logger.warning(
                f"Drive backup skipped: parent folder not found ({parent}). "
                f"Drive is not mounted or the project folder does not exist."
            )
    return filename


def _github_model_files(github_api_url=GITHUB_MODELS_API):
    """List files in Assets/models on GitHub (Contents API, no assumed file names)."""
    try:
        with urllib.request.urlopen(github_api_url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            return []
        return [item["name"] for item in payload if item.get("type") == "file"]
    except Exception as e:
        logging.getLogger(LOGGER_NAME).warning(f"GitHub artifact discovery failed: {e}")
        return []


def load_latest_model_artifact(
    model_name,
    local_dir=LOCAL_MODELS_DIR,
    drive_dir=DRIVE_MODELS_DIR,
    github_raw_base=GITHUB_RAW_BASE,
):
    """
    Load the newest serialized artifact for `model_name`:
    Local -> Google Drive -> GitHub. Returns the object, or None if not found.
    Each miss is logged with its reason so a silent skip is never possible.
    """
    logger = logging.getLogger(LOGGER_NAME)
    pattern = f"{model_name}_*.pkl"

    def _latest_match(directory, label):
        if not directory:
            return None
        if not os.path.isdir(directory):
            logger.info(f"[{model_name}] {label} folder not available: {directory}")
            return None
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if not matches:
            logger.info(f"[{model_name}] {label} folder reachable but holds no '{pattern}'")
            return None
        return matches[-1]

    # 1. Local
    latest = _latest_match(local_dir, "Local")
    if latest:
        logger.info(f"Found existing local artifact: {latest}")
        return joblib.load(latest)

    # 2. Google Drive (cache a local copy)
    latest = _latest_match(drive_dir, "Drive")
    if latest:
        logger.info(f"Found existing Drive artifact: {latest}")
        os.makedirs(local_dir, exist_ok=True)
        cached = os.path.join(local_dir, os.path.basename(latest))
        shutil.copy2(latest, cached)
        logger.info(f"Cached Drive artifact locally: {cached}")
        return joblib.load(cached)

    # 3. GitHub (Contents API discovery, raw download)
    matches = sorted(
        name for name in _github_model_files()
        if name.startswith(f"{model_name}_") and name.endswith(".pkl")
    )
    if matches:
        filename = matches[-1]
        github_url = f"{github_raw_base}/models/{filename}"
        logger.info(f"Found GitHub artifact: {github_url}")
        os.makedirs(local_dir, exist_ok=True)
        cached = os.path.join(local_dir, filename)
        try:
            urllib.request.urlretrieve(github_url, cached)
        except Exception as e:
            logger.warning(f"GitHub artifact download failed: {e}")
            return None
        logger.info(f"Cached GitHub artifact locally: {cached}")
        return joblib.load(cached)

    logger.info(f"No existing artifact found for {model_name} across Local, Drive, or GitHub.")
    return None


# =========================================================
# GENERIC DIAGNOSTICS (dataset-agnostic)
# =========================================================
def missing_value_audit(df, dataset_name="dataset"):
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    audit = pd.DataFrame({"missing_count": missing, "missing_pct": pct.round(2)})
    audit = audit[audit["missing_count"] > 0].sort_values("missing_count", ascending=False)
    audit = audit.reset_index().rename(columns={"index": "column"})
    logging.getLogger(LOGGER_NAME).info(f"[{dataset_name}] Missing-value audit: {len(audit)} columns affected")
    return audit


def duplicate_check(df, subset=None, dataset_name="dataset"):
    dupe_count = df.duplicated(subset=subset).sum()
    logging.getLogger(LOGGER_NAME).info(f"[{dataset_name}] Duplicate rows found: {dupe_count}")
    return dupe_count


def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def detect_outliers_iqr(series, k=1.5):
    lower, upper = iqr_bounds(series, k)
    return series[(series < lower) | (series > upper)]


def detect_outliers_zscore(series, threshold=3):
    z = zscore(series.dropna())
    return series.dropna()[np.abs(z) > threshold]


def distribution_summary(series, sample_size=None, random_state=42):
    data = series.dropna()
    if sample_size and len(data) > sample_size:
        data = data.sample(sample_size, random_state=random_state)
    return {
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "skew": skew(data),
        "kurtosis": kurtosis(data),
    }


def outlier_summary(df, columns, k=1.5, z_threshold=3, log_scale=False, dataset_name="dataset"):
    """
    One-row-per-column outlier report (detection only - treatment is decided
    in the notebook). With log_scale=True the bounds are computed on log values
    of the strictly positive entries and reported back in original units, which
    is the right scale for heavy-tailed pay columns.
    """
    rows = []
    for col in columns:
        values = df[col].dropna()
        if log_scale:
            values = values[values > 0]
        n = len(values)
        if n < 3:
            rows.append({"column": col, "n_values": n})
            continue
        scaled = np.log(values) if log_scale else values
        lower, upper = iqr_bounds(scaled, k)
        n_iqr = int(((scaled < lower) | (scaled > upper)).sum())
        std = scaled.std()
        n_z = int((np.abs((scaled - scaled.mean()) / std) > z_threshold).sum()) if std > 0 else 0
        if log_scale:
            lower, upper = float(np.exp(lower)), float(np.exp(upper))
        rows.append({
            "column": col,
            "n_values": n,
            "iqr_lower": round(float(lower), 2),
            "iqr_upper": round(float(upper), 2),
            "iqr_outliers": n_iqr,
            "iqr_outlier_pct": round(100 * n_iqr / n, 2),
            "z_outliers": n_z,
            "max_value": float(values.max()),
        })
    logging.getLogger(LOGGER_NAME).info(
        f"[{dataset_name}] Outlier summary built for {len(columns)} columns (log_scale={log_scale})"
    )
    return pd.DataFrame(rows)