"""
Sleep Data Visualizer — SQLite Edition
=======================================
Reads sleep data directly from a SQLite database and generates
one line graph per metric, outputting a single PNG image.

Setup:
    1. Create a .env file in the same folder as this script:

           DB_PATH=path/to/your/database.db
           DB_TABLE=your_table_name

    2. Install dependencies:
           pip install pandas matplotlib python-dotenv

Usage:
    # Plot all users
    python sleep_visualizer.py

    # Plot a specific user
    python sleep_visualizer.py --user "Test User"

    # Custom output filename
    python sleep_visualizer.py --user "Test User" --output my_output.png
"""

import argparse
import os
import sqlite3
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, AutoDateLocator
from dotenv import load_dotenv
import numpy as np


# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

DB_PATH  = os.getenv("DB_PATH")
DB_TABLE = os.getenv("DB_TABLE")


# ── Sleep cycle ordinal mapping (A→D, worst→best quality) ────────────────────
# Ordered from lightest/least restorative to deepest/most restorative
SLEEP_CYCLE_ORDER = {
    "Unsure":                   0,
    "Light sleep (N1-N2 stages)": 1,
    "REM sleep":                2,
    "Deep sleep (N3 stage)":    3,
}
SLEEP_CYCLE_LABELS = {v: k for k, v in SLEEP_CYCLE_ORDER.items()}


# ── Columns to plot ───────────────────────────────────────────────────────────
METRIC_COLUMNS = [
    "memory_score",
    "amount_of_sleep",
    "wake_up_in_sleep_cycle",
    "times_woke_up",
    "normal_wake_up_time",
]

METRIC_TITLES = {
    "memory_score":           "Memory Score",
    "amount_of_sleep":        "Amount of Sleep (hrs)",
    "wake_up_in_sleep_cycle": "Wake-Up Sleep Cycle Stage",
    "times_woke_up":          "Times Woke Up",
    "normal_wake_up_time":    "Normal Wake-Up Time",
}

USER_COLORS = [
    "#4FC3F7", "#FF8A65", "#81C784", "#CE93D8",
    "#FFD54F", "#4DB6AC", "#F48FB1", "#AED581",
]

BACKGROUND   = "#0E1117"
PANEL_BG     = "#161B22"
GRID_COLOR   = "#21262D"
TEXT_COLOR   = "#C9D1D9"
ACCENT_COLOR = "#58A6FF"


# ── DB connection ─────────────────────────────────────────────────────────────
def load_data(user_filter):
    if not DB_PATH:
        sys.exit("[ERROR] DB_PATH is not set in your .env file.")
    if not DB_TABLE:
        sys.exit("[ERROR] DB_TABLE is not set in your .env file.")
    if not os.path.exists(DB_PATH):
        sys.exit(f"[ERROR] Database file not found: {DB_PATH}")

    try:
        conn = sqlite3.connect(DB_PATH)
        if user_filter:
            query = f"SELECT * FROM {DB_TABLE} WHERE name = ?"
            df = pd.read_sql_query(query, conn, params=(user_filter,))
        else:
            df = pd.read_sql_query(f"SELECT * FROM {DB_TABLE}", conn)
        conn.close()
    except Exception as e:
        sys.exit(f"[ERROR] Failed to query database: {e}")

    if df.empty:
        msg = f"No data found in table '{DB_TABLE}'"
        if user_filter:
            msg += f" for user '{user_filter}'"
        sys.exit(f"[ERROR] {msg}.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_numeric(series):
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() > 0:
        return converted
    unique_vals = series.dropna().unique().tolist()
    mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
    return series.map(mapping).astype(float)


def _parse_wake_time(series):
    def _single(val):
        if pd.isna(val):
            return np.nan
        try:
            start = str(val).split("-")[0].strip()
            t = pd.to_datetime(start, format="%I:%M %p")
            return t.hour + t.minute / 60
        except Exception:
            return np.nan
    return series.apply(_single)


def _encode_sleep_cycle(series):
    """Map sleep cycle strings to ordinal ints using the known mapping."""
    def _single(val):
        if pd.isna(val):
            return np.nan
        # Try exact match first
        if val in SLEEP_CYCLE_ORDER:
            return float(SLEEP_CYCLE_ORDER[val])
        # Fallback: case-insensitive partial match
        val_lower = str(val).lower()
        for key, num in SLEEP_CYCLE_ORDER.items():
            if key.lower() in val_lower or val_lower in key.lower():
                return float(num)
        return np.nan
    return series.apply(_single)


def _parse_times_woke_up(series):
    """Parse times woke up values like '1-2 times' into numeric (midpoint)."""
    import re
    def _single(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        # Match patterns like "1-2 times" or "2-3 times"
        match = re.match(r'(\d+)\s*-\s*(\d+)\s*times?', val_str, re.IGNORECASE)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            return (low + high) / 2.0
        # Match single number patterns like "2 times" or just "2"
        match = re.match(r'(\d+)\s*(?:times?)?', val_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return np.nan
    return series.apply(_single)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_metrics(df, user_filter, output_path):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    if "normal_wake_up_time" in df.columns:
        df["normal_wake_up_time"] = _parse_wake_time(df["normal_wake_up_time"])

    if "wake_up_in_sleep_cycle" in df.columns:
        df["wake_up_in_sleep_cycle"] = _encode_sleep_cycle(df["wake_up_in_sleep_cycle"])

    if "times_woke_up" in df.columns:
        df["times_woke_up"] = _parse_times_woke_up(df["times_woke_up"])

    cols_to_plot = [c for c in METRIC_COLUMNS if c in df.columns]
    if not cols_to_plot:
        sys.exit("[ERROR] None of the expected metric columns were found in the data.")

    users = df["name"].unique().tolist()
    fig, axes = plt.subplots(
        len(cols_to_plot), 1,
        figsize=(14, 4.5 * len(cols_to_plot)),
        facecolor=BACKGROUND,
    )
    if len(cols_to_plot) == 1:
        axes = [axes]

    title_suffix = f" — {user_filter}" if user_filter else " — All Users"
    fig.suptitle(
        f"Sleep Analytics{title_suffix}",
        fontsize=22, fontweight="bold",
        color=ACCENT_COLOR, y=1.01,
        fontfamily="monospace",
    )

    for ax, col in zip(axes, cols_to_plot):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.8)
        ax.set_title(METRIC_TITLES.get(col, col),
                     color=TEXT_COLOR, fontsize=13, pad=8, loc="left",
                     fontfamily="monospace")
        ax.set_xlabel("Date", color=TEXT_COLOR, fontsize=9)

        for idx, user in enumerate(users):
            udf = df[df["name"] == user].copy()
            color = USER_COLORS[idx % len(USER_COLORS)]
            y = pd.to_numeric(udf[col], errors="coerce")
            ax.plot(udf["date"], y, marker="o", markersize=5,
                    linewidth=2, color=color, label=user, alpha=0.9)
            ax.fill_between(udf["date"], y, alpha=0.07, color=color)

        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%Y"))
        ax.tick_params(axis="x", colors=TEXT_COLOR)
        ax.tick_params(axis="y", colors=TEXT_COLOR)

        # ── Special y-axis formatting per column ──────────────────────────────
        if col == "wake_up_in_sleep_cycle":
            positions = sorted(SLEEP_CYCLE_LABELS.keys())
            labels    = [SLEEP_CYCLE_LABELS[p] for p in positions]
            ax.set_yticks(positions)
            ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=8)
            ax.set_ylim(-0.4, max(positions) + 0.4)

        elif col == "normal_wake_up_time":
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda h, _:
                    f"{int(h % 12) or 12}:00 {'AM' if h < 12 else 'PM'}")
            )

        if len(users) > 1:
            ax.legend(fontsize=8, framealpha=0.2, facecolor=PANEL_BG,
                      edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper left")

    plt.tight_layout(pad=2.5)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"[OK] Saved -> {output_path}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sleep data line-graph visualizer (SQLite)")
    parser.add_argument("--user",   default=None, help="Filter to a specific user name")
    parser.add_argument("--output", default="sleep_graphs.png",
                        help="Output image filename (default: sleep_graphs.png)")
    args = parser.parse_args()

    print(f"[·] Connecting to: {DB_PATH}")
    print(f"[·] Table: {DB_TABLE}")

    df = load_data(args.user)
    print(f"[·] Loaded {len(df)} rows | Users: {', '.join(df['name'].unique())}")

    plot_metrics(df, args.user, args.output)


if __name__ == "__main__":
    main()