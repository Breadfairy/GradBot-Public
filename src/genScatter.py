#!/usr/bin/env python3
"""
Quick scatter plot: trades vs net % vs HODL, colored by interval.

Usage:
  python3 scripts/genScatter.py <results.csv> [out.png]

Saves PNG to the provided out path or outputs/scatter.png by default.
"""

import argparse
import sys
import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from charting import BG_COLOR, TEXT_COLOR, GRID_COLOR, CLOSE_COLOR

from metrics import grossPctVsBench


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="genScatter.py",
        description="Scatter plot of trades vs performance metric.",
    )
    parser.add_argument("results_csv", help="Path to tuner results CSV.")
    parser.add_argument(
        "output_png",
        nargs="?",
        default=os.path.join("outputs", "scatter.png"),
        help="Destination PNG (default: outputs/scatter.png).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="Minimum trades to include (inclusive).",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=None,
        help="Maximum trades to include (inclusive).",
    )
    parser.add_argument(
        "--title-suffix",
        default=None,
        help="Optional text appended to the chart title.",
    )
    return parser.parse_args(argv)


def _filtered_df(
    path: str,
    tickers: list[str] | None,
    min_trades: int | None,
    max_trades: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if tickers:
        allowed = {str(t).upper() for t in tickers}
        df = df[df["ticker"].str.upper().isin(allowed)]
    if min_trades is not None:
        df = df[df["trades"] >= min_trades]
    if max_trades is not None:
        df = df[df["trades"] <= max_trades]
    return df


def _percent_vs_benchmark(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compute gross percent vs benchmark for scatter.

    Prefer gross portfolio values to match portfolio summary's
    "Gross Values Comparison". Fall back to post-tax values if
    gross columns are missing.
    """
    # Compute from gross values (preferred)
    if {"simValue", "benchValue"}.issubset(df.columns):
        bench = df["benchValue"].astype(float)
        sim = df["simValue"].astype(float)
        pct = ((sim / bench) - 1.0) * 100.0
        out = df.copy()
        out["pctVsBench"] = pct
        return out, "pctVsBench"

    # Fallback to post-tax values
    if {"simPostTax", "benchPostTax"}.issubset(df.columns):
        bench = df["benchPostTax"].astype(float)
        sim = df["simPostTax"].astype(float)
        pct = ((sim / bench) - 1.0) * 100.0
        out = df.copy()
        out["pctVsBench"] = pct
        return out, "pctVsBench"

    # If none of the above, just create a zero column to avoid crash
    out = df.copy()
    out["pctVsBench"] = 0.0
    return out, "pctVsBench"


def generate_scatter(
    results_csv: str,
    output_png: str,
    tickers: list[str] | None = None,
    min_trades: int | None = None,
    max_trades: int | None = None,
    title_suffix: str | None = None,
) -> None:
    df = _filtered_df(results_csv, tickers, min_trades, max_trades)
    if df.empty:
        print("No rows to plot after applying filters; skipping scatter.")
        return

    df, yColumn = _percent_vs_benchmark(df)
    yLabel = "Increase vs Benchmark (%)"
    title = "Trades vs % vs Benchmark"

    if not title_suffix:
        suffixParts = []
        if min_trades is not None:
            suffixParts.append(f">= {min_trades} trades")
        if max_trades is not None:
            suffixParts.append(f"<= {max_trades} trades")
        if suffixParts:
            title_suffix = ", ".join(suffixParts)
    if title_suffix:
        title = f"{title} ({title_suffix})"

    df = df[pd.notna(df[yColumn])]
    if df.empty:
        print("No finite metric values after filtering; skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

    if "interval" in df.columns:
        for interval, grp in df.groupby("interval"):
            ax.scatter(
                grp["trades"],
                grp[yColumn],
                label=str(interval),
                s=14,
                alpha=0.75,
                color=CLOSE_COLOR,
                edgecolors="none",
            )
        leg = ax.legend(title="interval", loc="best", frameon=True, fontsize=8)
        if leg:
            leg.get_title().set_color(TEXT_COLOR)
            for txt in leg.get_texts():
                txt.set_color(TEXT_COLOR)
            frame = leg.get_frame()
            if frame:
                frame.set_facecolor(BG_COLOR)
                frame.set_edgecolor(GRID_COLOR)
    else:
        ax.scatter(
            df["trades"],
            df[yColumn],
            s=14,
            alpha=0.75,
            color=CLOSE_COLOR,
            edgecolors="none",
        )

    ax.set_xlabel("Number of Trades", color=TEXT_COLOR)
    ax.set_ylabel(yLabel, color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR, pad=8)
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
        color=GRID_COLOR,
    )

    tradeMedian = float(df["trades"].median())
    metricMedian = float(df[yColumn].median())
    ax.axvline(
        tradeMedian,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )
    ax.axhline(
        metricMedian,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"Saved {os.path.basename(output_png)}")


def main():
    args = _parse_args(sys.argv[1:])
    generate_scatter(
        args.results_csv,
        args.output_png,
        tickers=None,
        min_trades=args.min_trades,
        max_trades=args.max_trades,
        title_suffix=args.title_suffix,
    )


if __name__ == "__main__":
    main()
