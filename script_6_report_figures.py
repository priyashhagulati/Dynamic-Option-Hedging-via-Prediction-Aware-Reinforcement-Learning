"""
Report Figures — BOLA Adaptive RL vs Vanilla RL vs Black-Scholes
Generates 5 publication-quality PNG figures for the project report.

Usage:
    python script_7_report_figures.py

Output:
    results/figures/fig1_cumulative_pnl.png
    results/figures/fig2_performance_metrics.png
    results/figures/fig3_rolling_sharpe.png
    results/figures/fig4_vix_regime_analysis.png
    results/figures/fig5_pnl_distribution.png
    results/figures/fig_all_combined.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_CSV = "results/bola_lstm_comparison.csv"
SUMMARY_CSV    = "results/bola_lstm_summary.csv"
FIGURES_DIR    = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

# Keys match CSV "Agent" column — used for data lookup only
PALETTE = {
    "Black-Scholes":    "#E07B39",   # burnt orange  (swapped)
    "Vanilla PPO+LSTM": "#5B8DB8",   # steel blue    (swapped)
    "BOLA+LSTM":        "#7B2D8B",   # purple
}
LINESTYLES = {
    "Black-Scholes":    "solid",
    "Vanilla PPO+LSTM": "solid",
    "BOLA+LSTM":        "solid",
}
LINEWIDTHS = {
    "Black-Scholes":    0.8,
    "Vanilla PPO+LSTM": 0.9,
    "BOLA+LSTM":        1.0,
}
# Vanilla drawn on top of BOLA so both remain visible in fig3
LINE_ZORDER = {
    "Black-Scholes":    3,
    "Vanilla PPO+LSTM": 5,
    "BOLA+LSTM":        4,
}
# No markers on any agent
MARKERS = {
    "Black-Scholes":    None,
    "Vanilla PPO+LSTM": None,
    "BOLA+LSTM":        None,
}

ADVANTAGE_RED = "#C0392B"   # red for BOLA-advantage callout
AGENT_ORDER = ["Black-Scholes", "Vanilla PPO+LSTM", "BOLA+LSTM"]

# Human-readable display names
DISPLAY_NAMES = {
    "Black-Scholes":    "Black-Scholes",
    "Vanilla PPO+LSTM": "Vanilla RL",
    "BOLA+LSTM":        "Prediction Aware RL",
}
TICK_NAMES = {
    "Black-Scholes":    "Black-Scholes",
    "Vanilla PPO+LSTM": "Vanilla RL",
    "BOLA+LSTM":        "Prediction Aware RL",
}

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Calibri", "Gill Sans MT", "Gill Sans",
                           "Trebuchet MS", "Verdana", "Arial",
                           "DejaVu Sans"],   # kept for Unicode glyph fallback
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.93,
    "legend.edgecolor":   "#bbbbbb",
    # all 4 spines visible
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.85,
    "axes.grid":          True,
    "grid.alpha":         0.38,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.55,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F8F8",
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.10,
    "savefig.facecolor":  "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("  Loading data …")
    df      = pd.read_csv(COMPARISON_CSV, parse_dates=["DATE"])
    summary = pd.read_csv(SUMMARY_CSV)
    return df, summary


def make_daily(df):
    """Collapse per-contract rows → one row per (DATE, AGENT)."""
    daily = (
        df.groupby(["DATE", "AGENT"])
        .agg(
            DAILY_PNL  = ("DAILY_PNL",   "sum"),
            TRANS_COST = ("TRANS_COST",  "sum"),
            VIX        = ("VIX",         "mean"),
            HEDGE_DEV  = ("AGENT_HEDGE",
                          lambda x: (x - df.loc[x.index, "DELTA_BS"]).abs().mean()),
        )
        .reset_index()
        .sort_values(["AGENT", "DATE"])
        .reset_index(drop=True)
    )
    daily["CUM_PNL"] = daily.groupby("AGENT")["DAILY_PNL"].cumsum()
    return daily


def rolling_sharpe(series, window=30):
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(252)


def save_fig(fig, name):
    path = f"{FIGURES_DIR}/{name}.png"
    fig.savefig(path)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Cumulative P&L
# ─────────────────────────────────────────────────────────────────────────────

def fig1_cumulative_pnl(daily):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    # Pre-extract aligned series for fill_between
    ppo_sub  = daily[daily["AGENT"] == "Vanilla PPO+LSTM"].sort_values("DATE")
    bola_sub = daily[daily["AGENT"] == "BOLA+LSTM"].sort_values("DATE")

    # Shade the advantage gap between BOLA and Vanilla RL
    ax.fill_between(
        bola_sub["DATE"],
        ppo_sub["CUM_PNL"].values  / 1e6,
        bola_sub["CUM_PNL"].values / 1e6,
        where=(bola_sub["CUM_PNL"].values >= ppo_sub["CUM_PNL"].values),
        color=PALETTE["BOLA+LSTM"], alpha=0.14,
        interpolate=True, zorder=1,
        label="_nolegend_",
    )

    # Plot each agent — no markers
    for agent in AGENT_ORDER:
        sub   = daily[daily["AGENT"] == agent].sort_values("DATE")
        dates = sub["DATE"].values
        vals  = sub["CUM_PNL"].values / 1e6
        ax.plot(
            dates, vals,
            label=DISPLAY_NAMES[agent],
            color=PALETTE[agent],
            linestyle="solid",
            linewidth=LINEWIDTHS[agent],
            alpha=0.93, zorder=3,
        )

    # Extend x-axis right margin — 85 days gives space for all three labels
    last_date  = daily["DATE"].max()
    first_date = daily["DATE"].min()
    ax.set_xlim(left=first_date  - pd.Timedelta(days=30),
                right=last_date + pd.Timedelta(days=85))

    # Y-axis padding based on data range (not value %) so 0.0M gets proper space
    y_all   = daily["CUM_PNL"].values / 1e6
    y_range = y_all.max() - y_all.min()
    ax.set_ylim(bottom=y_all.min() - y_range * 0.06,
                top=y_all.max()   + y_range * 0.13)

    # Final-value labels — same small x offset for all, staggered only in y
    # (xoff_pts, yoff_pts, ha, va)
    label_cfg = {
        "BOLA+LSTM":        (+7, +14, "left", "bottom"),  # above line
        "Vanilla PPO+LSTM": (+7, -14, "left", "top"),     # below line
        "Black-Scholes":    (+7,  +9, "left", "bottom"),  # above noisy end
    }
    for agent in AGENT_ORDER:
        sub  = daily[daily["AGENT"] == agent].sort_values("DATE")
        last = sub.iloc[-1]
        xoff, yoff, ha, va = label_cfg[agent]
        ax.annotate(
            f"{last['CUM_PNL']/1e6:.2f}M",
            xy=(last["DATE"], last["CUM_PNL"] / 1e6),
            xytext=(xoff, yoff), textcoords="offset points",
            fontsize=8.5, color=PALETTE[agent],
            ha=ha, va=va, fontweight="bold",
        )

    # BOLA-advantage callout — red text box, anchored inside lower-right
    bola_last = bola_sub.iloc[-1]["CUM_PNL"]
    ppo_last  = ppo_sub.iloc[-1]["CUM_PNL"]
    gap_pct   = (bola_last - ppo_last) / abs(ppo_last) * 100
    ax.text(
        0.985, 0.06,
        f"Prediction Aware RL vs Vanilla RL:  "
        f"+{(bola_last - ppo_last)/1e3:.0f}K INR  ({gap_pct:+.1f}%)",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8.2, color=ADVANTAGE_RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec=ADVANTAGE_RED, lw=0.9, alpha=0.92),
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.set_title("Cumulative Hedging P&L — Test Period (Jan 2024 – Apr 2026)",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L (INR millions)")
    ax.legend(frameon=True, loc="upper left",
              handlelength=2.8, borderpad=0.8, labelspacing=0.5)

    fig.tight_layout()
    save_fig(fig, "fig1_cumulative_pnl")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig2_performance_metrics(summary):
    cfg = [
        # (column,            title,                       unit,  better_high)
        ("Sharpe_Ann",       "Annualised Sharpe Ratio",   "",    True),
        ("Tracking_Error",   "Tracking Error",            "",    False),
        ("Mean_Daily_PNL",   "Mean Daily P&L",            "",    True),
        ("Total_Trans_Cost", "Total Transaction Cost",    "M INR", False),
    ]

    handles = [mpatches.Patch(facecolor=PALETTE[a], label=DISPLAY_NAMES[a])
               for a in AGENT_ORDER]

    def _draw_metric(ax, col, title, unit, better_high):
        x      = np.arange(len(AGENT_ORDER))
        scale  = 1e-6 if unit == "M INR" else 1.0
        vals   = [
            float(summary.loc[summary["Agent"] == a, col].values[0]) * scale
            for a in AGENT_ORDER
        ]
        bars = ax.bar(x, vals, width=0.55, color=[PALETTE[a] for a in AGENT_ORDER],
                      edgecolor="white", linewidth=0.8, zorder=3)
        y_pad = max(abs(v) for v in vals) * 0.025
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_pad,
                    f"{v:.2f}" if abs(v) < 500 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_ylim(bottom=0, top=max(vals) * 1.20)
        note = "higher is better" if better_high else "lower is better"
        ncol = "#1A7A4A"          if better_high else "#B03A2E"
        ax.text(0.97, 0.97, note, transform=ax.transAxes,
                fontsize=7.5, color=ncol, ha="right", va="top", style="italic")
        labels = [TICK_NAMES[a].replace(" ", "\n", 1) for a in AGENT_ORDER]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8, linespacing=1.3)
        ax.set_title(title, fontweight="bold", pad=8, fontsize=10)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, alpha=0.40, linestyle="--", zorder=0)
        ax.set_axisbelow(True)

    def _add_legend_and_title(fig, title):
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.04), frameon=True,
                   fontsize=9, edgecolor="#bbbbbb")
        fig.suptitle(title, fontweight="bold", fontsize=11, y=1.10)

    # ── combined 1×4 ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.4))
    for ax, (col, title, unit, better_high) in zip(axes, cfg):
        _draw_metric(ax, col, title, unit, better_high)
    _add_legend_and_title(fig, "Performance Metrics Summary — Test Set (2024–2026)")
    fig.tight_layout(w_pad=2.8)
    save_fig(fig, "fig2_performance_metrics")
    plt.close(fig)

    # ── pair A: Sharpe + Tracking Error ───────────────────────────────────────
    fig_a, axes_a = plt.subplots(1, 2, figsize=(8, 4.4))
    for ax, (col, title, unit, better_high) in zip(axes_a, cfg[:2]):
        _draw_metric(ax, col, title, unit, better_high)
    _add_legend_and_title(fig_a, "Risk-Adjusted Performance — Test Set (2024–2026)")
    fig_a.tight_layout(w_pad=2.8)
    save_fig(fig_a, "fig2_sharpe_tracking")
    plt.close(fig_a)

    # ── pair B: Mean Daily P&L + Total Transaction Cost ───────────────────────
    fig_b, axes_b = plt.subplots(1, 2, figsize=(8, 4.4))
    for ax, (col, title, unit, better_high) in zip(axes_b, cfg[2:]):
        _draw_metric(ax, col, title, unit, better_high)
    _add_legend_and_title(fig_b, "P&L and Cost Summary — Test Set (2024–2026)")
    fig_b.tight_layout(w_pad=2.8)
    save_fig(fig_b, "fig2_pnl_cost")
    plt.close(fig_b)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Rolling 30-day Sharpe + VIX strip
# ─────────────────────────────────────────────────────────────────────────────

def fig3_rolling_sharpe(daily):
    fig, (ax_top, ax_vix) = plt.subplots(
        2, 1, figsize=(9, 5.4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
        sharex=True,
    )

    rs_series = {}
    for agent in AGENT_ORDER:
        sub = daily[daily["AGENT"] == agent].set_index("DATE").sort_index()
        rs  = rolling_sharpe(sub["DAILY_PNL"], window=30)
        rs_series[agent] = rs
        ax_top.plot(rs.index, rs,
                    label=DISPLAY_NAMES[agent],
                    color=PALETTE[agent],
                    linestyle="solid",
                    linewidth=LINEWIDTHS[agent],
                    alpha=0.90, zorder=LINE_ZORDER[agent])
        if MARKERS[agent] is not None:
            mk, ms, ma = MARKERS[agent]
            valid = rs.dropna()
            step  = max(1, len(valid) // 16)
            ax_top.plot(valid.index[::step], valid.values[::step],
                        marker=mk, markersize=ms, color=PALETTE[agent],
                        alpha=ma, linestyle="none", zorder=4)

    # Shade BOLA advantage over Vanilla RL in Sharpe
    rs_ppo  = rs_series["Vanilla PPO+LSTM"]
    rs_bola = rs_series["BOLA+LSTM"]
    idx     = rs_ppo.index.intersection(rs_bola.index)
    ax_top.fill_between(idx,
                        rs_ppo.reindex(idx),
                        rs_bola.reindex(idx),
                        where=(rs_bola.reindex(idx) >= rs_ppo.reindex(idx)),
                        color=PALETTE["BOLA+LSTM"], alpha=0.13,
                        interpolate=True, zorder=1)

    ax_top.axhline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=2)
    ax_top.set_ylabel("Rolling Sharpe Ratio (ann.)")
    ax_top.set_title(
        "30-Day Rolling Sharpe Ratio with India VIX — Test Period",
        fontweight="bold", pad=10)
    ax_top.legend(frameon=True, loc="upper left",
                  handlelength=2.5, borderpad=0.8)

    vix = (daily[daily["AGENT"] == "Black-Scholes"]
           .set_index("DATE")["VIX"].sort_index())
    ax_vix.fill_between(vix.index, vix.values,
                        color="#9B59B6", alpha=0.50, linewidth=0,
                        label="India VIX")
    ax_vix.plot(vix.index, vix.values,
                color="#7D3C98", linewidth=0.8, alpha=0.75)
    ax_vix.set_ylabel("VIX (%)", fontsize=9)
    ax_vix.set_xlabel("Date")
    ax_vix.legend(frameon=True, fontsize=8, loc="upper left", borderpad=0.6)

    ax_vix.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_vix.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b '%y"))
    plt.setp(ax_vix.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.tight_layout()
    save_fig(fig, "fig3_rolling_sharpe")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — VIX Regime Analysis
# ─────────────────────────────────────────────────────────────────────────────

def fig4_vix_regime(daily):
    bs_vix = (daily[daily["AGENT"] == "Black-Scholes"]
              [["DATE", "VIX"]].copy())
    q33, q67 = bs_vix["VIX"].quantile([1/3, 2/3]).values

    rlabels = [
        f"Low VIX\n(<= {q33:.1f}%)",
        f"Medium VIX\n({q33:.1f} - {q67:.1f}%)",
        f"High VIX\n(> {q67:.1f}%)",
    ]

    def bucket(v):
        if v <= q33:   return rlabels[0]
        elif v <= q67: return rlabels[1]
        return rlabels[2]

    bs_vix["REGIME"] = bs_vix["VIX"].apply(bucket)
    rmap   = dict(zip(bs_vix["DATE"], bs_vix["REGIME"]))
    daily2 = daily.copy()
    daily2["REGIME"] = daily2["DATE"].map(rmap)
    daily2 = daily2.dropna(subset=["REGIME"])

    n_r = len(rlabels)
    n_a = len(AGENT_ORDER)
    bw  = 0.70 / n_a
    off = np.linspace(-(0.70 - bw) / 2, (0.70 - bw) / 2, n_a)
    xr  = np.arange(n_r)

    handles = [mpatches.Patch(facecolor=PALETTE[a], label=DISPLAY_NAMES[a])
               for a in AGENT_ORDER]

    def _draw_regime(ax, col, ylabel, title):
        all_means = []
        for i, agent in enumerate(AGENT_ORDER):
            means = [
                daily2[(daily2["AGENT"] == agent) & (daily2["REGIME"] == r)][col].mean()
                for r in rlabels
            ]
            all_means.extend(means)
            bars = ax.bar(xr + off[i], means, width=bw,
                          color=PALETTE[agent], label=DISPLAY_NAMES[agent],
                          edgecolor="white", linewidth=0.6, zorder=3)
            for bar, v in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(bar.get_height()) * 0.025,
                        f"{v:.0f}",
                        ha="center", va="bottom",
                        fontsize=6.5, rotation=0, color="#333333")
        ax.set_ylim(bottom=0, top=max(all_means) * 1.20)
        ax.set_xticks(xr)
        ax.set_xticklabels(rlabels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=8)
        ax.yaxis.grid(True, alpha=0.40, linestyle="--", zorder=0)
        ax.set_axisbelow(True)

    def _add_legend_and_title(fig, title):
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.04), frameon=True,
                   fontsize=9, edgecolor="#bbbbbb")
        fig.suptitle(title, fontweight="bold", fontsize=11, y=1.10)

    # ── combined 1×2 ──────────────────────────────────────────────────────────
    fig, (ax_pnl, ax_tc) = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_regime(ax_pnl, "DAILY_PNL",  "Mean Daily P&L (INR)",        "Mean Daily P&L by VIX Regime")
    _draw_regime(ax_tc,  "TRANS_COST", "Mean Daily Trans. Cost (INR)", "Mean Daily Trans. Cost by VIX Regime")
    _add_legend_and_title(fig, "Agent Performance Across VIX Regimes — Test Set (2024–2026)")
    fig.tight_layout(w_pad=3.5)
    save_fig(fig, "fig4_vix_regime_analysis")
    plt.close(fig)

    # ── standalone: Mean Daily P&L ────────────────────────────────────────────
    fig_pnl, ax_p = plt.subplots(1, 1, figsize=(6, 4.5))
    _draw_regime(ax_p, "DAILY_PNL", "Mean Daily P&L (INR)", "Mean Daily P&L by VIX Regime")
    _add_legend_and_title(fig_pnl, "Mean Daily P&L Across VIX Regimes — Test Set (2024–2026)")
    fig_pnl.tight_layout()
    save_fig(fig_pnl, "fig4a_pnl_by_regime")
    plt.close(fig_pnl)

    # ── standalone: Transaction Cost ─────────────────────────────────────────
    fig_tc, ax_t = plt.subplots(1, 1, figsize=(6, 4.5))
    _draw_regime(ax_t, "TRANS_COST", "Mean Daily Trans. Cost (INR)", "Mean Daily Trans. Cost by VIX Regime")
    _add_legend_and_title(fig_tc, "Mean Daily Transaction Cost Across VIX Regimes — Test Set (2024–2026)")
    fig_tc.tight_layout()
    save_fig(fig_tc, "fig4b_cost_by_regime")
    plt.close(fig_tc)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Daily P&L Distribution: Violin + ECDF
# ─────────────────────────────────────────────────────────────────────────────

def fig5_pnl_distribution(daily):
    data = [daily[daily["AGENT"] == a]["DAILY_PNL"].values for a in AGENT_ORDER]

    handles = [mpatches.Patch(facecolor=PALETTE[a], label=DISPLAY_NAMES[a])
               for a in AGENT_ORDER]

    def _draw_violin(ax):
        parts = ax.violinplot(data, positions=range(len(AGENT_ORDER)),
                              showmedians=False, showextrema=False, widths=0.68)
        for pc, agent in zip(parts["bodies"], AGENT_ORDER):
            pc.set_facecolor(PALETTE[agent])
            pc.set_alpha(0.65)
            pc.set_edgecolor(PALETTE[agent])
            pc.set_linewidth(1.2)
        bp = ax.boxplot(
            data, positions=range(len(AGENT_ORDER)),
            widths=0.14, patch_artist=True,
            medianprops  = dict(color="white",   linewidth=2.0),
            whiskerprops = dict(color="#444444", linewidth=1.0),
            capprops     = dict(color="#444444", linewidth=1.0),
            flierprops   = dict(marker=".", markersize=1.8,
                                alpha=0.20, color="#999999"),
            zorder=4,
        )
        for patch, agent in zip(bp["boxes"], AGENT_ORDER):
            patch.set_facecolor(PALETTE[agent])
            patch.set_alpha(0.90)
            patch.set_linewidth(0)
        ax.set_xticks(range(len(AGENT_ORDER)))
        ax.set_xticklabels([TICK_NAMES[a] for a in AGENT_ORDER], fontsize=9.5)
        ax.set_ylabel("Daily P&L (INR)")
        ax.set_title("Daily P&L Distribution  (Violin + Box)", fontweight="bold", pad=8)
        ax.axhline(0, color="#888888", linewidth=0.9, linestyle=":", zorder=2,
                   label="Break-even")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(frameon=True, fontsize=8, loc="lower right", borderpad=0.6)

    def _draw_ecdf(ax):
        for agent in AGENT_ORDER:
            pnl  = np.sort(daily[daily["AGENT"] == agent]["DAILY_PNL"].values)
            ecdf = np.arange(1, len(pnl) + 1) / len(pnl)
            ax.plot(pnl, ecdf,
                    label=DISPLAY_NAMES[agent],
                    color=PALETTE[agent],
                    linestyle="solid",
                    linewidth=LINEWIDTHS[agent],
                    alpha=0.90, zorder=3)
            if MARKERS[agent] is not None:
                mk, ms, ma = MARKERS[agent]
                step = max(1, len(pnl) // 14)
                ax.plot(pnl[::step], ecdf[::step],
                        marker=mk, markersize=ms, color=PALETTE[agent],
                        alpha=ma, linestyle="none", zorder=4)
        ax.axvline(0, color="#888888", linewidth=0.9, linestyle="--", zorder=2)
        ax.set_xlabel("Daily P&L (INR)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF of Daily P&L", fontweight="bold", pad=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(frameon=True, loc="lower right", handlelength=2.5, borderpad=0.8)

    def _add_legend_and_title(fig, title):
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.04), frameon=True,
                   fontsize=9, edgecolor="#bbbbbb")
        fig.suptitle(title, fontweight="bold", fontsize=11, y=1.10)

    # ── combined 1×2 ──────────────────────────────────────────────────────────
    fig, (ax_vln, ax_ecdf) = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_violin(ax_vln)
    _draw_ecdf(ax_ecdf)
    _add_legend_and_title(fig, "Daily P&L Risk Profile — Test Set (2024–2026)")
    fig.tight_layout(w_pad=3.5)
    save_fig(fig, "fig5_pnl_distribution")
    plt.close(fig)

    # ── standalone: Violin + Box ───────────────────────────────────────────────
    fig_v, ax_v = plt.subplots(1, 1, figsize=(6, 4.5))
    _draw_violin(ax_v)
    _add_legend_and_title(fig_v, "Daily P&L Distribution — Test Set (2024–2026)")
    fig_v.tight_layout()
    save_fig(fig_v, "fig5a_violin")
    plt.close(fig_v)

    # ── standalone: ECDF ──────────────────────────────────────────────────────
    fig_e, ax_e = plt.subplots(1, 1, figsize=(6, 4.5))
    _draw_ecdf(ax_e)
    _add_legend_and_title(fig_e, "Empirical CDF of Daily P&L — Test Set (2024–2026)")
    fig_e.tight_layout()
    save_fig(fig_e, "fig5b_ecdf")
    plt.close(fig_e)


# ─────────────────────────────────────────────────────────────────────────────
# FIG COMBINED — all 5 panels, A4-landscape
# ─────────────────────────────────────────────────────────────────────────────

def fig_combined(daily, summary):
    # VIX regime pre-computation
    bs_vix = daily[daily["AGENT"] == "Black-Scholes"][["DATE", "VIX"]].copy()
    q33, q67 = bs_vix["VIX"].quantile([1/3, 2/3]).values
    rlabels  = [f"Low\n<={q33:.0f}%",
                f"Mid\n{q33:.0f}-{q67:.0f}%",
                f"High\n>{q67:.0f}%"]

    def bucket(v):
        if v <= q33:   return rlabels[0]
        elif v <= q67: return rlabels[1]
        return rlabels[2]

    bs_vix["REGIME"] = bs_vix["VIX"].apply(bucket)
    rmap   = dict(zip(bs_vix["DATE"], bs_vix["REGIME"]))
    daily2 = daily.copy()
    daily2["REGIME"] = daily2["DATE"].map(rmap)
    daily2 = daily2.dropna(subset=["REGIME"])

    fig = plt.figure(figsize=(18, 13))
    gs  = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.42)

    ax_cum  = fig.add_subplot(gs[0, :])
    ax_shr  = fig.add_subplot(gs[1, :2])
    ax_vix  = fig.add_subplot(gs[1, 2:])
    ax_vln  = fig.add_subplot(gs[2, :2])
    ax_ecdf = fig.add_subplot(gs[2, 2:])

    panels = iter(["(a)", "(b)", "(c)", "(d)", "(e)"])

    # (a) Cumulative P&L with BOLA-advantage shading
    ppo_sub_c  = daily[daily["AGENT"] == "Vanilla PPO+LSTM"].sort_values("DATE")
    bola_sub_c = daily[daily["AGENT"] == "BOLA+LSTM"].sort_values("DATE")
    ax_cum.fill_between(
        bola_sub_c["DATE"],
        ppo_sub_c["CUM_PNL"].values  / 1e6,
        bola_sub_c["CUM_PNL"].values / 1e6,
        where=(bola_sub_c["CUM_PNL"].values >= ppo_sub_c["CUM_PNL"].values),
        color=PALETTE["BOLA+LSTM"], alpha=0.13, interpolate=True, zorder=1,
    )
    for agent in AGENT_ORDER:
        sub  = daily[daily["AGENT"] == agent].sort_values("DATE")
        ax_cum.plot(sub["DATE"], sub["CUM_PNL"] / 1e6,
                    label=DISPLAY_NAMES[agent],
                    color=PALETTE[agent], linestyle="solid",
                    linewidth=LINEWIDTHS[agent], alpha=0.93, zorder=3)
        if MARKERS[agent] is not None:
            mk, ms, ma = MARKERS[agent]
            step = max(1, len(sub) // 18)
            ax_cum.plot(sub["DATE"].values[::step],
                        (sub["CUM_PNL"].values / 1e6)[::step],
                        marker=mk, markersize=ms - 0.5, color=PALETTE[agent],
                        alpha=ma, linestyle="none", zorder=4)
        last = sub.iloc[-1]
        ax_cum.annotate(f"  {last['CUM_PNL']/1e6:.2f}M",
                        xy=(last["DATE"], last["CUM_PNL"] / 1e6),
                        fontsize=8, color=PALETTE[agent],
                        va="center", fontweight="bold")
    ax_cum.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax_cum.xaxis.set_major_locator(
        mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_cum.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b '%y"))
    plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_cum.set_ylabel("Cumulative P&L (INR M)")
    ax_cum.set_title(f"{next(panels)}  Cumulative Hedging P&L",
                     fontweight="bold")
    ax_cum.legend(frameon=True, ncol=3, loc="upper left",
                  handlelength=2.5, borderpad=0.7)

    # (b) Rolling Sharpe with BOLA advantage shading
    rs_c = {}
    for agent in AGENT_ORDER:
        sub = daily[daily["AGENT"] == agent].set_index("DATE").sort_index()
        rs  = rolling_sharpe(sub["DAILY_PNL"], window=30)
        rs_c[agent] = rs
        ax_shr.plot(rs.index, rs, label=DISPLAY_NAMES[agent],
                    color=PALETTE[agent], linestyle="solid",
                    linewidth=LINEWIDTHS[agent], alpha=0.90, zorder=3)
        if MARKERS[agent] is not None:
            mk, ms, ma = MARKERS[agent]
            valid = rs.dropna()
            step  = max(1, len(valid) // 14)
            ax_shr.plot(valid.index[::step], valid.values[::step],
                        marker=mk, markersize=ms - 0.5, color=PALETTE[agent],
                        alpha=ma, linestyle="none", zorder=4)
    idx_c = rs_c["Vanilla PPO+LSTM"].index.intersection(rs_c["BOLA+LSTM"].index)
    ax_shr.fill_between(
        idx_c,
        rs_c["Vanilla PPO+LSTM"].reindex(idx_c),
        rs_c["BOLA+LSTM"].reindex(idx_c),
        where=(rs_c["BOLA+LSTM"].reindex(idx_c) >= rs_c["Vanilla PPO+LSTM"].reindex(idx_c)),
        color=PALETTE["BOLA+LSTM"], alpha=0.13, interpolate=True, zorder=1,
    )
    ax_shr.axhline(0, color="#777", linewidth=0.9, linestyle="--", zorder=2)
    ax_shr.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 7]))
    ax_shr.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b '%y"))
    plt.setp(ax_shr.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_shr.set_ylabel("Rolling Sharpe (ann.)")
    ax_shr.set_title(f"{next(panels)}  30-Day Rolling Sharpe Ratio",
                     fontweight="bold")
    ax_shr.legend(frameon=True, loc="lower right",
                  handlelength=2.2, borderpad=0.6, fontsize=8)

    # (c) VIX Regime Mean P&L
    n_r  = len(rlabels)
    n_a  = len(AGENT_ORDER)
    bw   = 0.68 / n_a
    off  = np.linspace(-(0.68 - bw) / 2, (0.68 - bw) / 2, n_a)
    xr   = np.arange(n_r)
    for i, agent in enumerate(AGENT_ORDER):
        means = [
            daily2[(daily2["AGENT"] == agent) & (daily2["REGIME"] == r)
                   ]["DAILY_PNL"].mean()
            for r in rlabels
        ]
        ax_vix.bar(xr + off[i], means, width=bw,
                   color=PALETTE[agent], label=DISPLAY_NAMES[agent],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax_vix.set_xticks(xr)
    ax_vix.set_xticklabels(rlabels, fontsize=9)
    ax_vix.set_ylabel("Mean Daily P&L (INR)")
    ax_vix.set_title(f"{next(panels)}  Mean P&L by VIX Regime",
                     fontweight="bold")
    ax_vix.yaxis.grid(True, alpha=0.40, linestyle="--", zorder=0)
    ax_vix.set_axisbelow(True)
    ax_vix.legend(frameon=True, fontsize=8, loc="upper right",
                  borderpad=0.6, handlelength=1.5)

    # (d) Violin
    data = [daily[daily["AGENT"] == a]["DAILY_PNL"].values for a in AGENT_ORDER]
    parts = ax_vln.violinplot(data, positions=range(n_a),
                              showmedians=False, showextrema=False, widths=0.65)
    for pc, agent in zip(parts["bodies"], AGENT_ORDER):
        pc.set_facecolor(PALETTE[agent]); pc.set_alpha(0.65)
        pc.set_edgecolor(PALETTE[agent]); pc.set_linewidth(1.0)
    bp = ax_vln.boxplot(
        data, positions=range(n_a), widths=0.13, patch_artist=True,
        medianprops  = dict(color="white",   linewidth=1.8),
        whiskerprops = dict(color="#444444", linewidth=0.9),
        capprops     = dict(color="#444444", linewidth=0.9),
        flierprops   = dict(marker=".", markersize=1.3,
                            alpha=0.18, color="#999999"),
        zorder=4,
    )
    for patch, agent in zip(bp["boxes"], AGENT_ORDER):
        patch.set_facecolor(PALETTE[agent]); patch.set_alpha(0.90)
        patch.set_linewidth(0)
    ax_vln.set_xticks(range(n_a))
    ax_vln.set_xticklabels([TICK_NAMES[a] for a in AGENT_ORDER])
    ax_vln.set_ylabel("Daily P&L (INR)")
    ax_vln.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=2)
    ax_vln.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_vln.set_title(f"{next(panels)}  Daily P&L Distribution",
                     fontweight="bold")

    # (e) ECDF
    for agent in AGENT_ORDER:
        pnl  = np.sort(daily[daily["AGENT"] == agent]["DAILY_PNL"].values)
        ecdf = np.arange(1, len(pnl) + 1) / len(pnl)
        ax_ecdf.plot(pnl, ecdf, label=DISPLAY_NAMES[agent],
                     color=PALETTE[agent], linestyle="solid",
                     linewidth=LINEWIDTHS[agent], alpha=0.90, zorder=3)
        if MARKERS[agent] is not None:
            mk, ms, ma = MARKERS[agent]
            step = max(1, len(pnl) // 12)
            ax_ecdf.plot(pnl[::step], ecdf[::step],
                         marker=mk, markersize=ms - 0.5, color=PALETTE[agent],
                         alpha=ma, linestyle="none", zorder=4)
    ax_ecdf.axvline(0, color="#888", linewidth=0.8, linestyle="--", zorder=2)
    ax_ecdf.set_xlabel("Daily P&L (INR)")
    ax_ecdf.set_ylabel("Cumulative Probability")
    ax_ecdf.set_title(f"{next(panels)}  Empirical CDF of Daily P&L",
                      fontweight="bold")
    ax_ecdf.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_ecdf.legend(frameon=True, loc="lower right",
                   handlelength=2.0, borderpad=0.6, fontsize=8)

    fig.suptitle(
        "Prediction Aware RL  vs  Vanilla RL  vs  Black-Scholes  —  "
        "Test-Period Performance Summary  (Jan 2024 – Apr 2026)",
        fontweight="bold", fontsize=12.5, y=0.998,
    )
    save_fig(fig, "fig_all_combined")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  BOLA Report Figures  —  generating 6 PNG files")
    print("=" * 62)

    df, summary = load_data()

    print("  Aggregating to daily granularity …")
    daily = make_daily(df)

    print("  Rendering figures …")
    fig1_cumulative_pnl(daily)
    fig2_performance_metrics(summary)
    fig3_rolling_sharpe(daily)
    fig4_vix_regime(daily)
    fig5_pnl_distribution(daily)
    fig_combined(daily, summary)

    plt.close("all")
    print("=" * 62)
    print(f"  Done.  All figures → {FIGURES_DIR}/")
    print("=" * 62)
