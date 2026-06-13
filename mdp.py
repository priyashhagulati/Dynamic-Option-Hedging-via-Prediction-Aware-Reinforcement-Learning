"""
BOLA Project - MDP Flow Diagram Generator
Generates all diagrams as high-quality PNG images using matplotlib
Run: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

os.makedirs("diagrams", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "agent_dark":       "#1a1a2e",
    "agent_mid":        "#16213e",
    "agent_light":      "#0f3460",
    "agent_accent":     "#e94560",
    "env_dark":         "#0d3b2e",
    "env_mid":          "#1a5c42",
    "env_light":        "#2d8a5e",
    "env_accent":       "#4ecca3",
    "reward_dark":      "#2d1b69",
    "reward_mid":       "#5a2d82",
    "reward_light":     "#8b5cf6",
    "reward_accent":    "#f59e0b",
    "obs_dark":         "#1e3a5f",
    "obs_light":        "#3b82f6",
    "obs_accent":       "#93c5fd",
    "imp1":             "#dc2626",
    "imp2":             "#d97706",
    "imp3":             "#059669",
    "imp4":             "#7c3aed",
    "arrow":            "#94a3b8",
    "arrow_bright":     "#f1f5f9",
    "text_primary":     "#f8fafc",
    "text_secondary":   "#cbd5e1",
    "text_dark":        "#1e293b",
    "bg_main":          "#0a0a1a",
    "bg_card":          "#111827",
    "bg_card2":         "#1f2937",
    "grid":             "#1e293b",
    "bs_color":         "#f97316",
    "ppo_color":        "#06b6d4",
    "bola_color":       "#a855f7",
}

def make_gradient_background(ax, color1, color2, alpha=1.0):
    """Add gradient background to axes."""
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax.imshow(gradient, aspect='auto', extent=[0, 1, 0, 1],
              origin='lower', transform=ax.transAxes,
              cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                  'bg', [color1, color2]),
              alpha=alpha, zorder=0)

def rounded_box(ax, x, y, w, h, color, alpha=0.9, radius=0.02, 
                edgecolor=None, linewidth=1.5, zorder=2):
    """Draw a rounded rectangle."""
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=color, edgecolor=ec,
                          linewidth=linewidth, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color="#94a3b8", 
               lw=2, style="-|>", zorder=3, mutation_scale=15):
    """Draw a fancy arrow."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    lw=lw,
                    mutation_scale=mutation_scale,
                    connectionstyle="arc3,rad=0.0"
                ), zorder=zorder)

def text_with_shadow(ax, x, y, text, fontsize=10, color="white",
                     ha="center", va="center", fontweight="normal",
                     zorder=5, alpha=1.0):
    """Draw text with drop shadow."""
    ax.text(x + 0.002, y - 0.002, text, fontsize=fontsize,
            color="black", ha=ha, va=va, fontweight=fontweight,
            zorder=zorder-1, alpha=0.5,
            transform=ax.transAxes if False else ax.transData)
    return ax.text(x, y, text, fontsize=fontsize, color=color,
                   ha=ha, va=va, fontweight=fontweight,
                   zorder=zorder, alpha=alpha)


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: Agent-Environment Overview
# ═════════════════════════════════════════════════════════════════════════════

def diagram_1_agent_env_overview():
    fig = plt.figure(figsize=(20, 14), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(10, 13.3, "BOLA Hedging System — Agent & Environment Overview",
            fontsize=22, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center", zorder=10)
    ax.text(10, 12.85, "Nifty Options Delta-Hedging with Prediction-Aware RL",
            fontsize=13, color=COLORS["text_secondary"], ha="center", va="center")

    # ── ENVIRONMENT box ──────────────────────────────────────────────────────
    rounded_box(ax, 0.4, 7.2, 8.8, 5.2, COLORS["env_dark"],
                edgecolor=COLORS["env_accent"], linewidth=2.5, zorder=2)
    ax.text(4.8, 12.05, "ENVIRONMENT", fontsize=15, fontweight="bold",
            color=COLORS["env_accent"], ha="center", va="center", zorder=6)
    ax.text(4.8, 11.7, "Nifty Options Market  (NiftyHedgingEnv)",
            fontsize=10, color=COLORS["text_secondary"], ha="center", zorder=6)

    env_items = [
        ("SPOT Price  Sₜ",    "#4ecca3", 1.2,  10.6),
        ("VIX Sigma  σ_imp",  "#4ecca3", 1.2,  9.8),
        ("BS Delta  Δ_BS",    "#4ecca3", 1.2,  9.0),
        ("Time to Expiry  T", "#4ecca3", 1.2,  8.2),
        ("Option Price  Cₜ",  "#86efac", 5.0,  10.6),
        ("Moneyness  S/K",    "#86efac", 5.0,  9.8),
        ("Strike  K",         "#86efac", 5.0,  9.0),
        ("Expiry Date",       "#86efac", 5.0,  8.2),
    ]
    for label, col, ex, ey in env_items:
        rounded_box(ax, ex - 0.1, ey - 0.3, 3.2, 0.55,
                    COLORS["env_mid"], edgecolor=col,
                    linewidth=1.5, alpha=0.85, zorder=4)
        ax.text(ex + 1.5, ey, label, fontsize=9.5, color=col,
                ha="center", va="center", fontweight="bold", zorder=6)

    # ── AGENT box ────────────────────────────────────────────────────────────
    rounded_box(ax, 10.8, 7.2, 8.8, 5.2, COLORS["agent_dark"],
                edgecolor=COLORS["agent_accent"], linewidth=2.5, zorder=2)
    ax.text(15.2, 12.05, "BOLA AGENT", fontsize=15, fontweight="bold",
            color=COLORS["agent_accent"], ha="center", va="center", zorder=6)
    ax.text(15.2, 11.7, "PPO Actor-Critic + LSTM + 4 Improvements",
            fontsize=10, color=COLORS["text_secondary"], ha="center", zorder=6)

    agent_stages = [
        ("LSTM / Attention-LSTM\n(K=5 window encoder, hidden=64)",
         COLORS["agent_light"], "#93c5fd", 11.2, 10.3, 8.0, 0.9),
        ("PPO Actor-Critic  MLP(128→128)\nActor: π(a|s)   Critic: V(s)",
         "#312e81",              "#818cf8", 11.2, 9.15, 8.0, 0.9),
        ("BOLA Adaptation  (IMP 1-4)\nBJ Gap · RH Planning · MAE Gate · Band",
         "#3b0764",              "#c084fc", 11.2, 8.0,  8.0, 0.9),
    ]
    for label, fc, ec, bx, by, bw, bh in agent_stages:
        rounded_box(ax, bx, by, bw, bh, fc, edgecolor=ec,
                    linewidth=2, alpha=0.88, zorder=4)
        ax.text(bx + bw/2, by + bh/2, label, fontsize=9,
                color=COLORS["text_primary"], ha="center", va="center",
                fontweight="bold", zorder=6, linespacing=1.5)

    # ── OBSERVATION flow (ENV → AGENT) ───────────────────────────────────────
    rounded_box(ax, 5.8, 5.0, 8.4, 1.6, COLORS["obs_dark"],
                edgecolor=COLORS["obs_light"], linewidth=2.5, zorder=4)
    ax.text(10.0, 5.8, "OBSERVATION  oₜ  ∈  ℝ³⁵  (BOLA)  /  ℝ³⁰  (Base)",
            fontsize=11, fontweight="bold", color=COLORS["obs_accent"],
            ha="center", va="center", zorder=6)
    obs_parts = [
        "K×5 window (25 dims)", "Current state (4 dims)",
        "Hedge hₜ (1 dim)",     "VIX extras (5 dims)"
    ]
    for i, part in enumerate(obs_parts):
        ax.text(6.6 + i * 2.0, 5.25, part, fontsize=8,
                color=COLORS["text_secondary"], ha="center", va="center", zorder=6)

    # ── ACTION flow (AGENT → ENV) ─────────────────────────────────────────────
    rounded_box(ax, 5.8, 2.8, 8.4, 1.5, "#1c1c3a",
                edgecolor=COLORS["agent_accent"], linewidth=2.5, zorder=4)
    ax.text(10.0, 3.55, "ACTION  aₜ  ∈  [-1, +1]  →  Δhₜ ∈ [-0.20, +0.20]",
            fontsize=11, fontweight="bold", color=COLORS["agent_accent"],
            ha="center", va="center", zorder=6)
    ax.text(10.0, 3.05, "hₜ₊₁ = clip(hₜ + Δhₜ, -1, +1)   |   "
            "Filtered if |Δh| < adaptive_band ∈ [0.005, 0.018]",
            fontsize=8.5, color=COLORS["text_secondary"],
            ha="center", va="center", zorder=6)

    # ── REWARD box ────────────────────────────────────────────────────────────
    rounded_box(ax, 5.8, 0.5, 8.4, 1.8, "#1a0a2e",
                edgecolor=COLORS["reward_light"], linewidth=2.5, zorder=4)
    ax.text(10.0, 1.4, "REWARD  rₜ",
            fontsize=12, fontweight="bold", color=COLORS["reward_accent"],
            ha="center", va="center", zorder=6)
    ax.text(10.0, 0.95,
            "rₜ = pnl_norm  −  0.25·pnl²  −  0.50·(h−Δ_BS)²  −  0.10·cost_norm",
            fontsize=9.5, color=COLORS["text_secondary"],
            ha="center", va="center", zorder=6)

    # ── Arrows ────────────────────────────────────────────────────────────────
    # Env → Obs
    ax.annotate("", xy=(9.0, 6.6), xytext=(4.8, 7.2),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["env_accent"],
                                lw=2.5, mutation_scale=18), zorder=8)
    ax.text(6.2, 7.0, "emits", fontsize=9, color=COLORS["env_accent"],
            ha="center", va="center", style="italic", zorder=8)

    # Obs → Agent
    ax.annotate("", xy=(13.5, 7.2), xytext=(11.5, 6.6),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["obs_accent"],
                                lw=2.5, mutation_scale=18), zorder=8)
    ax.text(13.0, 7.0, "feeds", fontsize=9, color=COLORS["obs_accent"],
            ha="center", va="center", style="italic", zorder=8)

    # Agent → Action
    ax.annotate("", xy=(13.5, 4.3), xytext=(15.2, 7.2),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["agent_accent"],
                                lw=2.5, mutation_scale=18), zorder=8)
    ax.text(14.8, 5.5, "outputs", fontsize=9, color=COLORS["agent_accent"],
            ha="center", va="center", style="italic", zorder=8)

    # Action → Env
    ax.annotate("", xy=(4.8, 7.2), xytext=(7.0, 4.3),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["agent_accent"],
                                lw=2.5, mutation_scale=18), zorder=8)
    ax.text(5.5, 5.6, "modifies", fontsize=9, color=COLORS["agent_accent"],
            ha="center", va="center", style="italic", zorder=8)

    # Action → Reward
    ax.annotate("", xy=(10.0, 2.3), xytext=(10.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["reward_accent"],
                                lw=2.5, mutation_scale=18), zorder=8)

    # Obs → Reward (dashed)
    ax.annotate("", xy=(10.0, 2.3), xytext=(10.0, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["reward_light"],
                                lw=1.5, mutation_scale=12,
                                linestyle="dashed"), zorder=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        ("Environment", COLORS["env_accent"]),
        ("BOLA Agent",  COLORS["agent_accent"]),
        ("Observation", COLORS["obs_light"]),
        ("Action",      COLORS["agent_accent"]),
        ("Reward",      COLORS["reward_accent"]),
    ]
    for i, (lbl, col) in enumerate(legend_items):
        bx = 0.5 + i * 3.8
        rounded_box(ax, bx, 0.1, 3.4, 0.32, COLORS["bg_card"],
                    edgecolor=col, linewidth=1.5, alpha=0.9, zorder=5)
        ax.text(bx + 1.7, 0.26, lbl, fontsize=9, color=col,
                ha="center", va="center", fontweight="bold", zorder=7)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/01_agent_env_overview.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/01_agent_env_overview.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: State Space (Observation Vector)
# ═════════════════════════════════════════════════════════════════════════════

def diagram_2_state_space():
    fig = plt.figure(figsize=(20, 11), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(10, 10.4, "MDP State Space — Observation Vector  oₜ ∈ ℝ³⁵",
            fontsize=21, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(10, 9.95, "Full decomposition of the 35-dimensional BOLA state vector",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    # ── Block 1: K-window (25 dims) ───────────────────────────────────────────
    rounded_box(ax, 0.3, 1.0, 6.2, 8.5, COLORS["obs_dark"],
                edgecolor=COLORS["obs_light"], linewidth=2.5, zorder=2)
    ax.text(3.4, 9.15, "BLOCK 1 — Rolling Window", fontsize=12, fontweight="bold",
            color=COLORS["obs_accent"], ha="center", va="center", zorder=5)
    ax.text(3.4, 8.75, "Past K=5 trading days  ×  5 features = 25 dims",
            fontsize=9.5, color=COLORS["text_secondary"], ha="center", va="center", zorder=5)

    day_cols = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#c084fc"]
    window_feats = ["Spot/S₀", "Time T", "VIX σ", "Delta Δ", "Money"]
    for d in range(5):
        bx, by = 0.6 + d * 1.15, 2.0
        rounded_box(ax, bx, by, 1.0, 6.3, day_cols[d],
                    edgecolor="white", linewidth=1.2,
                    alpha=0.25, zorder=3)
        ax.text(bx + 0.5, 8.45, f"t-{5-d}", fontsize=9,
                color=day_cols[d], ha="center", va="center",
                fontweight="bold", zorder=5)
        for f, feat in enumerate(window_feats):
            fy = 2.4 + f * 1.1
            rounded_box(ax, bx + 0.05, fy - 0.35, 0.9, 0.65,
                        day_cols[d], edgecolor=day_cols[d],
                        linewidth=1, alpha=0.55, zorder=4)
            ax.text(bx + 0.5, fy, feat, fontsize=7.5,
                    color="white", ha="center", va="center",
                    fontweight="bold", zorder=6)

    feat_labels = ["Spot/S₀", "Time T", "VIX σ", "BS Δ", "Money"]
    for f, feat in enumerate(feat_labels):
        ax.text(0.0, 2.37 + f * 1.1, feat, fontsize=7,
                color=COLORS["text_secondary"], ha="right", va="center")

    rounded_box(ax, 0.35, 1.05, 6.1, 0.85, COLORS["obs_dark"],
                edgecolor=COLORS["obs_light"], linewidth=1.5, alpha=0.95, zorder=4)
    ax.text(3.4, 1.5, "dims 0 – 24   (K × 5 = 25 dims)   → fed into LSTM",
            fontsize=9, color=COLORS["obs_accent"],
            ha="center", va="center", fontweight="bold", zorder=6)

    # ── Block 2: Current state (4 dims) ───────────────────────────────────────
    rounded_box(ax, 7.0, 1.0, 3.8, 8.5, "#0f172a",
                edgecolor="#f97316", linewidth=2.5, zorder=2)
    ax.text(8.9, 9.15, "BLOCK 2", fontsize=12, fontweight="bold",
            color="#f97316", ha="center", va="center", zorder=5)
    ax.text(8.9, 8.75, "Current Snapshot  (4 dims)",
            fontsize=9.5, color=COLORS["text_secondary"], ha="center", va="center", zorder=5)

    curr_feats = [
        ("VIX Sigma σ", "Implied volatility\n(clipped 0→2)", "#f97316"),
        ("BS Delta Δ",  "Black-Scholes delta\n∈ [-1, +1]",   "#fb923c"),
        ("Time T",      "Days to expiry\n(normalised)",       "#fed7aa"),
        ("Moneyness",   "S/K ratio\nstrike-adjusted",         "#fef3c7"),
    ]
    for i, (name, desc, col) in enumerate(curr_feats):
        by = 2.2 + i * 1.5
        rounded_box(ax, 7.2, by, 3.4, 1.25, "#1c1917",
                    edgecolor=col, linewidth=1.5, alpha=0.9, zorder=4)
        ax.text(7.7, by + 0.75, name, fontsize=9.5, color=col,
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(9.2, by + 0.75, desc, fontsize=7.5,
                color=COLORS["text_secondary"], ha="center", va="center",
                linespacing=1.4, zorder=6)
        ax.text(10.2, by + 0.38, f"dim {25 + i}", fontsize=7,
                color=col, ha="center", va="center", zorder=6)

    rounded_box(ax, 7.05, 1.05, 3.7, 0.85, "#0f172a",
                edgecolor="#f97316", linewidth=1.5, alpha=0.95, zorder=4)
    ax.text(8.9, 1.5, "dims 25 – 28   (4 dims)",
            fontsize=9, color="#f97316",
            ha="center", va="center", fontweight="bold", zorder=6)

    # ── Block 3: Hedge (1 dim) ────────────────────────────────────────────────
    rounded_box(ax, 11.2, 1.0, 2.4, 8.5, "#0a1628",
                edgecolor="#06b6d4", linewidth=2.5, zorder=2)
    ax.text(12.4, 9.15, "BLOCK 3", fontsize=12, fontweight="bold",
            color="#06b6d4", ha="center", va="center", zorder=5)
    ax.text(12.4, 8.75, "Current Hedge  (1 dim)",
            fontsize=9.5, color=COLORS["text_secondary"], ha="center", va="center", zorder=5)

    rounded_box(ax, 11.4, 4.0, 2.0, 2.8, "#083344",
                edgecolor="#06b6d4", linewidth=2, alpha=0.9, zorder=4)
    ax.text(12.4, 5.65, "hₜ", fontsize=28, color="#06b6d4",
            ha="center", va="center", fontweight="bold", zorder=6)
    ax.text(12.4, 4.7, "Hedge ratio\n∈ [-1, +1]", fontsize=8.5,
            color=COLORS["text_secondary"], ha="center", va="center",
            linespacing=1.4, zorder=6)

    ax.text(12.4, 3.3, "Agent's current\nhedge position", fontsize=8,
            color=COLORS["text_secondary"], ha="center", va="center",
            linespacing=1.4, zorder=6)
    ax.text(12.4, 2.4, "Short stock ← 0 → Long stock", fontsize=7.5,
            color="#67e8f9", ha="center", va="center", zorder=6)

    rounded_box(ax, 11.25, 1.05, 2.3, 0.85, "#0a1628",
                edgecolor="#06b6d4", linewidth=1.5, alpha=0.95, zorder=4)
    ax.text(12.4, 1.5, "dim 29  (1 dim)",
            fontsize=9, color="#06b6d4",
            ha="center", va="center", fontweight="bold", zorder=6)

    # ── Block 4: VIX extras (5 dims) ─────────────────────────────────────────
    rounded_box(ax, 14.0, 1.0, 5.7, 8.5, "#0d0d1f",
                edgecolor=COLORS["bola_color"], linewidth=2.5, zorder=2)
    ax.text(16.85, 9.15, "BLOCK 4 — VIX Extras", fontsize=12, fontweight="bold",
            color=COLORS["bola_color"], ha="center", va="center", zorder=5)
    ax.text(16.85, 8.75, "BOLA-only enrichment  (5 dims)",
            fontsize=9.5, color=COLORS["text_secondary"], ha="center", va="center", zorder=5)

    vix_feats = [
        ("VIX Norm",     "VIX / 20-day MA\nclipped [0.5, 3.0]", "#a855f7", "Regime level",  "dim 30"),
        ("VIX Mom",      "5-day VIX diff\nclipped [−0.5, 0.5]", "#c084fc", "Regime change", "dim 31"),
        ("VIX Pred t+1", "LSTM forecast\n1 day ahead",           "#e879f9", "Near forecast", "dim 32"),
        ("VIX Pred t+2", "LSTM forecast\n2 days ahead",          "#f0abfc", "Mid forecast",  "dim 33"),
        ("VIX Pred t+3", "LSTM forecast\n3 days ahead",          "#fae8ff", "Far forecast",  "dim 34"),
    ]
    for i, (name, desc, col, role, dim) in enumerate(vix_feats):
        by = 1.9 + i * 1.2
        rounded_box(ax, 14.2, by, 5.3, 1.0, "#1a0a2e",
                    edgecolor=col, linewidth=1.5, alpha=0.9, zorder=4)
        rounded_box(ax, 14.25, by + 0.05, 1.5, 0.9, col,
                    edgecolor=col, linewidth=1, alpha=0.3, zorder=4)
        ax.text(15.0, by + 0.5, name, fontsize=9, color=col,
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(16.7, by + 0.65, desc, fontsize=7.5,
                color=COLORS["text_secondary"], ha="center", va="center",
                linespacing=1.3, zorder=6)
        ax.text(16.7, by + 0.25, role, fontsize=7,
                color=col, ha="center", va="center",
                style="italic", zorder=6)
        ax.text(19.0, by + 0.5, dim, fontsize=8,
                color=col, ha="center", va="center",
                fontweight="bold", zorder=6)

    rounded_box(ax, 14.05, 1.05, 5.6, 0.85, "#0d0d1f",
                edgecolor=COLORS["bola_color"], linewidth=1.5, alpha=0.95, zorder=4)
    ax.text(16.85, 1.5, "dims 30 – 34   (5 dims)   BOLA only",
            fontsize=9, color=COLORS["bola_color"],
            ha="center", va="center", fontweight="bold", zorder=6)

    # ── Dimension bar ─────────────────────────────────────────────────────────
    sections = [
        (0.3, 6.2,  "#3b82f6", "25", "Window"),
        (6.5, 1.3,  "#f97316", "4",  "Curr"),
        (7.8, 0.8,  "#06b6d4", "1",  "H"),
        (8.6, 2.85, "#a855f7", "5",  "VIX Ext"),
    ]
    scale = 11.45 / 35
    for val, size_d, col, label, name in sections:
        bx = 0.3 + val * scale
        bw = size_d * scale * 10
        rounded_box(ax, bx - 0.05, 0.15, bw, 0.6, col,
                    edgecolor="white", linewidth=1, alpha=0.8, zorder=5)
        ax.text(bx + bw/2 - 0.05, 0.45,
                f"{name} ({label}d)", fontsize=7.5, color="white",
                ha="center", va="center", fontweight="bold", zorder=7)

    ax.text(10.0, 0.07, "Total observation dimension: 25 + 4 + 1 + 5 = 35 (BOLA)  |  30 (Base without VIX extras)",
            fontsize=9, color=COLORS["text_secondary"], ha="center", va="center")

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/02_state_space.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/02_state_space.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: BOLA 4 Improvements
# ═════════════════════════════════════════════════════════════════════════════

def diagram_3_bola_improvements():
    fig = plt.figure(figsize=(22, 16), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(11, 15.3, "BOLA Agent — 4 Improvements in Detail",
            fontsize=22, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(11, 14.85, "How prediction-awareness, planning, confidence, and cost-filtering work together",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    imps = [
        {
            "title": "IMP 1 — Advantage-Based BJ Gap",
            "color": COLORS["imp1"],
            "bx": 0.3, "by": 7.8, "bw": 10.2, "bh": 6.5,
            "steps": [
                ("Input", "obs_base (30-dim)\nobs_vix  (35-dim)\nreward rₜ", "#991b1b"),
                ("Base Advantage",
                 "A_base = rₜ + γ·V_base(s')\n        − V_base(s)\n(TD-advantage, Base model)",
                 "#b91c1c"),
                ("BOLA Advantage",
                 "A_bola = rₜ + γ·V_bola(s')\n        − V_bola(s)\n(TD-advantage, BOLA model)",
                 "#dc2626"),
                ("BJ Gap",
                 "BJ = A_bola − A_base\n(zero-centred, stationary)\nNormalise: BJ_norm = BJ/σ_run",
                 "#ef4444"),
            ],
            "note": "Why advantage-based?\n• Zero-centred (vs raw V diff)\n• Stationary across training\n• Directly measures if BOLA\n  sees a better immediate step",
        },
        {
            "title": "IMP 2 — Receding-Horizon Planning",
            "color": COLORS["imp2"],
            "bx": 11.2, "by": 7.8, "bw": 10.2, "bh": 6.5,
            "steps": [
                ("Start",   "obs_curr = obs_vix\nfirst_action = None\ntotal_value = 0", "#92400e"),
                ("Step k=0","a_0 = BOLA.actor(obs_curr)\nV_0 = BOLA.critic(obs_curr)\nfirst_action = a_0", "#b45309"),
                ("Synthetic", "obs_next = roll_window(\n  obs_curr, VIX_forecast_k)\n(IMP 2: VIX replaces real obs)", "#d97706"),
                ("Repeat k=1,2", "Repeat for k=1,2\nAccumulate: Σ γᵏ·Vₖ\nonly a_0 is executed", "#f59e0b"),
            ],
            "note": "Receding-horizon:\n• K=3 step lookahead\n• VIX forecasts as synthetic\n  next-observations\n• Execute only first action\n• Re-plan every step",
        },
        {
            "title": "IMP 3 — Confidence Gating on α",
            "color": COLORS["imp3"],
            "bx": 0.3, "by": 0.5, "bw": 10.2, "bh": 6.8,
            "steps": [
                ("MAE Track",  "Each step:\n|VIX_pred_t+1 − VIX_actual|\nBuffer last 20 values", "#065f46"),
                ("Rolling MAE","MAE_roll = mean(buffer)\n(20-step rolling window)\nHigh MAE = bad forecasts", "#047857"),
                ("Gate α",    "α_eff = α · sigmoid(−λ·MAE)\nλ = 30\nα_eff → 0 when MAE is high", "#059669"),
                ("Blend Weight","w = α_eff · tanh(1.5·BJ_norm)\nw ∈ [0, α_eff]\nBlend: (1−w)·a_base + w·a_rh", "#10b981"),
            ],
            "note": "Why MAE gating?\n• Bad VIX forecasts → α→0\n• Bad forecasts won't hurt\n  the hedging quality\n• α_eff ≈ α when MAE ≈ 0\n• Fully automatic",
        },
        {
            "title": "IMP 4 — Adaptive No-Trade Band",
            "color": COLORS["imp4"],
            "bx": 11.2, "by": 0.5, "bw": 10.2, "bh": 6.8,
            "steps": [
                ("VIX Regime", "regime = 0.8·max(0, VIX_norm−1)\n+ 4.0·max(0, VIX_mom)\n+ 3.0·forecast_spike\nclip to [0, 1]", "#4c1d95"),
                ("Confidence", "confidence = w / α_eff\n(normalised blend weight)\nHigh w → narrow band", "#5b21b6"),
                ("Band Calc",  "band = 0.010 × (\n  1.35 − 0.70·regime\n  − 0.25·confidence)\nclip to [0.005, 0.018]", "#6d28d9"),
                ("Filter",     "trade = a_blend × 0.20\nif |trade| < band:\n  action = 0  (no trade)\nelse: execute trade", "#7c3aed"),
            ],
            "note": "Band behaviour:\n• Calm market → wider band\n  (fewer, better trades)\n• VIX stress → narrow band\n  (react quickly)\n• Base: 0.010 (tighter than v1)",
        },
    ]

    for imp in imps:
        col  = imp["color"]
        bx, by = imp["bx"], imp["by"]
        bw, bh = imp["bw"], imp["bh"]

        # Outer box
        rounded_box(ax, bx, by, bw, bh, COLORS["bg_card"],
                    edgecolor=col, linewidth=3, alpha=0.95, zorder=2)

        # Title bar
        rounded_box(ax, bx, by + bh - 0.75, bw, 0.72, col,
                    edgecolor=col, linewidth=2, alpha=0.9, zorder=3)
        ax.text(bx + bw/2, by + bh - 0.38, imp["title"],
                fontsize=13, fontweight="bold", color="white",
                ha="center", va="center", zorder=6)

        # Steps
        step_w = (bw - 0.5 - 2.4) / len(imp["steps"])
        for si, (sname, sdesc, scol) in enumerate(imp["steps"]):
            sx = bx + 0.25 + si * step_w
            sy = by + 1.2
            sh = bh - 2.5
            rounded_box(ax, sx, sy, step_w - 0.15, sh, scol,
                        edgecolor="white", linewidth=1.2,
                        alpha=0.75, zorder=4)
            ax.text(sx + (step_w - 0.15)/2, sy + sh - 0.3, sname,
                    fontsize=9, color="white", ha="center", va="center",
                    fontweight="bold", zorder=6)
            ax.text(sx + (step_w - 0.15)/2, sy + sh/2 - 0.1, sdesc,
                    fontsize=7.5, color="#f1f5f9", ha="center", va="center",
                    linespacing=1.45, zorder=6)

            if si < len(imp["steps"]) - 1:
                ax.annotate("",
                    xy=(sx + step_w - 0.02, sy + sh/2),
                    xytext=(sx + step_w - 0.17, sy + sh/2),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                    lw=1.5, mutation_scale=12), zorder=7)

        # Note box
        note_x = bx + bw - 2.35
        note_y = by + 1.2
        note_h = bh - 2.5
        rounded_box(ax, note_x, note_y, 2.25, note_h, COLORS["bg_card2"],
                    edgecolor=col, linewidth=1.5, alpha=0.9, zorder=4)
        ax.text(note_x + 1.12, note_y + note_h - 0.25, "Key Points",
                fontsize=8, color=col, ha="center", va="center",
                fontweight="bold", zorder=6)
        ax.text(note_x + 1.12, note_y + note_h/2 - 0.2, imp["note"],
                fontsize=7.5, color=COLORS["text_secondary"],
                ha="center", va="center", linespacing=1.5, zorder=6)

        # Bottom label
        rounded_box(ax, bx + 0.1, by + 0.08, bw - 0.2, 0.75, col,
                    edgecolor=col, linewidth=1, alpha=0.15, zorder=3)

    # ── Central flow arrow ────────────────────────────────────────────────────
    ax.text(11.0, 7.4, "← All 4 improvements combine to produce the final BOLA action →",
            fontsize=10, color=COLORS["text_secondary"], ha="center", va="center",
            style="italic")

    # Final blend equation
    rounded_box(ax, 3.0, 7.1, 15.5, 0.58, COLORS["bg_card"],
                edgecolor="#64748b", linewidth=1.5, alpha=0.8, zorder=4)
    ax.text(10.75, 7.38,
            "Final Action:   a_t  =  clip( (1 − w) · a_base  +  w · a_RH ,  −1, +1 )   →   "
            "filter( a_t,  adaptive_band )",
            fontsize=10, color=COLORS["text_primary"],
            ha="center", va="center", fontweight="bold", zorder=6)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/03_bola_improvements.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/03_bola_improvements.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4: Step-by-Step MDP Flow
# ═════════════════════════════════════════════════════════════════════════════

def diagram_4_mdp_stepflow():
    fig = plt.figure(figsize=(18, 26), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 26)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(9, 25.4, "BOLA MDP — Step-by-Step Decision Flow",
            fontsize=22, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(9, 25.0, "Complete per-timestep execution pipeline",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    steps = [
        {
            "num": "INIT",
            "title": "Episode Initialisation",
            "color": "#374151",
            "border": "#9ca3af",
            "content": [
                "Sample episode: (EXPIRY_DT, STRIKE, TYPE)",
                "Set h₀ = 0.0  (no initial hedge position)",
                "Set S₀ = SPOT at t=0  (normalisation anchor)",
                "Build initial observation  o₀ ∈ ℝ³⁵",
            ],
            "tag": "Start of contract",
        },
        {
            "num": "01",
            "title": "Observe  oₜ",
            "color": COLORS["obs_dark"],
            "border": COLORS["obs_light"],
            "content": [
                "Build K=5 day rolling window  (25 dims)",
                "Append current market snapshot  (4 dims)",
                "Append current hedge hₜ  (1 dim)",
                "Lookup VIX features for date  (5 dims)  [BOLA]",
                "→  oₜ ∈ ℝ³⁵  is ready",
            ],
            "tag": "STEP 1",
        },
        {
            "num": "02",
            "title": "Encode via LSTM",
            "color": COLORS["agent_dark"],
            "border": "#818cf8",
            "content": [
                "5-day window → LSTM(hidden=64) → h_lstm",
                "Concat: [h_lstm ‖ curr_4 ‖ hₜ ‖ vix_5]",
                "→ 128-dim feature vector  for Actor & Critic",
            ],
            "tag": "STEP 2",
        },
        {
            "num": "03",
            "title": "Dual Policy Query",
            "color": "#1c1c3a",
            "border": "#6366f1",
            "content": [
                "a_base = Base.actor(obs_base)     — vanilla PPO policy",
                "a_rh   = BOLA.RH_plan(obs_vix)   — receding-horizon (IMP 2)",
                "V_base = Base.critic(obs_base)    — for BJ gap",
                "V_bola = BOLA.critic(obs_vix)     — for BJ gap",
            ],
            "tag": "STEP 3",
        },
        {
            "num": "04",
            "title": "Compute BJ Gap  (IMP 1)",
            "color": "#2d0a0a",
            "border": COLORS["imp1"],
            "content": [
                "A_base = rₜ + γ·V_base(s')  −  V_base(s)    [TD-advantage]",
                "A_bola = rₜ + γ·V_bola(s')  −  V_bola(s)    [TD-advantage]",
                "BJ_raw  = A_bola − A_base                    [zero-centred]",
                "BJ_norm = BJ_raw / σ_running                [normalised]",
            ],
            "tag": "IMP 1",
        },
        {
            "num": "05",
            "title": "Confidence Gate  (IMP 3)",
            "color": "#042f1a",
            "border": COLORS["imp3"],
            "content": [
                "Update MAE buffer: |VIX_pred_t+1 − VIX_actual|",
                "MAE_roll = mean(buffer, last 20 steps)",
                "α_eff   = α · sigmoid(−30 · MAE_roll)",
                "w       = α_eff · tanh(1.5 · BJ_norm)",
            ],
            "tag": "IMP 3",
        },
        {
            "num": "06",
            "title": "Blend Action",
            "color": "#1a0a2e",
            "border": COLORS["reward_light"],
            "content": [
                "a_blend = (1 − w) · a_base  +  w · a_rh",
                "a_blend = clip(a_blend, −1, +1)",
                "trade   = a_blend × MAX_TRADE_STEP  (=0.20)",
            ],
            "tag": "STEP 6",
        },
        {
            "num": "07",
            "title": "No-Trade Filter  (IMP 4)",
            "color": "#1e0845",
            "border": COLORS["imp4"],
            "content": [
                "regime = 0.8·(VIX_norm−1)⁺ + 4.0·VIX_mom⁺ + 3.0·spike",
                "band   = 0.010 × (1.35 − 0.70·regime − 0.25·conf)",
                "band   = clip(band, 0.005, 0.018)",
                "if |trade| < band:  aₜ = 0   →   hold, no trade",
            ],
            "tag": "IMP 4",
        },
        {
            "num": "08",
            "title": "Execute & Compute Costs",
            "color": "#0a1e2e",
            "border": "#0284c7",
            "content": [
                "Δhₜ  = aₜ × 0.20",
                "hₜ₊₁ = clip(hₜ + Δhₜ,  −1, +1)",
                "TC   = ζ·|Δh|·Sₜ  +  ζ²·|Δh|²·Sₜ     (ζ=0.001, ζ²=0.002)",
            ],
            "tag": "STEP 8",
        },
        {
            "num": "09",
            "title": "Compute Reward  rₜ",
            "color": "#1a0a2e",
            "border": COLORS["reward_accent"],
            "content": [
                "opt_pnl   = Cₜ − Cₜ₊₁",
                "hedge_pnl = hₜ · (Sₜ₊₁ − Sₜ)",
                "daily_pnl = opt_pnl + hedge_pnl − TC",
                "rₜ = pnl_norm − 0.25·pnl² − 0.50·(h−Δ_BS)² − 0.10·cost_norm",
            ],
            "tag": "STEP 9",
        },
        {
            "num": "10",
            "title": "Transition  s_t → s_{t+1}",
            "color": "#0d3b2e",
            "border": COLORS["env_accent"],
            "content": [
                "Market evolves: Sₜ₊₁, σₜ₊₁, Cₜ₊₁  (exogenous)",
                "Hedge updates: hₜ → hₜ₊₁",
                "Window rolls: drop oldest day, append current",
                "If terminal: unwind hedge, add final TC, end episode",
            ],
            "tag": "STEP 10",
        },
    ]

    step_h   = 1.85
    step_gap = 0.18
    start_y  = 24.4
    cx       = 9.0
    box_w    = 15.0
    box_x    = cx - box_w / 2

    for i, step in enumerate(steps):
        sy = start_y - i * (step_h + step_gap)

        # Main box
        rounded_box(ax, box_x, sy - step_h, box_w, step_h,
                    step["color"], edgecolor=step["border"],
                    linewidth=2.5, alpha=0.93, zorder=3)

        # Left number badge
        rounded_box(ax, box_x, sy - step_h, 1.2, step_h,
                    step["border"], edgecolor=step["border"],
                    linewidth=1, alpha=0.35, zorder=4)
        ax.text(box_x + 0.6, sy - step_h/2, step["num"],
                fontsize=11, color=step["border"],
                ha="center", va="center", fontweight="bold", zorder=6)

        # Title
        ax.text(box_x + 1.5, sy - 0.28, step["title"],
                fontsize=11, color=step["border"],
                ha="left", va="center", fontweight="bold", zorder=6)

        # Tag pill
        pill_text = step["tag"]
        ax.text(box_x + box_w - 0.2, sy - 0.28, pill_text,
                fontsize=8, color=step["border"],
                ha="right", va="center", style="italic", zorder=6)

        # Content lines
        for li, line in enumerate(step["content"]):
            lx = box_x + 1.55
            ly = sy - 0.68 - li * 0.28
            ax.plot(lx - 0.15, ly, "o", color=step["border"],
                    markersize=3.5, zorder=6)
            ax.text(lx, ly, line, fontsize=8,
                    color=COLORS["text_secondary"], ha="left", va="center", zorder=6)

        # Arrow to next step
        if i < len(steps) - 1:
            arrow_col = step["border"]
            ax.annotate("",
                xy=(cx, sy - step_h - step_gap),
                xytext=(cx, sy - step_h),
                arrowprops=dict(arrowstyle="-|>", color=arrow_col,
                                lw=2.5, mutation_scale=18), zorder=7)

    # ── Terminal branch ───────────────────────────────────────────────────────
    term_y = start_y - len(steps) * (step_h + step_gap) + 0.1

    rounded_box(ax, 1.0, term_y - 1.0, 6.0, 0.85, "#064e3b",
                edgecolor="#4ecca3", linewidth=2, alpha=0.9, zorder=4)
    ax.text(4.0, term_y - 0.57, "t < T−1  →  Continue  (return to STEP 01)",
            fontsize=9, color="#4ecca3", ha="center", va="center",
            fontweight="bold", zorder=6)

    rounded_box(ax, 10.5, term_y - 1.0, 6.5, 0.85, "#450a0a",
                edgecolor="#ef4444", linewidth=2, alpha=0.9, zorder=4)
    ax.text(13.75, term_y - 0.57, "t = T−1  →  Unwind hedge  +  End episode",
            fontsize=9, color="#ef4444", ha="center", va="center",
            fontweight="bold", zorder=6)

    ax.annotate("", xy=(4.0, term_y - 1.0), xytext=(cx - 1.5, term_y),
                arrowprops=dict(arrowstyle="-|>", color="#4ecca3",
                                lw=2, mutation_scale=14), zorder=7)
    ax.annotate("", xy=(13.75, term_y - 1.0), xytext=(cx + 1.5, term_y),
                arrowprops=dict(arrowstyle="-|>", color="#ef4444",
                                lw=2, mutation_scale=14), zorder=7)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/04_mdp_step_flow.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/04_mdp_step_flow.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5: MDP Formal Definition
# ═════════════════════════════════════════════════════════════════════════════

def diagram_5_mdp_formal():
    fig = plt.figure(figsize=(20, 13), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(10, 12.4, "Markov Decision Process  M = (S, A, P, R, γ)",
            fontsize=22, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(10, 12.0, "Formal definition of the BOLA delta-hedging MDP",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    components = [
        {
            "symbol": "S",
            "name": "State Space",
            "color": COLORS["obs_dark"],
            "border": COLORS["obs_light"],
            "bx": 0.3, "by": 8.3, "bw": 9.3, "bh": 3.3,
            "lines": [
                ("sₜ = [ w_{t-K:t} | cₜ | hₜ | vₜ ]", True),
                ("w_{t-K:t} : K=5 day window × 5 features = 25 dims", False),
                ("  [Spot/S₀, T, VIX_σ, Delta, Moneyness]", False),
                ("cₜ : current snapshot = 4 dims", False),
                ("  [VIX_σ, BS_Δ, T, Moneyness]", False),
                ("hₜ : hedge ratio ∈ [−1,+1] = 1 dim", False),
                ("vₜ : VIX extras (BOLA) = 5 dims", False),
                ("  [norm, mom, pred_{t+1}, pred_{t+2}, pred_{t+3}]", False),
                ("Total: 35 dims (BOLA) | 30 dims (Base)", True),
            ],
        },
        {
            "symbol": "A",
            "name": "Action Space",
            "color": "#1a1130",
            "border": COLORS["agent_accent"],
            "bx": 10.0, "by": 8.3, "bw": 9.7, "bh": 3.3,
            "lines": [
                ("aₜ ∈ [−1, +1]   (continuous, 1-dimensional)", True),
                ("Δhₜ = aₜ × MAX_TRADE_STEP  (0.20)", False),
                ("hₜ₊₁ = clip(hₜ + Δhₜ, −1, +1)", False),
                ("", False),
                ("No-Trade Filter (IMP 4):", True),
                ("  band = 0.010 × regime_factor ∈ [0.005, 0.018]", False),
                ("  if |Δhₜ| < band:   aₜ := 0   (hold position)", False),
                ("", False),
                ("Effective range: Δh ∈ [−0.20, +0.20] per step", False),
            ],
        },
        {
            "symbol": "P",
            "name": "Transition Function",
            "color": COLORS["env_dark"],
            "border": COLORS["env_accent"],
            "bx": 0.3, "by": 4.5, "bw": 9.3, "bh": 3.4,
            "lines": [
                ("P(sₜ₊₁ | sₜ, aₜ) — stochastic market", True),
                ("", False),
                ("Market (exogenous, uncontrolled):", False),
                ("  Sₜ₊₁ ~ real Nifty price process", False),
                ("  σₜ₊₁ ~ real India VIX dynamics", False),
                ("  Cₜ₊₁ = BS(Sₜ₊₁, K, Tₜ₊₁, σₜ₊₁)", False),
                ("", False),
                ("Agent-controlled:", True),
                ("  hₜ₊₁ = clip(hₜ + Δhₜ, −1, +1)", False),
                ("  Window rolls: drop oldest, append cₜ", False),
            ],
        },
        {
            "symbol": "R",
            "name": "Reward Function",
            "color": "#1a0a2e",
            "border": COLORS["reward_accent"],
            "bx": 10.0, "by": 4.5, "bw": 9.7, "bh": 3.4,
            "lines": [
                ("rₜ = R(sₜ, aₜ, sₜ₊₁)", True),
                ("", False),
                ("opt_pnl   = Cₜ − Cₜ₊₁", False),
                ("hedge_pnl = hₜ · (Sₜ₊₁ − Sₜ)", False),
                ("TC        = ζ|Δh|Sₜ + ζ²|Δh|²Sₜ", False),
                ("pnl_norm  = (opt_pnl + hedge_pnl − TC) / Sₜ", False),
                ("", False),
                ("rₜ = pnl_norm  −  RA·pnl_norm²", True),
                ("       −  λ_δ·(hₜ−Δ_BS)²  −  λ_c·cost_norm", True),
                ("RA=0.25  |  λ_δ=0.50  |  λ_c=0.10", False),
            ],
        },
        {
            "symbol": "γ",
            "name": "Discount & Objective",
            "color": "#0d1f0d",
            "border": "#86efac",
            "bx": 0.3, "by": 0.4, "bw": 9.3, "bh": 3.7,
            "lines": [
                ("γ = 0.99   (long-horizon hedging)", True),
                ("", False),
                ("Objective:  max E[ Σₜ γᵗ · rₜ ]", True),
                ("", False),
                ("  ≡  minimise tracking error (P&L std)", False),
                ("  while controlling transaction costs", False),
                ("  and staying near Black-Scholes delta", False),
                ("", False),
                ("Episode: 1 option contract life", False),
                ("  Length: 2 → ~60 trading days", False),
            ],
        },
        {
            "symbol": "π",
            "name": "Policy & Algorithm",
            "color": "#0f172a",
            "border": "#f59e0b",
            "bx": 10.0, "by": 0.4, "bw": 9.7, "bh": 3.7,
            "lines": [
                ("PPO  (Proximal Policy Optimisation)", True),
                ("  clip=0.20  |  GAE λ=0.95  |  ent=0.005", False),
                ("  n_steps=512  |  batch=64  |  lr=3e-4", False),
                ("", False),
                ("BOLA blending:", True),
                ("  π_BOLA = (1−w)·π_base + w·π_bola", False),
                ("  w = α_eff · tanh(BJ_gap_norm)", False),
                ("", False),
                ("Value fn V(s): shared MLP critic 128→128→1", False),
                ("  Used in TD-advantage BJ gap (IMP 1)", False),
            ],
        },
    ]

    for comp in components:
        bx, by = comp["bx"], comp["by"]
        bw, bh = comp["bw"], comp["bh"]
        col = comp["color"]
        border = comp["border"]

        rounded_box(ax, bx, by, bw, bh, col,
                    edgecolor=border, linewidth=2.5, alpha=0.93, zorder=2)

        # Symbol badge
        rounded_box(ax, bx, by + bh - 0.9, 0.9, 0.9, border,
                    edgecolor=border, linewidth=1, alpha=0.4, zorder=3)
        ax.text(bx + 0.45, by + bh - 0.45, comp["symbol"],
                fontsize=18, color=border, ha="center", va="center",
                fontweight="bold", zorder=6)

        ax.text(bx + 1.1, by + bh - 0.45, comp["name"],
                fontsize=12, color=border, ha="left", va="center",
                fontweight="bold", zorder=6)

        for li, (line, bold) in enumerate(comp["lines"]):
            if not line:
                continue
            ly = by + bh - 1.15 - li * 0.28
            if ly < by + 0.08:
                continue
            fw = "bold" if bold else "normal"
            fc = COLORS["text_primary"] if bold else COLORS["text_secondary"]
            ax.text(bx + 0.15, ly, line, fontsize=8.5, color=fc,
                    ha="left", va="center", fontweight=fw, zorder=6)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/05_mdp_formal.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/05_mdp_formal.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 6: VIX Forecaster Integration
# ═════════════════════════════════════════════════════════════════════════════

def diagram_6_vix_forecaster():
    fig = plt.figure(figsize=(20, 11), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(10, 10.45, "VIX Forecaster LSTM — Integration in BOLA MDP",
            fontsize=21, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(10, 10.05, "How the VIX prediction model enriches the state and enables planning",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    # ── Offline Training block ───────────────────────────────────────────────
    rounded_box(ax, 0.3, 5.5, 8.5, 4.1, "#0d1f2e",
                edgecolor="#0ea5e9", linewidth=2.5, zorder=2)
    ax.text(4.55, 9.3, "OFFLINE PRE-TRAINING  (2010 → 2023)",
            fontsize=13, fontweight="bold", color="#0ea5e9",
            ha="center", va="center", zorder=5)

    train_steps = [
        ("Historical VIX\n(past 20 days)", "#1e3a5f", "#3b82f6"),
        ("VIXForecastLSTM\nhidden=32", "#1e1b4b", "#6366f1"),
        ("[pred_t+1\npred_t+2\npred_t+3]", "#2d1b69", "#8b5cf6"),
        ("MSE Loss\nBackprop", "#1a0a2e", "#c084fc"),
    ]
    for i, (label, fc, ec) in enumerate(train_steps):
        tx = 0.7 + i * 2.0
        ty = 6.8
        rounded_box(ax, tx, ty, 1.7, 2.1, fc, edgecolor=ec,
                    linewidth=2, alpha=0.9, zorder=4)
        ax.text(tx + 0.85, ty + 1.05, label, fontsize=8.5, color=ec,
                ha="center", va="center", fontweight="bold",
                linespacing=1.4, zorder=6)
        if i < len(train_steps) - 1:
            ax.annotate("",
                xy=(tx + 1.72, ty + 1.05),
                xytext=(tx + 1.68, ty + 1.05),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8",
                                lw=2, mutation_scale=14), zorder=7)

    rounded_box(ax, 0.5, 5.7, 8.1, 0.95, "#0a1628",
                edgecolor="#0ea5e9", linewidth=1.5, alpha=0.9, zorder=4)
    ax.text(4.55, 6.17, "Saved to:  models/vix_forecaster_lstm.pt  |  200 epochs  |  lr=0.001",
            fontsize=8.5, color="#7dd3fc", ha="center", va="center", zorder=6)

    # ── Online Inference block ────────────────────────────────────────────────
    rounded_box(ax, 0.3, 0.4, 8.5, 4.7, "#0d200d",
                edgecolor="#4ecca3", linewidth=2.5, zorder=2)
    ax.text(4.55, 4.8, "ONLINE INFERENCE  (every step, 2024 → 2026)",
            fontsize=13, fontweight="bold", color="#4ecca3",
            ha="center", va="center", zorder=5)

    infer_steps = [
        ("VIX window\nlast 20 days", "#0d3b2e", "#4ecca3"),
        ("model.eval()\nforward pass", "#064e3b", "#6ee7b7"),
        ("3 VIX forecasts\nclipped [0.01, 2.0]", "#065f46", "#a7f3d0"),
        ("Added to\nstate  sₜ", "#047857", "#d1fae5"),
    ]
    for i, (label, fc, ec) in enumerate(infer_steps):
        tx = 0.7 + i * 2.0
        ty = 1.7
        rounded_box(ax, tx, ty, 1.7, 2.0, fc, edgecolor=ec,
                    linewidth=2, alpha=0.9, zorder=4)
        ax.text(tx + 0.85, ty + 1.0, label, fontsize=8.5, color=ec,
                ha="center", va="center", fontweight="bold",
                linespacing=1.4, zorder=6)
        if i < len(infer_steps) - 1:
            ax.annotate("",
                xy=(tx + 1.72, ty + 1.0),
                xytext=(tx + 1.68, ty + 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8",
                                lw=2, mutation_scale=14), zorder=7)

    rounded_box(ax, 0.5, 0.6, 8.1, 0.95, "#042f1a",
                edgecolor="#4ecca3", linewidth=1.5, alpha=0.9, zorder=4)
    ax.text(4.55, 1.07, "Test-period t+1 MAE tracked for IMP 3 confidence gating",
            fontsize=8.5, color="#4ecca3", ha="center", va="center", zorder=6)

    # Arrow from training to inference
    ax.annotate("",
        xy=(4.55, 5.5), xytext=(4.55, 5.7),
        arrowprops=dict(arrowstyle="-|>", color="#94a3b8",
                        lw=2.5, mutation_scale=16), zorder=7)
    ax.text(4.55, 5.1, "load model weights", fontsize=8,
            color="#94a3b8", ha="center", va="center", style="italic")

    # ── Usage panel ───────────────────────────────────────────────────────────
    rounded_box(ax, 9.2, 0.4, 10.5, 9.2, "#0f0f1a",
                edgecolor="#a855f7", linewidth=2.5, zorder=2)
    ax.text(14.45, 9.3, "VIX Forecast Usage in MDP",
            fontsize=13, fontweight="bold", color="#a855f7",
            ha="center", va="center", zorder=5)

    uses = [
        {
            "imp": "State  sₜ",
            "col": COLORS["obs_light"],
            "title": "Block 4 of Observation Vector",
            "desc": (
                "vₜ = [VIX_norm, VIX_mom,\n"
                "       VIX_pred_{t+1}, VIX_pred_{t+2}, VIX_pred_{t+3}]\n"
                "→ 5 extra dims injected into BOLA state sₜ\n"
                "→ Policy learns to react to predicted VIX stress"
            ),
            "by": 7.0,
        },
        {
            "imp": "IMP 2",
            "col": COLORS["imp2"],
            "title": "Synthetic Next-Obs for RH Planning",
            "desc": (
                "At planning step k:\n"
                "  current_VIX ← VIX_pred_{t+k}\n"
                "  obs_{k+1} = roll_window(obs_k, forecast)\n"
                "→ Enables 3-step lookahead without real market data"
            ),
            "by": 5.05,
        },
        {
            "imp": "IMP 3",
            "col": COLORS["imp3"],
            "title": "MAE Confidence Gating",
            "desc": (
                "Each step: err = |VIX_pred_{t+1} − VIX_actual|\n"
                "MAE_roll = mean(err, last 20 steps)\n"
                "α_eff = α · sigmoid(−30 · MAE_roll)\n"
                "→ Auto-shrinks BOLA influence when forecasts are bad"
            ),
            "by": 3.1,
        },
        {
            "imp": "IMP 4",
            "col": COLORS["imp4"],
            "title": "Forecast Spike → Narrow Band",
            "desc": (
                "forecast_spike = max(VIX_preds) − current_VIX\n"
                "regime += 3.0 · forecast_spike\n"
                "band = 0.010 × (1.35 − 0.70·regime − ...)\n"
                "→ Reacts faster when VIX spike predicted"
            ),
            "by": 1.1,
        },
    ]

    for u in uses:
        by = u["by"]
        rounded_box(ax, 9.5, by, 9.9, 1.7, COLORS["bg_card"],
                    edgecolor=u["col"], linewidth=2, alpha=0.9, zorder=4)
        rounded_box(ax, 9.5, by, 1.1, 1.7, u["col"],
                    edgecolor=u["col"], linewidth=1, alpha=0.3, zorder=4)
        ax.text(10.05, by + 0.85, u["imp"], fontsize=10, color=u["col"],
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(10.9, by + 1.37, u["title"], fontsize=9.5, color=u["col"],
                ha="left", va="center", fontweight="bold", zorder=6)
        ax.text(10.9, by + 0.72, u["desc"], fontsize=7.8,
                color=COLORS["text_secondary"], ha="left", va="center",
                linespacing=1.4, zorder=6)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/06_vix_forecaster.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/06_vix_forecaster.png")


# ═════════════════════════════════════════════════════════════════════════════
# DIAGRAM 7: Architecture Overview
# ═════════════════════════════════════════════════════════════════════════════

def diagram_7_architecture():
    fig = plt.figure(figsize=(22, 13), facecolor=COLORS["bg_main"])
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg_main"])

    ax.text(11, 12.4, "BOLA — Full Neural Architecture",
            fontsize=22, fontweight="bold", color=COLORS["text_primary"],
            ha="center", va="center")
    ax.text(11, 12.0, "LSTM Feature Extractor + PPO Actor-Critic + BOLA Adaptation Layer",
            fontsize=12, color=COLORS["text_secondary"], ha="center", va="center")

    # ── Input Blocks ──────────────────────────────────────────────────────────
    inputs = [
        ("K×5 Window\n(25 dims)", COLORS["obs_dark"],  COLORS["obs_light"],   1.0, 7.5, 3.2, 3.8),
        ("Current\n(4 dims)",     "#1c1917",            "#f97316",             4.5, 8.2, 2.2, 3.1),
        ("Hedge hₜ\n(1 dim)",     "#083344",            "#06b6d4",             7.0, 8.5, 1.8, 2.8),
        ("VIX Extras\n(5 dims)",  "#1a0a2e",            COLORS["bola_color"],  9.1, 7.5, 2.8, 3.8),
    ]
    for label, fc, ec, ix, iy, iw, ih in inputs:
        rounded_box(ax, ix, iy, iw, ih, fc, edgecolor=ec,
                    linewidth=2, alpha=0.9, zorder=3)
        ax.text(ix + iw/2, iy + ih/2, label, fontsize=10,
                color=ec, ha="center", va="center",
                fontweight="bold", linespacing=1.5, zorder=5)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    rounded_box(ax, 1.0, 4.5, 3.2, 2.6, COLORS["agent_dark"],
                edgecolor="#818cf8", linewidth=2.5, zorder=3)
    ax.text(2.6, 6.05, "LSTM", fontsize=13, color="#818cf8",
            ha="center", va="center", fontweight="bold", zorder=5)
    ax.text(2.6, 5.65, "input_size = 5\nhidden = 64\nbatch_first = True",
            fontsize=8, color=COLORS["text_secondary"],
            ha="center", va="center", linespacing=1.4, zorder=5)
    ax.text(2.6, 5.0, "→  h_t ∈ ℝ⁶⁴", fontsize=9, color="#c7d2fe",
            ha="center", va="center", fontweight="bold", zorder=5)

    # Arrow from window to LSTM
    ax.annotate("", xy=(2.6, 7.5), xytext=(2.6, 7.1),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["obs_light"],
                                lw=2.5, mutation_scale=16), zorder=7)

    # ── Concat ────────────────────────────────────────────────────────────────
    rounded_box(ax, 1.0, 2.8, 11.2, 1.4, "#1e293b",
                edgecolor="#64748b", linewidth=2, alpha=0.9, zorder=3)
    ax.text(6.6, 3.75, "CONCAT  →  128-dim Feature Vector",
            fontsize=12, color="#e2e8f0", ha="center", va="center",
            fontweight="bold", zorder=5)
    ax.text(6.6, 3.2,
            "[ h_LSTM(64) | current(4) | hedge(1) | VIX_extras(5) ]   =   74 dims  →  projected to 128",
            fontsize=8.5, color=COLORS["text_secondary"],
            ha="center", va="center", zorder=5)

    # Arrows into concat
    arrow_srcs = [(2.6, 4.5), (5.6, 8.2), (7.9, 8.5), (10.5, 7.5)]
    arrow_cols = [COLORS["obs_light"], "#f97316", "#06b6d4", COLORS["bola_color"]]
    for (ax_, ay_), col in zip(arrow_srcs, arrow_cols):
        ax.annotate("",
            xy=(ax_, 4.2), xytext=(ax_, ay_ - 0.02),
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=2, mutation_scale=14), zorder=7)

    # ── MLP Layers ────────────────────────────────────────────────────────────
    mlp_layers = [
        ("MLP Layer 1\n128 units + Tanh", "#1e1b4b", "#6366f1", 1.0, 1.1, 3.5),
        ("MLP Layer 2\n128 units + Tanh", "#1a1130", "#818cf8", 4.8, 1.1, 3.5),
    ]
    for label, fc, ec, lx, ly, lw in mlp_layers:
        rounded_box(ax, lx, ly, lw, 1.4, fc, edgecolor=ec,
                    linewidth=2, alpha=0.9, zorder=3)
        ax.text(lx + lw/2, ly + 0.7, label, fontsize=10, color=ec,
                ha="center", va="center", fontweight="bold",
                linespacing=1.5, zorder=5)

    ax.annotate("", xy=(1.75, 2.8), xytext=(1.75, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#6366f1",
                                lw=2.5, mutation_scale=15), zorder=7)
    ax.annotate("", xy=(4.8, 1.8), xytext=(4.5, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="#818cf8",
                                lw=2.5, mutation_scale=15), zorder=7)

    # ── Actor / Critic ────────────────────────────────────────────────────────
    heads = [
        ("ACTOR  π(a|s)", "Hedge action\naₜ ∈ [−1,+1]",    "#7f1d1d", COLORS["imp1"], 1.0),
        ("CRITIC  V(s)",  "State value\nV ∈ ℝ",             "#064e3b", "#10b981",      4.9),
    ]
    for title, desc, fc, ec, hx in heads:
        rounded_box(ax, hx, 0.2 - 0.05, 3.5, 0.82, fc, edgecolor=ec,
                    linewidth=2, alpha=0.9, zorder=4)
        ax.text(hx + 1.75, 0.62, title, fontsize=10, color=ec,
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(hx + 1.75, 0.28, desc, fontsize=8,
                color=COLORS["text_secondary"], ha="center", va="center",
                linespacing=1.3, zorder=6)

    ax.annotate("", xy=(2.75, 1.1), xytext=(2.75, 0.97),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["imp1"],
                                lw=2.5, mutation_scale=14), zorder=7)
    ax.annotate("", xy=(6.65, 1.1), xytext=(6.65, 0.97),
                arrowprops=dict(arrowstyle="-|>", color="#10b981",
                                lw=2.5, mutation_scale=14), zorder=7)

    # ── BOLA Adaptation ───────────────────────────────────────────────────────
    rounded_box(ax, 13.0, 1.0, 8.6, 10.2, "#0d0a1a",
                edgecolor=COLORS["bola_color"], linewidth=3, alpha=0.93, zorder=2)
    ax.text(17.3, 10.9, "BOLA ADAPTATION LAYER",
            fontsize=14, fontweight="bold", color=COLORS["bola_color"],
            ha="center", va="center", zorder=5)

    bola_blocks = [
        ("BJ Gap\n(IMP 1)", "A_bola − A_base\n(TD-advantage)", COLORS["imp1"], 13.3, 8.7, 3.8, 1.7),
        ("RH Plan\n(IMP 2)", "K=3 lookahead\nexec step 0",     COLORS["imp2"], 17.4, 8.7, 3.8, 1.7),
        ("MAE Gate\n(IMP 3)","α_eff = α·σ(−30·MAE)\nw = α_eff·tanh(BJ)",
         COLORS["imp3"], 13.3, 6.5, 7.9, 1.8),
        ("Band Filter\n(IMP 4)", "if |Δh| < band: hold\nband ∈ [0.005, 0.018]",
         COLORS["imp4"], 13.3, 4.3, 7.9, 1.8),
        ("Blend Action", "(1−w)·a_base + w·a_rh\n→ clip → filter → execute",
         "#64748b", 13.3, 2.3, 7.9, 1.6),
    ]
    for label, desc, ec, bx, by, bw, bh in bola_blocks:
        rounded_box(ax, bx, by, bw, bh, COLORS["bg_card"],
                    edgecolor=ec, linewidth=2, alpha=0.9, zorder=4)
        ax.text(bx + bw/2, by + bh - 0.38, label, fontsize=10, color=ec,
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(bx + bw/2, by + bh/2 - 0.18, desc, fontsize=8.5,
                color=COLORS["text_secondary"], ha="center", va="center",
                linespacing=1.4, zorder=6)

    bola_arrows = [
        (17.3, 8.7, 17.3, 8.4),
        (15.2, 8.7, 15.2, 8.3),
        (17.3, 6.5, 17.3, 6.35),
        (17.3, 4.3, 17.3, 3.9),
    ]
    for x1, y1, x2, y2 in bola_arrows:
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#64748b",
                            lw=2, mutation_scale=12), zorder=7)

    # Arrow from PPO outputs to BOLA layer
    ax.annotate("",
        xy=(13.0, 5.5), xytext=(9.0, 2.5),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["bola_color"],
                        lw=2.5, mutation_scale=16,
                        connectionstyle="arc3,rad=-0.25"), zorder=7)
    ax.text(11.5, 4.2, "a_base,\nV_base", fontsize=8.5,
            color=COLORS["bola_color"], ha="center", va="center",
            style="italic", zorder=8)

    plt.tight_layout(pad=0)
    plt.savefig("diagrams/07_architecture.png",
                dpi=180, bbox_inches="tight", facecolor=COLORS["bg_main"])
    plt.close()
    print("  Saved: diagrams/07_architecture.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BOLA MDP Diagram Generator")
    print("=" * 60)
    print("Generating diagrams...\n")

    diagram_1_agent_env_overview()
    diagram_2_state_space()
    diagram_3_bola_improvements()
    diagram_4_mdp_stepflow()
    diagram_5_mdp_formal()
    diagram_6_vix_forecaster()
    diagram_7_architecture()

    print("\n" + "=" * 60)
    print("All 7 diagrams saved to  ./diagrams/")
    print("=" * 60)
    print("""
  01_agent_env_overview.png  — Agent/Env interaction loop
  02_state_space.png         — 35-dim observation breakdown
  03_bola_improvements.png   — 4 IMPs in detail
  04_mdp_step_flow.png       — Per-step execution pipeline
  05_mdp_formal.png          — Formal MDP (S,A,P,R,γ,π)
  06_vix_forecaster.png      — VIX LSTM integration
  07_architecture.png        — Full neural architecture
""")