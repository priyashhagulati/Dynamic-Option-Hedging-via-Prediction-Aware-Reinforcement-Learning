"""
==============================================================================
BOLA Project — Full Pipeline (v3: Structural Fixes)
==============================================================================
Architecture:
  Stage 1: Offline Bayesian-style PPO actor-critic learns baseline policy/value
  Stage 2: VIX LSTM forecasts (t+1, t+2, t+3) are fed to BOLA PPO online

  Past K days of [Spot_norm, T, VIX_Sigma, Delta, Moneyness]   -> LSTM
  + Current step [VIX_sigma, BS_delta, T, moneyness]            -> concat
  + Current hedge                                               -> concat
  + VIX extras [norm, momentum, forecast t+1:t+3] (BOLA only)  -> concat
        ↓
    PPO Actor-Critic (MLP 128→128)
        ↓
      Hedge ratio ∈ [-1, 1]

Fixes vs v2:
  FIX 1: Agent now sees current-step state — removes information gap vs BS
  FIX 2: Delta-anchored reward: -(pnl² + λ_δ·(hedge-delta)² + λ_c·tc)
          Guides agent toward BS delta; lets it deviate only when beneficial
  FIX 3: BS agent in evaluate() uses current row delta correctly
  FIX 4: Evaluation starts from step 0 (zero-padded window) — fair vs BS
  FIX 5: 500k training steps
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_PATH  = "results/bs_features.csv"
VIX_PATH       = "data/india_vix.csv"
DATA_START     = pd.Timestamp("2010-07-19")
DATA_END       = pd.Timestamp("2026-04-26")
TRAIN_END      = pd.Timestamp("2023-12-31")
TEST_START     = pd.Timestamp("2024-01-01")
TRANS_COST     = 0.001
TRANS_COST_QUAD = 0.002   # quadratic market-impact style term
MAX_TRADE_STEP = 0.20     # action now means trade increment, not absolute hedge
LAMBDA_COST    = 0.1      # transaction cost penalty weight
LAMBDA_DELTA   = 0.5      # FIX 2: delta-anchoring strength
RISK_AVERSION  = 0.25     # mean-variance utility coefficient
GAMMA          = 0.99
K_WINDOW       = 5        # LSTM lookback (days)
LSTM_HIDDEN    = 64
BASE_FEATURES  = 5        # per-day in window: [spot_norm, T, vix_sigma, delta, moneyness]
CURR_FEATURES  = 4        # FIX 1: current-step extras: [vix_sigma, delta_bs, T, moneyness]
VIX_REALIZED_DIMS = 2     # BOLA-only: [VIX_norm, VIX_momentum]
VIX_FORECAST_DIMS = 3     # BOLA-only: predicted [VIX t+1, VIX t+2, VIX t+3]
VIX_EXTRA_DIMS = VIX_REALIZED_DIMS + VIX_FORECAST_DIMS
ALPHA          = 0.5  # BOLA max adaptation weight
BJ_NORM_FLOOR  = 1e-6
VIX_FEAR_LEVEL = 0.25
BOLA_NO_TRADE_BAND = 0.015  # base execution filter: skip tiny BOLA trades to reduce churn/cost
BOLA_BAND_MIN = 0.006       # adaptive band lower bound: react faster in stressed markets
BOLA_BAND_MAX = 0.022       # adaptive band upper bound: trade less in calm markets
USE_SAVED_MODELS = os.getenv("BOLA_USE_SAVED_MODELS", "1").lower() not in {"0", "false", "no"}
USE_ATTENTION_LSTM = os.getenv("BOLA_USE_ATTENTION", "0").lower() in {"1", "true", "yes"}
DEFAULT_TIMESTEPS = int(os.getenv("BOLA_TIMESTEPS", "500000"))
VIX_PREDICTOR_PATH = "models/vix_forecaster_lstm.pt"
VIX_PRED_WINDOW = 20
VIX_PRED_EPOCHS = int(os.getenv("BOLA_VIX_EPOCHS", "200"))
VIX_PRED_LR = float(os.getenv("BOLA_VIX_LR", "0.001"))

# Derived dimensions
# Vanilla obs: K*5 + CURR_FEATURES + 1(hedge)      = 30
# BOLA obs:    K*5 + CURR_FEATURES + 1 + VIX_EXTRA = 35
OBS_DIM_BASE = K_WINDOW * BASE_FEATURES + CURR_FEATURES + 1
OBS_DIM_BOLA = OBS_DIM_BASE + VIX_EXTRA_DIMS


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(features_path=FEATURES_PATH, vix_path=VIX_PATH):
    print("=" * 70)
    print("STAGE 0 — Data Loading")
    print("=" * 70)

    df = pd.read_csv(features_path, parse_dates=["DATE", "EXPIRY_DT"])
    df = df.sort_values(["DATE", "EXPIRY_DT", "STRIKE", "TYPE"]).reset_index(drop=True)
    df = df[df["DATE"].between(DATA_START, DATA_END)].reset_index(drop=True)
    df = df.dropna(subset=["DELTA", "VIX_SIGMA"]).reset_index(drop=True)

    if not os.path.exists(vix_path):
        raise FileNotFoundError(f"VIX file not found: '{vix_path}'")
    vix_raw   = pd.read_csv(vix_path, parse_dates=["DATE"]).set_index("DATE").sort_index()
    vix_raw   = vix_raw.loc[(vix_raw.index >= DATA_START) & (vix_raw.index <= DATA_END)]
    vix_sigma = vix_raw["VIX_CLOSE"] / 100.0
    vix_20ma  = vix_sigma.rolling(20, min_periods=1).mean()
    vix_norm  = (vix_sigma / vix_20ma.replace(0, 1e-6)).clip(0.5, 3.0)
    vix_mom   = vix_sigma.diff(5).fillna(0.0)

    vix_features = pd.DataFrame({
        "VIX_sigma":    vix_sigma,
        "VIX_norm":     vix_norm,
        "VIX_momentum": vix_mom,
    })

    print(f"  VIX: {vix_sigma.min()*100:.1f}% – {vix_sigma.max()*100:.1f}%  "
          f"({vix_sigma.index.min().date()} to {vix_sigma.index.max().date()})")

    df_train = df[df["DATE"] <= TRAIN_END]
    df_test  = df[df["DATE"] >= TEST_START]

    def build_episodes(data, label):
        eps = []
        for _, grp in data.groupby(["EXPIRY_DT", "STRIKE", "TYPE"]):
            grp = grp.sort_values("DATE").reset_index(drop=True)
            if len(grp) >= 2:   # FIX 4: accept short episodes (window is zero-padded)
                eps.append(grp)
        print(f"  [{label}]  Episodes: {len(eps):,}  |  "
              f"{data['DATE'].min().date()} -> {data['DATE'].max().date()}")
        return eps

    train_label = f"train ({df_train['DATE'].min().date()} to {TRAIN_END.date()})"
    test_label = f"test  ({TEST_START.date()} to {df_test['DATE'].max().date()})"
    train_eps = build_episodes(df_train, train_label)
    test_eps  = build_episodes(df_test,  test_label)
    split_info = {"train_end": TRAIN_END, "test_start": TEST_START}
    return train_eps, test_eps, vix_features, split_info


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A — VIX PREDICTION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class VIXForecastLSTM(nn.Module):
    """Forecast India VIX sigma for horizons t+1, t+2, and t+3."""

    def __init__(self, hidden_size=32, horizon=VIX_FORECAST_DIMS):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n.squeeze(0))


def _make_vix_forecast_dataset(vix_sigma, cutoff_date,
                               window=VIX_PRED_WINDOW, horizon=VIX_FORECAST_DIMS):
    values = vix_sigma.values.astype(np.float32)
    dates = vix_sigma.index
    xs, ys = [], []
    for end_idx in range(window, len(values) - horizon):
        if dates[end_idx + horizon - 1] > cutoff_date:
            break
        xs.append(values[end_idx - window:end_idx])
        ys.append(values[end_idx:end_idx + horizon])

    if not xs:
        raise ValueError("Not enough VIX history to train the forecast model.")

    x = torch.tensor(np.asarray(xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.asarray(ys), dtype=torch.float32)
    return x, y


def train_or_load_vix_forecaster(vix_features, split_info, model_path=VIX_PREDICTOR_PATH):
    """
    Train/load the Stage-2 VIX prediction model and append forecast columns:
    VIX_PRED_T1, VIX_PRED_T2, VIX_PRED_T3.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    vix_sigma = vix_features["VIX_sigma"].dropna().sort_index()
    model = VIXForecastLSTM()

    if USE_SAVED_MODELS and os.path.exists(model_path):
        print(f"\n  Loading VIX prediction model -> {model_path}")
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
    else:
        print(f"\n{'='*70}")
        print("STAGE 2A — VIX Prediction Model")
        print(f"{'='*70}")
        x_train, y_train = _make_vix_forecast_dataset(
            vix_sigma,
            cutoff_date=split_info["train_end"],
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=VIX_PRED_LR)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, VIX_PRED_EPOCHS + 1):
            optimiser.zero_grad()
            pred = model(x_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            optimiser.step()
            if epoch == 1 or epoch % 50 == 0 or epoch == VIX_PRED_EPOCHS:
                print(f"  epoch {epoch:>3}/{VIX_PRED_EPOCHS} | VIX forecast MSE: {loss.item():.8f}")

        torch.save({"model_state": model.state_dict()}, model_path)
        print(f"  Saved -> {model_path}")

    model.eval()
    enriched = vix_features.copy()
    values = vix_sigma.values.astype(np.float32)
    dates = vix_sigma.index
    preds = []

    with torch.no_grad():
        for idx, date in enumerate(dates):
            if idx < VIX_PRED_WINDOW:
                current = float(values[idx])
                pred_row = np.array([current] * VIX_FORECAST_DIMS, dtype=np.float32)
            else:
                window = torch.tensor(values[idx - VIX_PRED_WINDOW:idx],
                                      dtype=torch.float32).view(1, VIX_PRED_WINDOW, 1)
                pred_row = model(window).squeeze(0).numpy().astype(np.float32)
                pred_row = np.clip(pred_row, 0.01, 2.0)
            preds.append((date, pred_row))

    for horizon in range(VIX_FORECAST_DIMS):
        col = f"VIX_PRED_T{horizon + 1}"
        enriched[col] = np.nan
        for date, pred_row in preds:
            enriched.loc[date, col] = float(pred_row[horizon])

    pred_cols = [f"VIX_PRED_T{i + 1}" for i in range(VIX_FORECAST_DIMS)]
    test_mask = enriched.index >= split_info["test_start"]
    actual_next = enriched["VIX_sigma"].shift(-1)
    mae = (enriched.loc[test_mask, pred_cols[0]]
           - actual_next.loc[test_mask]).abs().mean()
    print(f"  VIX forecast columns added: {', '.join(pred_cols)}")
    if not np.isnan(mae):
        print(f"  Test-period t+1 forecast absolute error: {mae:.5f}")
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lookup_vix_features(vix_features, date):
    """Return BOLA VIX extras for a date, including the LSTM VIX forecast."""
    def safe_float(value, default):
        try:
            value = float(value)
            return default if np.isnan(value) else value
        except Exception:
            return default

    try:
        avail = vix_features.index[vix_features.index <= date]
        if len(avail):
            row = vix_features.loc[avail[-1]]
            sigma = float(np.clip(safe_float(row.get("VIX_sigma", 0.15), 0.15), 0.01, 2.0))
            forecasts = [
                float(np.clip(safe_float(row.get(f"VIX_PRED_T{i + 1}", sigma), sigma), 0.01, 2.0))
                for i in range(VIX_FORECAST_DIMS)
            ]
            return np.array([
                float(np.clip(row["VIX_norm"],     0.5, 3.0)),
                float(np.clip(row["VIX_momentum"], -0.5, 0.5)),
                *forecasts,
            ], dtype=np.float32)
    except Exception:
        pass
    return np.array([1.0, 0.0, 0.15, 0.15, 0.15], dtype=np.float32)


def _build_obs(ep, step_idx, hedge, S0, vix_features=None, use_vix=False):
    """
    Build the full observation vector.

    Structure:
      [K-step window (zero-padded)]   K*5 dims
      [current: vix_sigma, delta, T, moneyness]  4 dims  ← FIX 1
      [hedge]                          1 dim
      [VIX_norm, VIX_momentum, VIX_pred_t1:t3]  5 dims  (BOLA only)

    FIX 1: agent sees current-step state → same info as Black-Scholes.
    """
    # K-step window (rows step_idx-K to step_idx-1, zero-padded if needed)
    window = []
    for j in range(step_idx - K_WINDOW, step_idx):
        if 0 <= j < len(ep):
            r    = ep.loc[j]
            _vix = r["VIX_SIGMA"]
            window.extend([
                float(r["SPOT"]) / S0,
                float(r["T"]),
                float(np.clip(_vix, 0.0, 2.0)) if not np.isnan(_vix) else 0.0,
                float(r["DELTA"]),
                float(r["MONEYNESS"]),
            ])
        else:
            window.extend([0.0] * BASE_FEATURES)

    # Current step state (FIX 1)
    row  = ep.loc[step_idx]
    _cv  = row["VIX_SIGMA"]
    curr = [
        float(np.clip(_cv, 0.0, 2.0)) if not np.isnan(_cv) else 0.0,
        float(row["DELTA"]),
        float(row["T"]),
        float(row["MONEYNESS"]),
    ]

    obs = np.array(window + curr + [hedge], dtype=np.float32)

    # BOLA-only extras: [VIX_norm, VIX_momentum, predicted VIX sigma t+1:t+3]
    if use_vix and vix_features is not None:
        date = row["DATE"]
        vfe  = _lookup_vix_features(vix_features, date)
        obs  = np.append(obs, vfe).astype(np.float32)

    return obs


def _trade_from_policy(action, hedge):
    """Map policy output in [-1, 1] to a bounded hedge trade increment."""
    raw_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    trade      = raw_action * MAX_TRADE_STEP
    new_hedge  = float(np.clip(hedge + trade, -1.0, 1.0))
    return trade, new_hedge


def _target_hedge_trade(target_hedge, hedge):
    """Convert a target hedge level into the corresponding trade increment."""
    new_hedge = float(np.clip(float(np.asarray(target_hedge).reshape(-1)[0]), -1.0, 1.0))
    trade     = new_hedge - hedge
    return trade, new_hedge


def _transaction_cost(spot, trade):
    """Linear spread + quadratic market impact."""
    abs_trade = abs(float(trade))
    linear    = TRANS_COST * abs_trade * float(spot)
    impact    = TRANS_COST_QUAD * (abs_trade ** 2) * float(spot)
    return linear + impact


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class NiftyHedgingEnv(gym.Env):
    """
    FIX 1: Observation now includes current-step [vix_sigma, delta_bs, T, moneyness].
    FIX 2: Delta-anchored reward penalises deviation from BS delta.
    FIX 4: Episodes start from step 0 (window zero-padded).
    NEW: PPO action is now a bounded trade increment; hedge inventory is part of state.
    NEW: Transaction costs include linear spread and quadratic impact.
    NEW: Reward follows a mean-variance style utility and closes residual hedge at episode end.

    Obs structure (vanilla): [K*5 window | 4 current | 1 hedge] = 30
    Obs structure (BOLA):    [K*5 window | 4 current | 1 hedge | 5 VIX_extra] = 35
    """
    metadata = {"render_modes": []}

    def __init__(self, episodes, trans_cost=TRANS_COST, lambda_cost=LAMBDA_COST,
                 lambda_delta=LAMBDA_DELTA, use_vix=False, vix_features=None, seed=42):
        super().__init__()
        self.episodes      = episodes
        self.trans_cost    = trans_cost
        self.lambda_cost   = lambda_cost
        self.lambda_delta  = lambda_delta
        self.use_vix       = use_vix
        self.vix_features  = vix_features

        n_obs = OBS_DIM_BOLA if use_vix else OBS_DIM_BASE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_internals()

    def _reset_internals(self):
        self.ep          = None
        self.step_idx    = 0     # FIX 4: start from 0
        self.hedge       = 0.0
        self.S0          = 1.0
        self.episode_pnl = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx       = self.np_random.integers(0, len(self.episodes))
        self.ep   = self.episodes[idx].reset_index(drop=True)
        self.step_idx    = 0     # FIX 4
        self.hedge       = 0.0
        self.episode_pnl = []
        self.S0          = max(self.ep.loc[0, "SPOT"], 1.0)
        return _build_obs(self.ep, 0, 0.0, self.S0, self.vix_features, self.use_vix), {}

    def step(self, action):
        ep     = self.ep
        i      = self.step_idx
        row    = ep.loc[i]
        next_i = i + 1

        trade, new_hedge = _trade_from_policy(action, self.hedge)
        curr_spot  = row["SPOT"]
        curr_opt   = row["OPTION_PRICE"]
        bs_delta   = float(row["DELTA"])        # current BS delta
        trans_cost = _transaction_cost(curr_spot, trade)

        if next_i < len(ep):
            nrow      = ep.loc[next_i]
            opt_pnl   = curr_opt - nrow["OPTION_PRICE"]
            hedge_pnl = new_hedge * (nrow["SPOT"] - curr_spot)
            daily_pnl = opt_pnl + hedge_pnl - trans_cost

            terminated = (next_i >= len(ep) - 1)
            terminal_cost = 0.0
            if terminated:
                # Close any residual hedge at the end of the episode.
                terminal_cost = _transaction_cost(nrow["SPOT"], -new_hedge)
                daily_pnl    -= terminal_cost

            # Mean-variance style reward with delta anchor and nonlinear costs.
            delta_dev = (new_hedge - bs_delta) ** 2
            pnl_norm  = daily_pnl / (curr_spot + 1e-8)
            cost_norm = (trans_cost + terminal_cost) / (curr_spot + 1e-8)
            reward    = (pnl_norm
                         - RISK_AVERSION * (pnl_norm ** 2)
                         - self.lambda_delta * delta_dev
                         - self.lambda_cost  * cost_norm)

            self.step_idx    = next_i
            self.hedge       = 0.0 if terminated else new_hedge
            self.episode_pnl.append(daily_pnl)

            info = {"daily_pnl": daily_pnl, "trans_cost": trans_cost + terminal_cost,
                    "delta_bs": bs_delta,    "agent_hedge": new_hedge,
                    "trade": trade,          "terminal_cost": terminal_cost}
        else:
            reward, terminated = 0.0, True
            info = {}

        obs = _build_obs(ep, min(next_i, len(ep)-1), self.hedge,
                         self.S0, self.vix_features, self.use_vix)
        return obs, reward, terminated, False, info

    def get_episode_stats(self):
        pnl = np.array(self.episode_pnl)
        if not len(pnl):
            return {"tracking_error": 0.0, "total_pnl": 0.0, "n_steps": 0}
        return {"tracking_error": float(np.std(pnl)),
                "total_pnl":      float(np.sum(pnl)),
                "n_steps":        len(pnl)}

    def render(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# LSTM FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Obs layout:  [K*5 window | 4 curr | 1 hedge | (5 VIX)]
    LSTM processes the K*5 window → h_T (64 dims)
    Output: [h_T | 4 curr | 1 hedge | (5 VIX)]  = 69 or 74 dims
    """

    def __init__(self, observation_space: spaces.Box, use_vix: bool = False):
        self.use_vix   = use_vix
        self.window_dim = K_WINDOW * BASE_FEATURES  # 25
        # After LSTM: h_T(64) + curr(4) + hedge(1) + VIX_extra(5 if BOLA)
        features_dim = LSTM_HIDDEN + CURR_FEATURES + 1 + (VIX_EXTRA_DIMS if use_vix else 0)
        super().__init__(observation_space, features_dim=features_dim)

        self.lstm = nn.LSTM(
            input_size  = BASE_FEATURES,
            hidden_size = LSTM_HIDDEN,
            num_layers  = 1,
            batch_first = True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B           = obs.shape[0]
        window_flat = obs[:, :self.window_dim]                   # (B, 25)
        rest        = obs[:, self.window_dim:]                   # (B, 4+1[+2])

        window_seq  = window_flat.view(B, K_WINDOW, BASE_FEATURES)
        _, (h_n, _) = self.lstm(window_seq)
        h_last      = h_n.squeeze(0)                             # (B, 64)

        return torch.cat([h_last, rest], dim=1)


class AttentionLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Attention-LSTM extractor for retraining experiments.

    The LSTM emits one hidden vector per lookback day. A small attention head
    learns which days matter most, then the weighted context is concatenated
    with current state, hedge, and optional BOLA VIX extras.
    """

    def __init__(self, observation_space: spaces.Box, use_vix: bool = False):
        self.use_vix = use_vix
        self.window_dim = K_WINDOW * BASE_FEATURES
        features_dim = LSTM_HIDDEN + CURR_FEATURES + 1 + (VIX_EXTRA_DIMS if use_vix else 0)
        super().__init__(observation_space, features_dim=features_dim)

        self.lstm = nn.LSTM(
            input_size=BASE_FEATURES,
            hidden_size=LSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN // 2),
            nn.Tanh(),
            nn.Linear(LSTM_HIDDEN // 2, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        window_flat = obs[:, :self.window_dim]
        rest = obs[:, self.window_dim:]

        window_seq = window_flat.view(batch_size, K_WINDOW, BASE_FEATURES)
        lstm_out, _ = self.lstm(window_seq)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        return torch.cat([context, rest], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 / 2B — PPO TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class TrackingCallback(BaseCallback):
    def __init__(self, log_interval=50_000):
        super().__init__()
        self.ep_rewards   = []
        self.log_interval = log_interval
        self._last_log    = 0

    def _on_step(self):
        for done, info in zip(self.locals.get("dones", []),
                               self.locals.get("infos", [])):
            if done and "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if (self.num_timesteps - self._last_log) >= self.log_interval:
            if self.ep_rewards:
                print(f"    step {self.num_timesteps:>7,}  |  "
                      f"mean_reward (last 100 eps): "
                      f"{np.mean(self.ep_rewards[-100:]):.6f}")
            self._last_log = self.num_timesteps
        return True


def train_ppo_lstm(train_episodes, vix_features,
                   total_timesteps=500_000,
                   use_vix=False,
                   model_tag="vanilla_ppo_lstm",
                   use_attention=False):
    print(f"\n{'='*70}")
    stage_name = "STAGE 2B — Prediction Enhanced RL" if use_vix else "STAGE 1 — Offline Bayesian Learning"
    print(f"{stage_name}  [{model_tag}]")
    obs_dim = OBS_DIM_BOLA if use_vix else OBS_DIM_BASE
    extractor_cls = AttentionLSTMFeaturesExtractor if use_attention else LSTMFeaturesExtractor
    extractor_name = "Attention-LSTM" if use_attention else "LSTM"
    print(f"  use_vix={use_vix}  |  extractor={extractor_name}  |  obs_dim={obs_dim}  |  steps={total_timesteps:,}")
    print(f"{'='*70}")

    def make_env():
        return NiftyHedgingEnv(
            train_episodes, use_vix=use_vix, vix_features=vix_features)

    vec_env  = DummyVecEnv([make_env])
    callback = TrackingCallback()

    policy_kwargs = dict(
        features_extractor_class  = extractor_cls,
        features_extractor_kwargs = dict(use_vix=use_vix),
        net_arch                  = [128, 128],
        activation_fn             = nn.Tanh,
    )

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = 3e-4,
        n_steps       = 512,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = GAMMA,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.005,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = policy_kwargs,
        verbose       = 0,
    )

    vix_text = "+ realised/forecast VIX" if use_vix else ""
    print(f"  Architecture: {extractor_name}({BASE_FEATURES}→{LSTM_HIDDEN}) "
          f"→ concat(curr+hedge{vix_text}) → MLP(128→128) → Actor/Critic")
    print(f"  Action: bounded trade increment in [-{MAX_TRADE_STEP:.2f}, +{MAX_TRADE_STEP:.2f}] hedge units")
    print(f"  Training ...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(f"models/{model_tag}")
    print(f"  Saved → models/{model_tag}.zip")

    if callback.ep_rewards:
        print(f"  Episodes={len(callback.ep_rewards):,}  |  "
              f"Final mean reward: {np.mean(callback.ep_rewards[-100:]):.6f}")
    return model


def load_or_train_ppo(train_episodes, vix_features,
                      total_timesteps, use_vix, model_tag, use_attention):
    """
    Load a saved PPO model only if its observation space still matches the
    current environment contract; otherwise retrain a compatible checkpoint.
    """
    expected_obs_dim = OBS_DIM_BOLA if use_vix else OBS_DIM_BASE
    model_path = f"models/{model_tag}.zip"

    if USE_SAVED_MODELS and os.path.exists(model_path):
        label = "BOLA offline LSTM" if use_vix else "Vanilla PPO+LSTM"
        print(f"\n  Loading saved {label} -> {model_path}")
        model = PPO.load(f"models/{model_tag}", device="auto")
        loaded_obs_shape = getattr(model.observation_space, "shape", None)

        if loaded_obs_shape == (expected_obs_dim,):
            return model

        print(f"  Saved model incompatible: obs_shape={loaded_obs_shape}, "
              f"expected=({expected_obs_dim},)")
        print("  Retraining a fresh model to match the current observation layout...")

    return train_ppo_lstm(
        train_episodes, vix_features,
        total_timesteps=total_timesteps,
        use_vix=use_vix,
        model_tag=model_tag,
        use_attention=use_attention,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — BOLA ONLINE ADAPTATION
# ─────────────────────────────────────────────────────────────────────────────

class BOLAAgent:
    """
    BOLA with normalised BJ Gap, asymmetric weighting, and an adaptive execution
    no-trade band. It uses value with prediction minus value without prediction
    as the Bellman-Jensen gap, then reacts faster when the VIX forecast implies
    near-term regime pressure.
    """

    def __init__(self, model_base, model_bola, alpha=ALPHA, no_trade_band=BOLA_NO_TRADE_BAND):
        self.model_base = model_base
        self.model_bola = model_bola
        self.alpha      = alpha
        self.no_trade_band = no_trade_band
        self.bj_history = []
        self.weight_history = []
        self.band_history = []
        self.filtered_trades = 0
        self._run_mean  = 0.0
        self._run_var   = 1.0
        self._run_n     = 0

    def _update_stats(self, x):
        self._run_n += 1
        d = x - self._run_mean
        self._run_mean += d / self._run_n
        self._run_var  += d * (x - self._run_mean)

    def _norm(self, gap):
        if self._run_n < 2:
            return gap
        std = max(np.sqrt(self._run_var / self._run_n), BJ_NORM_FLOOR)
        return gap / std

    def get_action(self, obs_base, obs_vix):
        a_base, _ = self.model_base.predict(obs_base, deterministic=True)
        a_bola, _ = self.model_bola.predict(obs_vix,  deterministic=True)

        with torch.no_grad():
            v_base = self.model_base.policy.predict_values(
                torch.as_tensor(obs_base, dtype=torch.float32,
                                device=self.model_base.device).unsqueeze(0)).item()
            v_bola = self.model_bola.policy.predict_values(
                torch.as_tensor(obs_vix, dtype=torch.float32,
                                device=self.model_bola.device).unsqueeze(0)).item()

        bj_raw = v_bola - v_base
        self.bj_history.append(bj_raw)
        self._update_stats(bj_raw)

        bj_norm   = self._norm(bj_raw)
        bj_signed = bj_norm * 1.5 if bj_norm > 0 else bj_norm

        # Only move toward the BOLA policy when its value estimate is better.
        w      = self.alpha * max(0.0, float(np.tanh(bj_signed)))
        action = np.clip((1 - w) * a_base + w * a_bola, -1.0, 1.0)
        self.weight_history.append(w)

        if len(obs_vix) >= OBS_DIM_BOLA:
            vix_norm = float(obs_vix[-VIX_EXTRA_DIMS])
            vix_mom = float(obs_vix[-VIX_EXTRA_DIMS + 1])
            vix_preds = np.asarray(obs_vix[-VIX_FORECAST_DIMS:], dtype=np.float32)
            current_vix = float(obs_vix[K_WINDOW * BASE_FEATURES])
        else:
            vix_norm = 1.0
            vix_mom = 0.0
            vix_preds = np.array([0.15] * VIX_FORECAST_DIMS, dtype=np.float32)
            current_vix = 0.15
        forecast_spike = float(np.clip(np.max(vix_preds) - current_vix, 0.0, 0.5))
        regime_pressure = np.clip(
            0.8 * max(0.0, vix_norm - 1.0)
            + 4.0 * max(0.0, vix_mom)
            + 3.0 * forecast_spike,
            0.0,
            1.0,
        )
        confidence = np.clip(w / max(self.alpha, 1e-8), 0.0, 1.0)
        adaptive_band = self.no_trade_band * (1.35 - 0.70 * regime_pressure - 0.25 * confidence)
        adaptive_band = float(np.clip(adaptive_band, BOLA_BAND_MIN, BOLA_BAND_MAX))
        self.band_history.append(adaptive_band)

        trade = float(np.asarray(action).reshape(-1)[0]) * MAX_TRADE_STEP
        if abs(trade) < adaptive_band:
            action = np.array([0.0], dtype=np.float32)
            self.filtered_trades += 1

        return action, bj_raw, w


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent_fn, test_episodes, vix_features, name, action_mode="trade"):
    """
    FIX 3: BS agent accesses current delta from obs correctly.
    FIX 4: Evaluation starts from step 0 (window zero-padded) — fair vs BS.
    """
    rows = []
    for ep in test_episodes:
        ep = ep.reset_index(drop=True)
        if len(ep) < 2:
            continue

        hedge = 0.0
        S0    = max(ep.loc[0, "SPOT"], 1.0)

        # FIX 4: start from 0 not K_WINDOW
        for i in range(len(ep) - 1):
            row      = ep.loc[i]
            next_row = ep.loc[i + 1]
            date     = row["DATE"]

            obs_base = _build_obs(ep, i, hedge, S0, None,         False)
            obs_vix  = _build_obs(ep, i, hedge, S0, vix_features, True)
            vix_extra = _lookup_vix_features(vix_features, date)

            action = agent_fn(obs_base, obs_vix)
            if action_mode == "target":
                trade, new_hedge = _target_hedge_trade(action, hedge)
            else:
                trade, new_hedge = _trade_from_policy(action, hedge)
            curr_spot  = row["SPOT"]
            trans_cost = _transaction_cost(curr_spot, trade)

            opt_pnl   = row["OPTION_PRICE"] - next_row["OPTION_PRICE"]
            hedge_pnl = new_hedge * (next_row["SPOT"] - curr_spot)
            daily_pnl = opt_pnl + hedge_pnl - trans_cost
            terminal_cost = 0.0
            if i == len(ep) - 2:
                terminal_cost = _transaction_cost(next_row["SPOT"], -new_hedge)
                daily_pnl    -= terminal_cost

            rows.append({
                "DATE":        date,
                "AGENT":       name,
                "DAILY_PNL":   daily_pnl,
                "TRANS_COST":  trans_cost + terminal_cost,
                "AGENT_HEDGE": new_hedge,
                "TRADE":       trade,
                "DELTA_BS":    float(row["DELTA"]),
                "SPOT":        curr_spot,
                "VIX":         row["VIX"],
                "VIX_SIGMA":   row["VIX_SIGMA"],
                "VIX_NORM":    float(vix_extra[0]),
                "VIX_MOMENTUM": float(vix_extra[1]),
                "VIX_PRED_T1": float(vix_extra[2]),
                "VIX_PRED_T2": float(vix_extra[3]),
                "VIX_PRED_T3": float(vix_extra[4]),
                "T":           row["T"],
                "TYPE":        row.get("TYPE", ""),
            })
            hedge = 0.0 if i == len(ep) - 2 else new_hedge

    return pd.DataFrame(rows)


def bs_agent(obs_base, obs_vix):
    """
    FIX 3: Current-step delta is at position K*5 + 1 in obs_base
    (after window=K*5, then curr=[vix_sigma, delta_bs, T, moneyness]).
    """
    delta_idx = K_WINDOW * BASE_FEATURES + 1  # index of current delta_bs
    return np.array([obs_base[delta_idx]])


def metrics(df, name):
    pnl = df["DAILY_PNL"]
    te  = pnl.std()
    mu  = pnl.mean()
    sh  = (mu / te * np.sqrt(252)) if te > 0 else 0.0
    return {
        "Agent":            name,
        "Tracking_Error":   round(te, 4),
        "Sharpe_Ann":       round(sh, 4),
        "Mean_Daily_PNL":   round(mu, 4),
        "Total_PNL":        round(pnl.sum(), 2),
        "Total_Trans_Cost": round(df["TRANS_COST"].sum(), 2),
        "N_Steps":          len(df),
    }


def print_results(m_list, bola_agent=None):
    print(f"\n{'='*76}")
    print("STAGE 4 — BENCHMARKING  (Test: 2024-01-01 to 2026-04-26)")
    print(f"{'='*76}")
    print(f"{'Agent':<22} {'Track.Err':>10} {'Sharpe':>8} "
          f"{'Mean P&L':>10} {'Total P&L':>12} {'Trans.Cost':>12}")
    print("-" * 76)
    for m in m_list:
        print(f"{m['Agent']:<22} "
              f"{m['Tracking_Error']:>10.4f} "
              f"{m['Sharpe_Ann']:>8.4f} "
              f"{m['Mean_Daily_PNL']:>10.4f} "
              f"{m['Total_PNL']:>12.2f} "
              f"{m['Total_Trans_Cost']:>12.2f}")
    print("=" * 76)

    bs   = next(m for m in m_list if "Black" in m["Agent"])
    bola = next((m for m in m_list if "BOLA"  in m["Agent"]), None)
    ppo  = next((m for m in m_list if m["Agent"] == "Vanilla PPO+LSTM"), None)

    if bola:
        print(f"\n  BOLA vs BS  — TE: {(bs['Tracking_Error']-bola['Tracking_Error'])/bs['Tracking_Error']*100:+.1f}%"
              f"  |  Sharpe: {bola['Sharpe_Ann']-bs['Sharpe_Ann']:+.4f}")
    if ppo and bola:
        print(f"  BOLA vs PPO — TE: {(ppo['Tracking_Error']-bola['Tracking_Error'])/ppo['Tracking_Error']*100:+.1f}%"
              f"  |  Sharpe: {bola['Sharpe_Ann']-ppo['Sharpe_Ann']:+.4f}")

    if bola_agent and bola_agent.bj_history:
        bj = np.array(bola_agent.bj_history)
        ws = [bola_agent.alpha * max(0.0, np.tanh(bola_agent._norm(g) * 1.5 if bola_agent._norm(g) > 0 else bola_agent._norm(g))) for g in bj]
        print(f"\n  BJ Gap: mean={bj.mean():.5f}  std={bj.std():.5f}  "
              f"max={bj.max():.5f}")
        print(f"  Adapt weight w: mean={np.mean(ws):.4f}  max={np.max(ws):.4f}")
        if hasattr(bola_agent, "filtered_trades"):
            band = np.array(getattr(bola_agent, "band_history", []), dtype=float)
            band_text = ""
            if len(band):
                band_text = f" | adaptive band mean={band.mean():.4f} min={band.min():.4f} max={band.max():.4f}"
            print(f"  BOLA execution filter: skipped {bola_agent.filtered_trades:,} tiny trades "
                  f"(base band {bola_agent.no_trade_band:.3f} hedge units){band_text}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Stage 0
    train_eps, test_eps, vix_features, split_info = load_and_split()
    vix_features = train_or_load_vix_forecaster(vix_features, split_info)

    # Environment verification
    print(f"\n{'='*70}")
    print("Environment & LSTM Extractor Verification")
    print(f"{'='*70}")
    env    = NiftyHedgingEnv(train_eps, use_vix=False)
    obs, _ = env.reset()
    print(f"  Obs (vanilla): {obs.shape[0]}  "
          f"= K*{BASE_FEATURES}({K_WINDOW*BASE_FEATURES}) + curr({CURR_FEATURES}) + hedge(1)")
    env_v     = NiftyHedgingEnv(train_eps, use_vix=True, vix_features=vix_features)
    obs_v,  _ = env_v.reset()
    print(f"  Obs (BOLA)   : {obs_v.shape[0]}  "
          f"= {OBS_DIM_BASE} + VIX_extra({VIX_EXTRA_DIMS}: norm, momentum, pred t+1:t+3)")
    extractor_cls = AttentionLSTMFeaturesExtractor if USE_ATTENTION_LSTM else LSTMFeaturesExtractor
    ext = extractor_cls(env.observation_space, use_vix=False)
    out = ext(torch.FloatTensor(obs).unsqueeze(0))
    extractor_label = "Attention-LSTM" if USE_ATTENTION_LSTM else "LSTM"
    print(f"  {extractor_label} extractor output: {out.shape[1]} dims  ✓")
    print(f"  FIX 1 verified: current delta is obs[{K_WINDOW*BASE_FEATURES+1}]  "
          f"= {obs[K_WINDOW*BASE_FEATURES+1]:.4f} (should match row DELTA)")

    # Training — Adjust TIMESTEPS for time budget:
    #   200_000  → ~4 min   (quick test)
    #   500_000  → ~10 min  (recommended)
    TIMESTEPS = DEFAULT_TIMESTEPS
    vanilla_tag = "vanilla_ppo_attention_lstm" if USE_ATTENTION_LSTM else "vanilla_ppo_lstm"
    bola_tag = "bola_attention_lstm" if USE_ATTENTION_LSTM else "bola_offline_lstm"

    model_vanilla = load_or_train_ppo(
        train_eps, vix_features,
        total_timesteps=TIMESTEPS, use_vix=False, model_tag=vanilla_tag,
        use_attention=USE_ATTENTION_LSTM)

    model_bola_offline = load_or_train_ppo(
        train_eps, vix_features,
        total_timesteps=TIMESTEPS, use_vix=True, model_tag=bola_tag,
        use_attention=USE_ATTENTION_LSTM)

    # Stage 3
    print(f"\n{'='*70}")
    print("STAGE 3 — BOLA Online Adaptation Initialised")
    print(f"{'='*70}")
    bola = BOLAAgent(model_vanilla, model_bola_offline, alpha=ALPHA)
    print(f"  α={ALPHA}  |  BJ Gap = value(with VIX prediction) - value(without prediction)")

    # Stage 4
    print(f"\n{'='*70}")
    print("STAGE 4 — Evaluation (2024-01-01 to 2026-04-26) ...")
    print(f"{'='*70}")

    print("  Black-Scholes ...")
    df_bs = evaluate(bs_agent, test_eps, vix_features, "Black-Scholes", action_mode="target")

    print("  Vanilla PPO+LSTM ...")
    df_ppo = evaluate(
        lambda ob, ov: model_vanilla.predict(ob, deterministic=True)[0],
        test_eps, vix_features, "Vanilla PPO+LSTM", action_mode="trade")

    print("  BOLA+LSTM ...")
    df_bola = evaluate(
        lambda ob, ov: bola.get_action(ob, ov)[0],
        test_eps, vix_features, "BOLA+LSTM", action_mode="trade")

    m_list = [
        metrics(df_bs,   "Black-Scholes"),
        metrics(df_ppo,  "Vanilla PPO+LSTM"),
        metrics(df_bola, "BOLA+LSTM"),
    ]
    print_results(m_list, bola_agent=bola)

    # Hedge deviation analysis
    print(f"\n  Hedge deviation from BS delta (lower = better aligned):")
    for name, d in [("Black-Scholes", df_bs), ("Vanilla PPO", df_ppo), ("BOLA", df_bola)]:
        dev = (d["AGENT_HEDGE"] - d["DELTA_BS"]).abs().mean()
        print(f"    {name:<22}  mean |hedge - delta| = {dev:.4f}")

    df_all = pd.concat([df_bs, df_ppo, df_bola], ignore_index=True)
    df_all.to_csv("results/bola_lstm_comparison.csv",  index=False)
    pd.DataFrame(m_list).to_csv("results/bola_lstm_summary.csv", index=False)
    print(f"\n  Saved -> results/bola_lstm_comparison.csv")
    print(f"\n✅  BOLA + LSTM Pipeline Complete.")
