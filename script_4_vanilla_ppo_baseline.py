"""Standalone vanilla PPO+LSTM baseline against Black-Scholes."""

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

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_PATH = "results/bs_features.csv"
TRANS_COST    = 0.001        # 0.1% one-way
LAMBDA_COST   = 0.1          # transaction cost penalty weight
GAMMA         = 0.99
K_WINDOW      = 5            # LSTM lookback (days)
LSTM_HIDDEN   = 64
# BASE_FEATURES per day: [spot_norm, T, VIX_SIGMA, delta_bs, moneyness]
BASE_FEATURES = 5
DATA_START    = pd.Timestamp("2010-07-19")
DATA_END      = pd.Timestamp("2026-04-26")
TRAIN_END     = pd.Timestamp("2023-12-31")
TEST_START    = pd.Timestamp("2024-01-01")
DEFAULT_TIMESTEPS = int(os.getenv("VANILLA_PPO_TIMESTEPS", "200000"))
OUTPUT_LEVEL = os.getenv("VANILLA_PPO_OUTPUT", "summary").lower()
VERBOSE_OUTPUT = OUTPUT_LEVEL in {"verbose", "debug"}
QUIET_OUTPUT = OUTPUT_LEVEL in {"quiet", "silent"}

# Observation dim = K*BASE_FEATURES + 1 (hedge) = 26
OBS_DIM = K_WINDOW * BASE_FEATURES + 1


def log(message="", *, verbose=False):
    """Compact default output; verbose mode keeps training/debug details."""
    if QUIET_OUTPUT:
        return
    if verbose and not VERBOSE_OUTPUT:
        return
    print(message)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD & SPLIT DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    log("Vanilla PPO baseline")

    df = pd.read_csv(FEATURES_PATH, parse_dates=["DATE", "EXPIRY_DT"])
    df = df.sort_values(
        ["DATE", "EXPIRY_DT", "STRIKE", "TYPE"]).reset_index(drop=True)
    df = df[df["DATE"].between(DATA_START, DATA_END)].reset_index(drop=True)

    # Drop rows without valid VIX/DELTA inside the requested date window.
    df = df.dropna(subset=["DELTA", "VIX_SIGMA"]).reset_index(drop=True)

    df_train = df[df["DATE"] <= TRAIN_END]
    df_test  = df[df["DATE"] >= TEST_START]

    def make_episodes(data, label):
        eps = []
        for _, grp in data.groupby(["EXPIRY_DT", "STRIKE", "TYPE"]):
            grp = grp.sort_values("DATE").reset_index(drop=True)
            if len(grp) >= K_WINDOW + 1:
                eps.append(grp)
        log(f"  {label}: {len(eps):,} episodes "
            f"({data['DATE'].min().date()} to {data['DATE'].max().date()})")
        return eps

    train_label = f"train ({df_train['DATE'].min().date()} to {TRAIN_END.date()})"
    test_label = f"test  ({TEST_START.date()} to {df_test['DATE'].max().date()})"
    train_eps = make_episodes(df_train, train_label)
    test_eps  = make_episodes(df_test,  test_label)
    return train_eps, test_eps


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GYMNASIUM ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class NiftyHedgingEnv(gym.Env):
    """
    Vanilla PPO environment — NO VIX in observation.

    Observation (26 dims):
      Flat K-day window: K × [spot_norm, T, vix_sigma, delta_bs, moneyness] = 25
      Current hedge position: 1
      Total: 26

    Action:
      Continuous hedge ratio ∈ [-1, 1]

    Reward:
      -(daily_pnl²) - λ × trans_cost    (normalised by spot²)
      Agent learns to minimise P&L variance = minimise tracking error
    """

    metadata = {"render_modes": []}

    def __init__(self, episodes, trans_cost=TRANS_COST,
                 lambda_cost=LAMBDA_COST, seed=42):
        super().__init__()
        self.episodes     = episodes
        self.trans_cost   = trans_cost
        self.lambda_cost  = lambda_cost

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

    def _reset_state(self):
        self.ep          = None
        self.step_idx    = K_WINDOW
        self.hedge       = 0.0
        self.S0          = 1.0
        self.episode_pnl = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx      = self.np_random.integers(0, len(self.episodes))
        self.ep  = self.episodes[idx].reset_index(drop=True)
        self.step_idx    = K_WINDOW
        self.hedge       = 0.0
        self.episode_pnl = []
        self.S0          = max(self.ep.loc[0, "SPOT"], 1.0)
        return self._obs(), {}

    def step(self, action):
        i        = self.step_idx
        row      = self.ep.loc[i]
        next_i   = i + 1

        new_hedge  = float(np.clip(action[0], -1.0, 1.0))
        d_hedge    = new_hedge - self.hedge
        spot       = row["SPOT"]
        trans_cost = abs(d_hedge) * spot * self.trans_cost

        if next_i < len(self.ep):
            nr        = self.ep.loc[next_i]
            opt_pnl   = row["OPTION_PRICE"] - nr["OPTION_PRICE"]
            hedge_pnl = new_hedge * (nr["SPOT"] - spot)
            daily_pnl = opt_pnl + hedge_pnl - trans_cost

            reward    = -(daily_pnl**2) - self.lambda_cost * trans_cost
            reward    = reward / (spot**2 + 1e-8)

            terminated        = (next_i == len(self.ep) - 1)
            self.step_idx     = next_i
            self.hedge        = new_hedge
            self.episode_pnl.append(daily_pnl)
            info = {
                "daily_pnl":   daily_pnl,
                "trans_cost":  trans_cost,
                "delta_bs":    row["DELTA"],
                "agent_hedge": new_hedge,
            }
        else:
            reward, terminated = 0.0, True
            info = {}

        return self._obs(), reward, terminated, False, info

    def _obs(self):
        window = []
        for j in range(self.step_idx - K_WINDOW, self.step_idx):
            if 0 <= j < len(self.ep):
                r    = self.ep.loc[j]
                _vix = r["VIX_SIGMA"]
                window.extend([
                    float(r["SPOT"]) / self.S0,
                    float(r["T"]),
                    float(np.clip(_vix, 0.0, 2.0)) if not np.isnan(_vix) else 0.0,
                    float(r["DELTA"]),
                    float(r["MONEYNESS"]),
                ])
            else:
                window.extend([0.0] * BASE_FEATURES)
        return np.array(window + [self.hedge], dtype=np.float32)

    def get_stats(self):
        pnl = np.array(self.episode_pnl)
        if not len(pnl):
            return {"tracking_error": 0.0, "total_pnl": 0.0}
        return {
            "tracking_error": float(np.std(pnl)),
            "total_pnl":      float(np.sum(pnl)),
        }

    def render(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LSTM FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class LSTMExtractor(BaseFeaturesExtractor):
    """
    Reshapes flat 26-dim observation into:
      LSTM input : (batch, K=5, features=5)  → last hidden h_T (64 dims)
      Concat     : [h_T, current_hedge]       → 65 dims output

    This 65-dim vector feeds into PPO's actor and critic MLPs.
    """

    def __init__(self, observation_space: spaces.Box):
        out_dim = LSTM_HIDDEN + 1   # 64 + 1 (hedge)
        super().__init__(observation_space, features_dim=out_dim)
        self.lstm = nn.LSTM(
            input_size  = BASE_FEATURES,
            hidden_size = LSTM_HIDDEN,
            num_layers  = 1,
            batch_first = True,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B           = obs.shape[0]
        window_flat = obs[:, :K_WINDOW * BASE_FEATURES]          # (B, 25)
        hedge       = obs[:, K_WINDOW * BASE_FEATURES:]           # (B, 1)
        seq         = window_flat.view(B, K_WINDOW, BASE_FEATURES) # (B,5,5)
        _, (h_n, _) = self.lstm(seq)
        h_last      = h_n.squeeze(0)                              # (B, 64)
        return torch.cat([h_last, hedge], dim=1)                  # (B, 65)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN VANILLA PPO + LSTM
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeCallback(BaseCallback):
    """Track episode rewards and optionally log progress."""
    def __init__(self, log_interval=10_000):
        super().__init__()
        self.log_interval = log_interval
        self.ep_rewards   = []
        self._last_log    = 0

    def _on_step(self):
        for done, info in zip(
                self.locals.get("dones", []),
                self.locals.get("infos", [])):
            if done and "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        if VERBOSE_OUTPUT and (self.num_timesteps - self._last_log) >= self.log_interval:
            if self.ep_rewards:
                recent = np.mean(self.ep_rewards[-100:])
                log(f"    step {self.num_timesteps:>7,} | "
                    f"mean_reward(last 100 eps): {recent:.6f}", verbose=True)
            self._last_log = self.num_timesteps
        return True


def train_vanilla_ppo(train_episodes, total_timesteps=200_000):
    log("\nTraining vanilla PPO+LSTM")
    log(f"  Timesteps: {total_timesteps:,}")
    log(f"  Architecture: LSTM({BASE_FEATURES}->{LSTM_HIDDEN}) "
        f"-> MLP(128->128) -> Actor/Critic", verbose=True)

    vec_env  = DummyVecEnv([lambda: NiftyHedgingEnv(train_episodes)])
    callback = EpisodeCallback(log_interval=10_000)

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
        ent_coef      = 0.01,
        vf_coef       = 0.5,
        max_grad_norm = 0.5,
        policy_kwargs = dict(
            features_extractor_class  = LSTMExtractor,
            features_extractor_kwargs = {},
            net_arch                  = [128, 128],
            activation_fn             = nn.Tanh,
        ),
        verbose       = 0,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("models/vanilla_ppo_lstm")
    log("  Saved: models/vanilla_ppo_lstm.zip")
    if callback.ep_rewards:
        log(f"  Episodes={len(callback.ep_rewards):,}; "
            f"final mean reward={np.mean(callback.ep_rewards[-100:]):.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE ON TEST SET (2024-01-01 to 2026-04-26)
# ─────────────────────────────────────────────────────────────────────────────

def build_obs(ep, step_idx, hedge, S0):
    """Build flat 26-dim observation from episode DataFrame."""
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
    return np.array(window + [hedge], dtype=np.float32)


def evaluate_ppo(model, test_episodes):
    """Run trained PPO on test set, return per-step DataFrame."""
    log("  Evaluating Vanilla PPO+LSTM", verbose=True)
    rows = []
    for ep in test_episodes:
        ep    = ep.reset_index(drop=True)
        if len(ep) < K_WINDOW + 1:
            continue
        hedge = 0.0
        S0    = max(ep.loc[0, "SPOT"], 1.0)

        for i in range(K_WINDOW, len(ep) - 1):
            row      = ep.loc[i]
            next_row = ep.loc[i + 1]
            obs      = build_obs(ep, i, hedge, S0)

            action, _ = model.predict(obs, deterministic=True)
            new_hedge = float(np.clip(action[0], -1.0, 1.0))
            d_hedge   = new_hedge - hedge
            spot      = row["SPOT"]
            tc        = abs(d_hedge) * spot * TRANS_COST

            opt_pnl   = row["OPTION_PRICE"] - next_row["OPTION_PRICE"]
            hdg_pnl   = new_hedge * (next_row["SPOT"] - spot)
            daily_pnl = opt_pnl + hdg_pnl - tc

            rows.append({
                "DATE":        row["DATE"],
                "AGENT":       "Vanilla PPO+LSTM",
                "DAILY_PNL":   daily_pnl,
                "OPTION_PNL":  opt_pnl,
                "HEDGE_PNL":   hdg_pnl,
                "TRANS_COST":  tc,
                "AGENT_HEDGE": new_hedge,
                "DELTA_BS":    row["DELTA"],
                "SPOT":        spot,
                "VIX":         row["VIX"],
                "VIX_SIGMA":   row["VIX_SIGMA"],
                "T":           row["T"],
                "TYPE":        row.get("TYPE", ""),
            })
            hedge = new_hedge

    log(f"    PPO simulation rows: {len(rows):,}", verbose=True)
    return pd.DataFrame(rows)


def evaluate_black_scholes(test_episodes):
    """
    Black-Scholes baseline on test set.
    Hedge ratio = BS delta at each step (no learning).
    """
    log("  Evaluating Black-Scholes", verbose=True)
    rows = []
    for ep in test_episodes:
        ep    = ep.reset_index(drop=True)
        if len(ep) < 2:
            continue
        hedge = 0.0

        for i in range(len(ep) - 1):
            row      = ep.loc[i]
            next_row = ep.loc[i + 1]
            spot     = row["SPOT"]

            # BS hedge = delta
            new_hedge = float(np.clip(row["DELTA"], -1.0, 1.0))
            d_hedge   = new_hedge - hedge
            tc        = abs(d_hedge) * spot * TRANS_COST

            opt_pnl   = row["OPTION_PRICE"] - next_row["OPTION_PRICE"]
            hdg_pnl   = new_hedge * (next_row["SPOT"] - spot)
            daily_pnl = opt_pnl + hdg_pnl - tc

            rows.append({
                "DATE":        row["DATE"],
                "AGENT":       "Black-Scholes",
                "DAILY_PNL":   daily_pnl,
                "OPTION_PNL":  opt_pnl,
                "HEDGE_PNL":   hdg_pnl,
                "TRANS_COST":  tc,
                "AGENT_HEDGE": new_hedge,
                "DELTA_BS":    row["DELTA"],
                "SPOT":        spot,
                "VIX":         row["VIX"],
                "VIX_SIGMA":   row["VIX_SIGMA"],
                "T":           row["T"],
                "TYPE":        row.get("TYPE", ""),
            })
            hedge = new_hedge

    log(f"    BS simulation rows: {len(rows):,}", verbose=True)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — METRICS & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df, name):
    pnl  = df["DAILY_PNL"]
    te   = pnl.std()
    mu   = pnl.mean()
    sh   = (mu / te * np.sqrt(252)) if te > 0 else 0.0
    tc   = df["TRANS_COST"].sum()
    return {
        "Agent":             name,
        "Total_PNL":         round(pnl.sum(), 2),
        "Mean_Daily_PNL":    round(mu, 4),
        "Tracking_Error":    round(te, 4),
        "Sharpe_Ratio_Ann":  round(sh, 4),
        "Total_Trans_Cost":  round(tc, 2),
        "N_Steps":           len(df),
    }


def print_report(metrics_list, df_bs, df_ppo):
    log("\nBaseline comparison (test: 2024-01-01 to 2026-04-26)")

    # Main table
    log(f"{'Metric':<25} {'Black-Scholes':>16} {'Vanilla PPO+LSTM':>18}")
    log("-" * 62)
    keys = [
        ("Tracking Error (σ)",  "Tracking_Error"),
        ("Sharpe Ratio (ann.)", "Sharpe_Ratio_Ann"),
        ("Mean Daily P&L",      "Mean_Daily_PNL"),
        ("Total P&L",           "Total_PNL"),
        ("Total Trans. Cost",   "Total_Trans_Cost"),
        ("Steps evaluated",     "N_Steps"),
    ]
    for label, key in keys:
        bs_val  = metrics_list[0][key]
        ppo_val = metrics_list[1][key]
        log(f"{label:<25} {str(bs_val):>16} {str(ppo_val):>18}")

    # PPO improvement over BS
    bs_te  = metrics_list[0]["Tracking_Error"]
    ppo_te = metrics_list[1]["Tracking_Error"]
    if bs_te > 0:
        delta = (bs_te - ppo_te) / bs_te * 100
        sign  = "↓ better" if delta > 0 else "↑ worse"
        log(f"\nTracking error change (PPO vs BS): {delta:+.1f}% {sign}")

    bs_sh  = metrics_list[0]["Sharpe_Ratio_Ann"]
    ppo_sh = metrics_list[1]["Sharpe_Ratio_Ann"]
    log(f"Sharpe change (PPO vs BS): {ppo_sh - bs_sh:+.4f}")

    # Per option type breakdown
    log(f"\n{'Type':<8} {'BS Track.Err':>14} {'PPO Track.Err':>14} "
        f"{'BS Mean P&L':>13} {'PPO Mean P&L':>13}", verbose=True)
    log("-" * 65, verbose=True)
    for ot in ["CE", "PE"]:
        bs_t  = df_bs[df_bs["TYPE"]  == ot]["DAILY_PNL"]
        ppo_t = df_ppo[df_ppo["TYPE"] == ot]["DAILY_PNL"]
        if len(bs_t) and len(ppo_t):
            log(f"{ot:<8} "
                f"{bs_t.std():>14.4f} "
                f"{ppo_t.std():>14.4f} "
                f"{bs_t.mean():>13.4f} "
                f"{ppo_t.mean():>13.4f}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 0: load data
    train_eps, test_eps = load_data()

    # Step 1+2: environment + LSTM verified (no separate call needed —
    #           PPO instantiation handles this internally)
    log("\nEnvironment verification", verbose=True)
    log(f"  Observation dim: {OBS_DIM} = K({K_WINDOW}) * features({BASE_FEATURES}) + hedge(1)", verbose=True)
    log(f"  LSTM hidden: {LSTM_HIDDEN}; extractor output: {LSTM_HIDDEN + 1}", verbose=True)
    log("  Action dim: 1 (continuous hedge ratio in [-1, 1])", verbose=True)

    # Step 3: train PPO
    # Adjust for your time budget:
    #   50_000  → ~3 min  (quick check)
    #   200_000 → ~10 min (good baseline)
    #   500_000 → ~25 min (strong baseline)
    TIMESTEPS = DEFAULT_TIMESTEPS
    model = train_vanilla_ppo(train_eps, total_timesteps=TIMESTEPS)

    # Step 4: evaluate both agents on test set
    log("\nEvaluating test set")
    df_bs  = evaluate_black_scholes(test_eps)
    df_ppo = evaluate_ppo(model, test_eps)

    # Step 5: metrics and report
    m_bs  = compute_metrics(df_bs,  "Black-Scholes")
    m_ppo = compute_metrics(df_ppo, "Vanilla PPO+LSTM")
    print_report([m_bs, m_ppo], df_bs, df_ppo)

    # Save compact summary only; per-step frames are intentionally kept in memory.
    pd.DataFrame([m_bs, m_ppo]).to_csv(
        "results/baseline_summary.csv", index=False)

    log("\nSaved: results/baseline_summary.csv")
    log("Saved: models/vanilla_ppo_lstm.zip")
    log("Vanilla PPO baseline complete.")
