#!/usr/bin/env python3
"""
ETH-USD Macro Supervisor v31 (Orchestrator Edition)
===================================================
Evolution of v30 with Orchestrator-level scaling and LLM Advisor integration.

v31 changes vs v30
-------------------
1. Advisor Bridge: Reads 'advisor_state.json' for 'conviction' score (0.0 to 1.0).
2. Dynamic Scaling: Provides get_capital_scale() method for tactical bots.
3. Bot Registry: Tracks supported regimes for centralized lifecycle management.
4. Enhanced State Persistence: regime_state.json now includes advisor notes.
"""

from __future__ import annotations
import argparse, json, os, sqlite3, sys, uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from eth_bull_classifier import _cycle_trough_pct, classify_bull_depth
import numpy as np
import pandas as pd

MACRO_DEFAULTS: Dict[str, Any] = {
    "pause_dd_trigger":           0.12,
    "pause_rsi_max":              38,
    "pause_ma_mult":              0.998,
    "peak_window":                720,
    "min_pause_h1_bars":          336,
    "resume_rsi_min":             55,
    "resume_ma_mult":             1.010,
    "resume_ts_min":              -0.002,
    "ema_fast":                   20,
    "ema_slow":                   50,
    "rsi_period":                 14,
    "rapid_descent_h24_trigger":  0.07,
    "bull_ts_min":                0.003,
    "bull_rsi_min":               50,
    "recovery_bars":              168,
    "bull_hold_bars":             96,
    "recovery_hold_bars":         72,
    "rapid_descent_block_bars":   96,
    "regime5_min_dwell_bars":     3,
    "correction_crash_floor":     0.08,
}

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    bot_id TEXT NOT NULL,
    preset TEXT DEFAULT '',
    regime TEXT,
    entry_ts TEXT,
    exit_ts TEXT,
    entry_price REAL,
    exit_price REAL,
    qty REAL,
    reason TEXT,
    pnl REAL,
    pnl_after_fees REAL,
    fees REAL,
    bars_held INTEGER,
    win INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS regime_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts TEXT,
    from_regime TEXT,
    to_regime TEXT,
    trigger TEXT DEFAULT '',
    price REAL,
    rsi REAL,
    drawdown_pct REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS advisor_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts TEXT,
    conviction REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

class MacroSupervisor:
    def __init__(self, db_path: Optional[str] = None, **kwargs):
        cfg = {**MACRO_DEFAULTS, **kwargs}
        self.pause_dd_trigger         = cfg["pause_dd_trigger"]
        self.pause_rsi_max            = cfg["pause_rsi_max"]
        self.pause_ma_mult            = cfg["pause_ma_mult"]
        self.peak_window              = cfg["peak_window"]
        self.min_pause_bars           = cfg["min_pause_h1_bars"]
        self.resume_rsi_min           = cfg["resume_rsi_min"]
        self.resume_ma_mult           = cfg["resume_ma_mult"]
        self.resume_ts_min            = cfg["resume_ts_min"]
        self.ema_fast                 = cfg["ema_fast"]
        self.ema_slow                 = cfg["ema_slow"]
        self.rsi_period               = cfg["rsi_period"]
        self.rapid_descent_trigger    = cfg["rapid_descent_h24_trigger"]
        self.bull_ts_min              = cfg["bull_ts_min"]
        self.bull_rsi_min             = cfg["bull_rsi_min"]
        self.recovery_bars            = cfg["recovery_bars"]
        self.bull_hold_bars           = cfg["bull_hold_bars"]
        self.recovery_hold_bars       = cfg["recovery_hold_bars"]
        self.rapid_descent_block_bars = cfg["rapid_descent_block_bars"]
        self.regime5_min_dwell_bars   = cfg["regime5_min_dwell_bars"]
        self.correction_crash_floor   = cfg["correction_crash_floor"]

        self._session_id          = str(uuid.uuid4())[:12]
        self.current_regime       = "RANGE"
        self.pause_events         = []
        self.resume_events        = []
        self.total_pause_bars     = 0
        self.total_h1_bars        = 0
        self._h1_regime5          = None
        self._regime_transitions  = []
        
        # Capital Orchestration (v32)
        self.total_capital        = 10000.0  # Global pool
        self.bot_weights = {
            "trendbot":      0.25,  # 25%
            "rangebot":      0.25,  # 25%
            "correctionbot": 0.20,  # 20%
            "recoverybot":   0.15,  # 15%
            "hedgebot":      0.15,  # 15%
        }
        self._deployed_capital = {} # bot_id -> float
        
        # Advisor / Orchestration State
        self.conviction_score     = 1.0
        self.advisor_notes        = "System default: High Conviction"
        self._risk_events         = [] # list of (timestamp, bot_id, requested, allowed, reason)
        
        _dir         = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(_dir, "trading_system_v30.db")
        self.regime_json        = os.path.join(_dir, "regime_state.json")
        self.advisor_json       = os.path.join(_dir, "advisor_state.json")
        self._transitions_csv   = os.path.join(_dir, "eth_transitions_validated.csv")
        
        self.advisor_bridge_enabled = True
        self._init_db()
        self._load_advisor_state()

    def request_allocation(self, bot_id: str, requested_amount: float) -> float:
        """
        Enforces Budgeting & Throttling.
        Returns the allowed allocation amount (0.0 to requested_amount).
        """
        # Determine strategy type from bot_id (e.g. trendbot_eth_usd -> trendbot)
        strategy = bot_id.split("_")[0].lower()
        weight = self.bot_weights.get(strategy, 0.0)
        
        # Hard Budget Cap
        total_budget = self.total_capital * weight
        current_usage = self._deployed_capital.get(bot_id, 0.0)
        headroom = max(0.0, total_budget - current_usage)
        
        # Conviction Scaling
        conviction = self.get_capital_scale()
        adjusted_request = requested_amount * conviction
        
        # Final Allowed (Budget limited)
        allowed = min(adjusted_request, headroom)
        
        if allowed < requested_amount:
            reason = "Budget Cap" if headroom < adjusted_request else "Low Conviction"
            self._risk_events.append({
                "ts": datetime.now().isoformat() if hasattr(self, 'now') else "SIM",
                "bot_id": bot_id,
                "requested": requested_amount,
                "allowed": allowed,
                "reason": reason
            })
            
        return allowed

    def generate_llm_digest(self, bots: list) -> str:
        """
        Generates a high-density Markdown report for an LLM Analyst (Jules).
        Summarizes global state, fleet performance, and risk events.
        """
        import json
        lines = []
        lines.append("# ETH TRADING SYSTEM: Performance Digest")
        lines.append(f"**Session ID:** {self._session_id}")
        lines.append(f"**Current Regime:** {self.current_regime}")
        lines.append(f"**Conviction Score:** {self.conviction_score}")
        lines.append(f"**Total Capital Pool:** ${self.total_capital:,.2f}")
        lines.append("")
        
        lines.append("## Fleet Summary")
        lines.append("| Bot ID | Equity | Realized PnL | Deployed | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for b in bots:
            s = b.get_state_summary()
            deployed = self._deployed_capital.get(b.bot_id, 0.0)
            status = "ACTIVE" if s.get("active_position") else "IDLE"
            lines.append(f"| {b.bot_id} | ${s['equity']:,.2f} | ${s['realized_pnl']:,.2f} | ${deployed:,.2f} | {status} |")
        lines.append("")
        
        if self._risk_events:
            lines.append("## Risk & Throttling Events (Recent)")
            for e in self._risk_events[-10:]:
                lines.append(f"- **{e['bot_id']}**: Requested ${e['requested']:.2f} -> Allowed ${e['allowed']:.2f} (Reason: {e['reason']})")
            lines.append("")
            
        lines.append("## Recent Tactical History")
        for b in bots:
            trades = b.get_recent_trades(3)
            if trades:
                lines.append(f"### {b.bot_id}")
                for t in trades:
                    side = t.get("side", "N/A")
                    reason = t.get("reason", "N/A")
                    pnl = t.get("pnl_after_fees", 0.0)
                    lines.append(f"- {side} @ {t.get('price', 0.0):.2f} ({reason}) | PnL: ${pnl:.2f}")
        
        lines.append("")
        lines.append("## System Logs & Observations")
        lines.append(f"- Advisor Notes: {self.advisor_notes}")
        
        return "\n".join(lines)

    def update_bot_status_realtime(self, bot_id: str, deployed_amount: float):
        """Called by bots or orchestrator loop to update the central ledger."""
        self._deployed_capital[bot_id] = deployed_amount

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.executescript(_DDL)
        con.commit()
        con.close()

    def _load_advisor_state(self) -> None:
        """Read external conviction signals from the LLM Advisor module."""
        if not self.advisor_bridge_enabled or not os.path.exists(self.advisor_json):
            return
        try:
            with open(self.advisor_json, "r") as f:
                data = json.load(f)
                self.conviction_score = float(data.get("conviction", 1.0))
                self.advisor_notes    = data.get("notes", "No notes provided")
        except Exception as exc:
            print(f"[WARN] Failed to load advisor state: {exc}")

    def get_capital_scale(self) -> float:
        """Returns a multiplier (0.0 to 1.0) for tactical bot capital deployment."""
        # Always refresh state before returning scale if in a live loop
        self._load_advisor_state()
        return max(0.0, min(1.0, self.conviction_score))

    def _calc_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_h1_signals(self, df1h: pd.DataFrame) -> pd.DataFrame:
        h = df1h.copy().sort_values("ts").reset_index(drop=True)
        h["fast_ema"]       = h["close"].ewm(span=self.ema_fast, adjust=False).mean()
        h["slow_ema"]       = h["close"].ewm(span=self.ema_slow, adjust=False).mean()
        h["trend_strength"] = ((h["fast_ema"] - h["slow_ema"])
                               / h["slow_ema"].replace(0, np.nan))
        h["rsi"]            = self._calc_rsi(h["close"])
        h["rolling_peak"]   = h["close"].rolling(self.peak_window, min_periods=1).max()
        h["drawdown"]       = (h["close"] - h["rolling_peak"]) / h["rolling_peak"]
        h["pct_chg_24h"]    = h["close"].pct_change(24)

        paused            = False
        pause_start_bar   = None
        pause_trigger     = ""
        last_resume_bar:  Optional[int] = None
        bull_entered_bar: Optional[int] = None
        pause_col   = np.zeros(len(h), dtype=bool)
        regime5_col = ["RANGE"] * len(h)

        committed_regime    = "RANGE"
        candidate_regime    = "RANGE"
        candidate_dwell_ct  = 0
        dwell_col           = np.zeros(len(h), dtype=int)
        committed_dwell_ct  = 0

        for idx in range(len(h)):
            row = h.iloc[idx]
            rsi = row["rsi"]
            close_v    = float(row["close"])
            slow_ema   = float(row["slow_ema"])
            dd         = float(row["drawdown"]) if not pd.isna(row["drawdown"]) else 0.0
            ts_val     = float(row["trend_strength"]) if not pd.isna(row["trend_strength"]) else 0.0
            rapid_drop = float(row["pct_chg_24h"])    if not pd.isna(row["pct_chg_24h"])    else 0.0

            if pd.isna(rsi):
                pause_col[idx]   = paused
                raw_r5 = self._classify_regime5(paused, pause_trigger, dd, None, 0.0, close_v, slow_ema, 50.0)
            else:
                if not paused:
                    bars_since_resume = (idx - last_resume_bar) if last_resume_bar is not None else None
                    in_recovery_hold  = (bars_since_resume is not None and bars_since_resume < self.recovery_hold_bars)
                    in_rd_block       = (bars_since_resume is not None and bars_since_resume < self.rapid_descent_block_bars)

                    trigger_slow  = (not in_recovery_hold and dd < -self.pause_dd_trigger and rsi < self.pause_rsi_max and close_v < slow_ema * self.pause_ma_mult)
                    trigger_rapid = (not in_recovery_hold and not in_rd_block and rapid_drop < -self.rapid_descent_trigger)

                    if trigger_slow or trigger_rapid:
                        paused           = True
                        pause_start_bar  = idx
                        pause_trigger    = "rapid_descent" if trigger_rapid else "drawdown"
                        self.pause_events.append({"ts": row["ts"], "price": close_v, "trigger": pause_trigger})
                else:
                    bars_paused = idx - pause_start_bar
                    if (bars_paused >= self.min_pause_bars and rsi > self.resume_rsi_min and close_v > slow_ema * self.resume_ma_mult and ts_val > self.resume_ts_min):
                        paused          = False
                        last_resume_bar = idx
                        self.resume_events.append({"ts": row["ts"], "price": close_v, "bars_paused": bars_paused})

                pause_col[idx]   = paused
                bars_since_res   = (idx - last_resume_bar) if last_resume_bar is not None else None
                raw_r5 = self._classify_regime5(paused, pause_trigger, dd, bars_since_res, ts_val, close_v, slow_ema, float(rsi), bull_entered_bar, idx)

            committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct = self._apply_dwell(raw_r5, committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct)
            if committed_regime == "BULL" and bull_entered_bar is None: bull_entered_bar = idx
            regime5_col[idx] = committed_regime
            dwell_col[idx]   = committed_dwell_ct

        h["macro_pause"]    = pause_col
        h["regime5"]        = regime5_col
        h["regime5_dwell"]  = dwell_col
        self.total_pause_bars = int(pause_col.sum())
        self.total_h1_bars    = len(h)
        self.current_regime   = regime5_col[-1] if regime5_col else "RANGE"
        self._build_transition_log(h)
        return h

    def _apply_dwell(self, raw_r5, committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct):
        min_dwell = self.regime5_min_dwell_bars
        if min_dwell == 0:
            if raw_r5 == committed_regime: committed_dwell_ct += 1
            else:
                committed_dwell_ct = 1
                committed_regime = raw_r5
            return committed_regime, raw_r5, committed_dwell_ct, committed_dwell_ct

        if raw_r5 == committed_regime:
            candidate_dwell_ct = 0
            committed_dwell_ct += 1
        else:
            if raw_r5 == candidate_regime: candidate_dwell_ct += 1
            else:
                candidate_regime = raw_r5
                candidate_dwell_ct = 1
            if candidate_dwell_ct >= min_dwell:
                committed_regime = candidate_regime
                committed_dwell_ct = candidate_dwell_ct
                candidate_dwell_ct = 0
        return committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct

    def _classify_regime5(self, paused, pause_trigger, drawdown, bars_since_res, trend_strength, close, slow_ema, rsi, bull_entered_bar=None, current_bar=None) -> str:
        if paused:
            if pause_trigger == "rapid_descent" or drawdown < -self.pause_dd_trigger or drawdown < -self.correction_crash_floor:
                return "CRASH"
            return "CORRECTION"
        if bars_since_res is not None and bars_since_res < self.recovery_bars:
            return "RECOVERY"
        bull_conditions_met = (trend_strength > self.bull_ts_min and close > slow_ema and rsi > self.bull_rsi_min)
        in_bull_hold = (bull_entered_bar is not None and current_bar is not None and (current_bar - bull_entered_bar) < self.bull_hold_bars)
        if bull_conditions_met or in_bull_hold: return "BULL"
        return "RANGE"

    def apply_to_df(self, df5m: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
        h     = self._compute_h1_signals(df1h)
        h_sig = h.set_index("ts")[["macro_pause", "regime5", "rsi"]]
        out   = df5m.copy().set_index("ts").join(h_sig, how="left")
        out["macro_pause"] = out["macro_pause"].ffill().fillna(False).astype(bool)
        out["regime5"]     = out["regime5"].ffill().fillna("RANGE")
        out["rsi"]         = out["rsi"].ffill()
        self._h1_ts_index     = pd.to_datetime(h["ts"])
        self._h1_r5_series    = h["regime5"]
        self._h1_close_arr    = h["close"].values
        self._write_regime_state_json()
        return out.reset_index()

    def _build_transition_log(self, h: pd.DataFrame) -> None:
        self._regime_transitions = []
        prev_regime = None
        for _, row in h.iterrows():
            r = str(row["regime5"])
            if r != prev_regime:
                self._regime_transitions.append({"ts": str(row["ts"]), "from_regime": r, "price": round(float(row["close"]), 2)})
                prev_regime = r

    def _write_regime_state_json(self) -> None:
        state = {
            "session_id":     self._session_id,
            "current_regime": self.current_regime,
            "conviction":     self.conviction_score,
            "advisor_notes":  self.advisor_notes,
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.regime_json, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            print(f"[WARN] Failed to write regime state: {exc}")

    def export_transitions_csv(self, path: Optional[str] = None) -> None:
        if not self._regime_transitions: return
        try: pd.DataFrame(self._regime_transitions).to_csv(path or self._transitions_csv, index=False)
        except Exception as exc: print(f"[WARN] export_transitions_csv failed: {exc}")

if __name__ == "__main__":
    sup = MacroSupervisor()
    print(f"MacroSupervisor v31 initialized. Conviction: {sup.conviction_score}")
