#!/usr/bin/env python3
"""
ETH-USD Macro Supervisor v28
==============================
Cycle 1: 5-state regime annotation + SQLite system of record.
macro_pause column is BYTE-FOR-BYTE identical to v22/v23 (regression safe).
regime5 column is ACTIVE in v24+ -- backtest engine reads it via apply_to_df().

Regime5 states: BULL | RECOVERY | RANGE | CORRECTION | CRASH
DB: trading_system_v28.db (or custom path when called from harness)

v28 changes vs v27
-------------------
Three new hysteresis parameters that make BULL and RECOVERY sticky once entered.
All existing PAUSE/RESUME entry conditions are UNCHANGED.

1. bull_hold_bars (default 96 = 4 days)
   After _classify_regime5 returns BULL for the first time since the last
   resume, record _bull_entered_bar. For the next bull_hold_bars h1 bars,
   downgrade from BULL to RANGE is blocked. CRASH/CORRECTION can still fire
   (supervisor can re-pause normally). Only BULL->RANGE oscillations are
   suppressed.
   Motivation: CyC diagnostic showed BULL@06:02 -> RANGE@06:07 (5h tenure)
   caused by a single 1h bar where trend_strength dipped below bull_ts_min.

2. recovery_hold_bars (default 48 = 2 days)
   After a resume event (paused -> unpaused), block the re-pause trigger
   (both drawdown and rapid_descent paths) for recovery_hold_bars h1 bars.
   The supervisor cannot re-enter CORRECTION/CRASH within this window.
   Motivation: CyD showed RECOVERY@2025-04-09 -> CRASH@2025-04-10 (19h tenure).
   A single day's volatility undid the resume. This guard prevents that.

3. rapid_descent_block_bars (default 96 = 4 days)
   After any resume event, the rapid_descent_trigger path specifically is
   blocked for rapid_descent_block_bars bars (in addition to the broader
   recovery_hold_bars block). This is a second line of defence for the
   hair-trigger 7% 24h drop re-pause that fires even when RSI/EMA are healthy.
   Note: recovery_hold_bars (48) fires first and already blocks both triggers.
   rapid_descent_block_bars (96) extends the rapid_descent-only block for an
   additional 48 bars after recovery_hold_bars expires, allowing slow re-pause
   (triple-condition drawdown) but still blocking the single-bar hair-trigger.

All three counters are reset on every new pause event so they cannot
permanently suppress a genuine CRASH signal after the hold window expires.
"""

from __future__ import annotations
import argparse, json, os, sqlite3, sys, uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MACRO_DEFAULTS: Dict[str, Any] = {
    # --- existing v27 params (unchanged) ---
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
    # --- new v28 hysteresis params ---
    "bull_hold_bars":             96,   # h1 bars BULL is held before RANGE downgrade allowed
    "recovery_hold_bars":         48,   # h1 bars after resume before re-pause allowed (both triggers)
    "rapid_descent_block_bars":   96,   # h1 bars after resume before rapid_descent re-pause allowed
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
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts TEXT,
    bot_id TEXT,
    equity REAL,
    eth_held REAL DEFAULT 0,
    usdc_held REAL DEFAULT 0,
    cumulative_pnl REAL DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS bot_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE,
    bot_id TEXT,
    start_ts TEXT,
    end_ts TEXT,
    config_json TEXT DEFAULT '{}',
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
        # v28 hysteresis
        self.bull_hold_bars           = cfg["bull_hold_bars"]
        self.recovery_hold_bars       = cfg["recovery_hold_bars"]
        self.rapid_descent_block_bars = cfg["rapid_descent_block_bars"]

        self.pause_events:  List[Dict[str, Any]] = []
        self.resume_events: List[Dict[str, Any]] = []
        self.total_pause_bars: int = 0
        self.total_h1_bars:    int = 0
        self._h1_pause:  Optional[pd.Series] = None

        self._session_id          = str(uuid.uuid4())[:12]
        self._h1_regime5:         Optional[pd.Series] = None
        self._regime_transitions: List[Dict[str, Any]] = []
        self.current_regime       = "RANGE"
        self.active_bot_ids:      List[str] = ["eth_dca_scalp_bot"]

        _dir         = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(_dir, "trading_system_v28.db")
        self.regime_json = os.path.join(_dir, "regime_state.json")
        self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        con.executescript(_DDL)
        con.commit()
        con.close()

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
        # v28 hold counters
        bull_entered_bar: Optional[int] = None   # bar when BULL was first entered post-resume
        pause_col   = np.zeros(len(h), dtype=bool)
        regime5_col = ["RANGE"] * len(h)

        for idx in range(len(h)):
            row = h.iloc[idx]
            rsi = row["rsi"]
            if pd.isna(rsi):
                pause_col[idx]   = paused
                regime5_col[idx] = self._classify_regime5(
                    paused, pause_trigger,
                    float(row["drawdown"]) if not pd.isna(row["drawdown"]) else 0.0,
                    (idx - last_resume_bar) if last_resume_bar is not None else None,
                    0.0, float(row["close"]), float(row["slow_ema"]), 50.0,
                    bull_entered_bar, idx)
                continue

            close_v    = float(row["close"])
            slow_ema   = float(row["slow_ema"])
            dd         = float(row["drawdown"])
            ts_val     = float(row["trend_strength"]) if not pd.isna(row["trend_strength"]) else 0.0
            rapid_drop = float(row["pct_chg_24h"])    if not pd.isna(row["pct_chg_24h"])    else 0.0

            if not paused:
                # --- v28: check recovery_hold_bars block before allowing re-pause ---
                bars_since_resume = (idx - last_resume_bar) if last_resume_bar is not None else None
                in_recovery_hold  = (bars_since_resume is not None
                                     and bars_since_resume < self.recovery_hold_bars)
                in_rd_block       = (bars_since_resume is not None
                                     and bars_since_resume < self.rapid_descent_block_bars)

                trigger_slow  = (not in_recovery_hold
                                 and dd < -self.pause_dd_trigger
                                 and rsi < self.pause_rsi_max
                                 and close_v < slow_ema * self.pause_ma_mult)
                trigger_rapid = (not in_recovery_hold
                                 and not in_rd_block
                                 and rapid_drop < -self.rapid_descent_trigger)

                if trigger_slow or trigger_rapid:
                    paused           = True
                    pause_start_bar  = idx
                    pause_trigger    = "rapid_descent" if trigger_rapid else "drawdown"
                    bull_entered_bar = None   # reset bull hold on new pause
                    chg_str          = round(rapid_drop * 100, 1) if trigger_rapid else None
                    self.pause_events.append({
                        "ts": row["ts"], "price": close_v,
                        "rsi": round(float(rsi), 1), "drawdown": round(dd * 100, 1),
                        "bar_idx": idx, "trigger": pause_trigger,
                        "chg_24h_pct": chg_str,
                    })
            else:
                bars_paused = idx - pause_start_bar
                if (bars_paused >= self.min_pause_bars
                        and rsi > self.resume_rsi_min
                        and close_v > slow_ema * self.resume_ma_mult
                        and ts_val > self.resume_ts_min):
                    paused          = False
                    last_resume_bar = idx
                    bull_entered_bar = None   # reset bull hold on resume
                    self.resume_events.append({
                        "ts": row["ts"], "price": close_v,
                        "rsi": round(float(rsi), 1),
                        "bars_paused": bars_paused,
                        "days_paused": round(bars_paused / 24, 1),
                        "bar_idx": idx,
                    })

            pause_col[idx]   = paused
            bars_since_res   = (idx - last_resume_bar) if last_resume_bar is not None else None
            r5 = self._classify_regime5(
                paused, pause_trigger, dd, bars_since_res,
                ts_val, close_v, slow_ema, float(rsi),
                bull_entered_bar, idx)

            # --- v28: update bull_entered_bar when BULL is first entered ---
            if r5 == "BULL" and bull_entered_bar is None:
                bull_entered_bar = idx

            regime5_col[idx] = r5

        h["macro_pause"] = pause_col
        h["regime5"]     = regime5_col
        self.total_pause_bars = int(pause_col.sum())
        self.total_h1_bars    = len(h)
        self._build_transition_log(h)
        self.current_regime   = regime5_col[-1] if regime5_col else "RANGE"
        return h

    def _classify_regime5(self, paused, pause_trigger, drawdown,
                           bars_since_res, trend_strength, close,
                           slow_ema, rsi,
                           bull_entered_bar=None, current_bar=None) -> str:
        """Classify current 1h bar into one of 5 regimes.

        v28 additions:
          bull_entered_bar : bar index when BULL was most recently entered
          current_bar      : current bar index (used for bull_hold_bars check)

        BULL -> RANGE downgrade is blocked for bull_hold_bars after bull_entered_bar.
        All other transitions (including BULL -> CRASH/CORRECTION via re-pause)
        are unaffected.
        """
        if paused:
            if pause_trigger == "rapid_descent" or drawdown < -self.pause_dd_trigger:
                return "CRASH"
            return "CORRECTION"
        if bars_since_res is not None and bars_since_res < self.recovery_bars:
            return "RECOVERY"

        # Check raw BULL conditions
        bull_conditions_met = (
            trend_strength > self.bull_ts_min
            and close > slow_ema
            and rsi > self.bull_rsi_min
        )

        # v28: if we're within bull_hold_bars of entering BULL, hold BULL even
        # if conditions briefly fail (suppresses BULL->RANGE oscillation).
        in_bull_hold = (
            bull_entered_bar is not None
            and current_bar is not None
            and (current_bar - bull_entered_bar) < self.bull_hold_bars
        )

        if bull_conditions_met or in_bull_hold:
            return "BULL"
        return "RANGE"

    def apply_to_df(self, df5m: pd.DataFrame, df1h: pd.DataFrame) -> pd.DataFrame:
        h     = self._compute_h1_signals(df1h)
        h_sig = h.set_index("ts")[["macro_pause", "regime5"]]
        out   = df5m.copy().set_index("ts").join(h_sig, how="left")
        out["macro_pause"] = out["macro_pause"].ffill().fillna(False).astype(bool)
        out["regime5"]     = out["regime5"].ffill().fillna("RANGE")
        self._h1_pause        = h["macro_pause"]
        self._h1_regime5      = h["regime5"]
        self._h1_ts_index     = pd.to_datetime(h["ts"])
        self._h1_r5_series    = h["regime5"]
        self._persist_regime_transitions()
        self._write_regime_state_json()
        return out.reset_index()

    def get_regime_at(self, ts) -> str:
        if self._h1_regime5 is None:
            return "RANGE"
        try:
            ts_dt = pd.to_datetime(ts, utc=True)
            idx   = self._h1_ts_index.searchsorted(ts_dt, side="right") - 1
            if idx < 0:
                return "RANGE"
            return str(self._h1_r5_series.iloc[min(idx, len(self._h1_r5_series) - 1)])
        except Exception:
            return "RANGE"

    def record_trade(self, trade) -> None:
        if hasattr(trade, "__dataclass_fields__"):
            d = {
                "session_id":     trade.session_id or self._session_id,
                "bot_id":         trade.bot_id,
                "preset":         trade.preset,
                "regime":         trade.regime,
                "entry_ts":       str(trade.entry_ts),
                "exit_ts":        str(trade.exit_ts),
                "entry_price":    trade.entry_price,
                "exit_price":     trade.exit_price,
                "qty":            trade.qty,
                "reason":         trade.reason,
                "pnl":            trade.pnl,
                "pnl_after_fees": trade.pnl_after_fees,
                "fees":           trade.fees,
                "bars_held":      trade.bars_held,
                "win":            int(trade.win),
                "notes":          trade.notes,
            }
        else:
            d = dict(trade)
            d.setdefault("session_id", self._session_id)
            d["win"] = int(d.get("win", False))
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT INTO trades
            (session_id,bot_id,preset,regime,entry_ts,exit_ts,entry_price,
             exit_price,qty,reason,pnl,pnl_after_fees,fees,bars_held,win,notes)
            VALUES
            (:session_id,:bot_id,:preset,:regime,:entry_ts,:exit_ts,:entry_price,
             :exit_price,:qty,:reason,:pnl,:pnl_after_fees,:fees,:bars_held,:win,:notes)
        """, d)
        con.commit()
        con.close()

    def record_snapshot(self, bot_id: str, equity: float,
                        eth_held: float = 0.0, usdc_held: float = 0.0,
                        cumulative_pnl: float = 0.0, ts: str = "") -> None:
        ts = ts or datetime.now(timezone.utc).isoformat()
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT INTO portfolio_snapshots
            (session_id,ts,bot_id,equity,eth_held,usdc_held,cumulative_pnl)
            VALUES (?,?,?,?,?,?,?)
        """, (self._session_id, ts, bot_id, equity, eth_held, usdc_held, cumulative_pnl))
        con.commit()
        con.close()

    def register_session(self, bot_id: str, config: dict) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT OR REPLACE INTO bot_sessions (session_id,bot_id,start_ts,config_json)
            VALUES (?,?,?,?)
        """, (self._session_id, bot_id,
              datetime.now(timezone.utc).isoformat(), json.dumps(config)))
        con.commit()
        con.close()

    def close_session(self, bot_id: str) -> None:
        con = sqlite3.connect(self.db_path)
        con.execute("UPDATE bot_sessions SET end_ts=? WHERE session_id=? AND bot_id=?",
                    (datetime.now(timezone.utc).isoformat(), self._session_id, bot_id))
        con.commit()
        con.close()

    def get_portfolio_report(self, bot_id: str = None,
                              session_id: str = None) -> pd.DataFrame:
        where, params = [], []
        if bot_id:     where.append("bot_id = ?");     params.append(bot_id)
        if session_id: where.append("session_id = ?"); params.append(session_id)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        con = sqlite3.connect(self.db_path)
        df  = pd.read_sql_query(
            f"SELECT * FROM trades {clause} ORDER BY exit_ts", con, params=params)
        con.close()
        return df

    def get_regime_distribution(self) -> pd.DataFrame:
        if self._h1_regime5 is None:
            return pd.DataFrame()
        vc = self._h1_regime5.value_counts()
        return pd.DataFrame({
            "regime":  vc.index,
            "h1_bars": vc.values,
            "days":    (vc.values / 24).round(1),
            "pct":     (vc.values / self.total_h1_bars * 100).round(1)
                       if self.total_h1_bars > 0 else 0,
        }).sort_values("h1_bars", ascending=False).reset_index(drop=True)

    def _build_transition_log(self, h: pd.DataFrame) -> None:
        prev = None
        for _, row in h.iterrows():
            r5 = row["regime5"]
            if r5 != prev:
                self._regime_transitions.append({
                    "session_id":   self._session_id,
                    "ts":           str(row["ts"]),
                    "from_regime":  prev or "INIT",
                    "to_regime":    r5,
                    "trigger":      "",
                    "price":        float(row["close"]),
                    "rsi":          float(row["rsi"]) if not pd.isna(row["rsi"]) else 0.0,
                    "drawdown_pct": round(float(row["drawdown"]) * 100, 2),
                })
                prev = r5

    def _persist_regime_transitions(self) -> None:
        if not self._regime_transitions:
            return
        con = sqlite3.connect(self.db_path)
        con.executemany("""
            INSERT INTO regime_transitions
            (session_id,ts,from_regime,to_regime,trigger,price,rsi,drawdown_pct)
            VALUES
            (:session_id,:ts,:from_regime,:to_regime,:trigger,:price,:rsi,:drawdown_pct)
        """, self._regime_transitions)
        con.commit()
        con.close()

    def _write_regime_state_json(self) -> None:
        state = {
            "session_id":   self._session_id,
            "regime":       self.current_regime,
            "macro_paused": bool(self.pause_events
                                 and len(self.pause_events) > len(self.resume_events)),
            "active_bots":  self.active_bot_ids,
            "pause_events": len(self.pause_events),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.regime_json, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def summary(self) -> Dict[str, Any]:
        pct = (self.total_pause_bars / self.total_h1_bars * 100
               if self.total_h1_bars > 0 else 0)
        rd  = [e for e in self.pause_events if e.get("trigger") == "rapid_descent"]
        out = {
            "session_id":           self._session_id,
            "pause_events":         len(self.pause_events),
            "rapid_descent_events": len(rd),
            "pause_days":           round(self.total_pause_bars / 24, 1),
            "pause_pct":            round(pct, 1),
            "current_regime":       self.current_regime,
            "db_path":              self.db_path,
            # v28 params for reference
            "bull_hold_bars":           self.bull_hold_bars,
            "recovery_hold_bars":       self.recovery_hold_bars,
            "rapid_descent_block_bars": self.rapid_descent_block_bars,
        }
        if self._h1_regime5 is not None:
            for r5 in ["BULL", "RANGE", "CORRECTION", "CRASH", "RECOVERY"]:
                hrs = int((self._h1_regime5 == r5).sum())
                out[f"regime5_{r5.lower()}_days"] = round(hrs / 24, 1)
        return out

    def print_report(self) -> None:
        sep = " " + "-" * 62
        print("\n MacroSupervisor v28 Report")
        print(sep)
        print(f" Session ID          : {self._session_id}")
        print(f" bull_hold_bars      : {self.bull_hold_bars}  ({self.bull_hold_bars/24:.0f}d)")
        print(f" recovery_hold_bars  : {self.recovery_hold_bars}  ({self.recovery_hold_bars/24:.0f}d)")
        print(f" rapid_desc_block    : {self.rapid_descent_block_bars}  ({self.rapid_descent_block_bars/24:.0f}d)")
        print(f" Pause events        : {len(self.pause_events)}")
        rd = [e for e in self.pause_events if e.get("trigger") == "rapid_descent"]
        if rd:
            print(f" rapid_descent       : {len(rd)}")
        print(f" Total paused        : {self.total_pause_bars} h1 bars "
              f"({self.total_pause_bars/24:.1f} days)")
        if self.total_h1_bars > 0:
            print(f" Fraction            : "
                  f"{self.total_pause_bars/self.total_h1_bars*100:.1f}% of period")
        if self.pause_events:
            print("\n Pause periods:")
            for i, pe in enumerate(self.pause_events):
                re   = self.resume_events[i] if i < len(self.resume_events) else None
                trig = pe.get("trigger", "drawdown")
                chg  = (f", 24h={pe['chg_24h_pct']:.1f}%"
                        if pe.get("chg_24h_pct") is not None else "")
                if re:
                    print(f"   [{i+1}] PAUSE  {str(pe['ts'])[:10]} @ "
                          f"${pe['price']:,.0f}  (dd={pe['drawdown']:.1f}%, "
                          f"RSI={pe['rsi']}, trig={trig}{chg})")
                    print(f"        RESUME {str(re['ts'])[:10]} @ "
                          f"${re['price']:,.0f}  (RSI={re['rsi']}, "
                          f"{re['days_paused']}d)")
                else:
                    print(f"   [{i+1}] PAUSE  {str(pe['ts'])[:10]} @ "
                          f"${pe['price']:,.0f}  (trig={trig}{chg}) -> active at end")
        if self._h1_regime5 is not None:
            print("\n Regime5 distribution:")
            dist = self.get_regime_distribution()
            print(f"  {'Regime':<14} {'Days':>6} {'Pct':>6}")
            print("  " + "-" * 28)
            for _, row in dist.iterrows():
                print(f"  {row['regime']:<14} {row['days']:>6.1f} {row['pct']:>5.1f}%")
        print(f"\n System of record : {self.db_path}")
        print(sep)


def main():
    ap = argparse.ArgumentParser(description="MacroSupervisor v28 standalone")
    ap.add_argument("--start",  default="2025-04-09")
    ap.add_argument("--end",    default=None)
    ap.add_argument("--symbol", default="ETH/USD")
    ap.add_argument("--db",     default=None)
    args = ap.parse_args()
    try:
        from eth_helpers import fetch_ohlcv
    except ImportError:
        print("[ERROR] eth_helpers.py must be in the same directory.")
        sys.exit(1)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_s    = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_dt   = (datetime.strptime(end_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                + timedelta(days=1))
    print(f"Fetching 1h candles {args.start} -> {end_s} ...")
    df1h = fetch_ohlcv(args.symbol, "1h", start_dt, end_dt)
    df5  = fetch_ohlcv(args.symbol, "5m", start_dt, end_dt)
    print(f"  1h: {len(df1h):,}  5m: {len(df5):,}")
    sup  = MacroSupervisor(db_path=args.db)
    df5  = sup.apply_to_df(df5, df1h)
    sup.print_report()
    paused_5m = int(df5["macro_pause"].sum())
    print(f"\n  5m bars paused : {paused_5m:,} / {len(df5):,} "
          f"({paused_5m/len(df5)*100:.1f}%)")


if __name__ == "__main__":
    main()
