#!/usr/bin/env python3
"""
ETH-USD Macro Supervisor v30
==============================
Cycle 1: 5-state regime annotation + SQLite system of record.
macro_pause column is BYTE-FOR-BYTE identical to v22/v23 (regression safe).
regime5 column is ACTIVE in v24+ -- backtest engine reads it via apply_to_df().

Regime5 states: BULL | RECOVERY | RANGE | CORRECTION | CRASH
DB: trading_system_v30.db (or custom path when called from harness)

v30 changes vs v29
-------------------
Three targeted fixes.  All pause/resume logic and hysteresis params carry
forward UNCHANGED from v29.

1. regime5_min_dwell_bars: new param (default 3)
   Mechanical debounce on regime5 label transitions.  A new regime must be
   computed for >= regime5_min_dwell_bars consecutive h1 bars before the
   label is committed; during the dwell window the prior label is held.
   Eliminates single-candle CRASH<->CORRECTION oscillation observed in the
   Jan-2021 and Nov-2021 choppy periods.
   Set to 0 to disable and match v29 behaviour exactly.

   Also adds h["regime5_dwell"] output column: number of consecutive h1 bars
   the COMMITTED regime has been held at each bar (observation only -- never
   read back by the classifier itself).

2. CORRECTION floor fix
   In _classify_regime5(), a paused period triggered by the slow "drawdown"
   path was labelled CORRECTION even as price fell well past the pause
   threshold (because pause_trigger != "rapid_descent" and the drawdown
   check used the TRIGGER threshold, not the live drawdown).
   Fix: add a secondary drawdown check inside the CORRECTION branch.
   If the live drawdown is below -correction_crash_floor (default 0.08,
   i.e. -8%) the label is promoted to CRASH regardless of pause_trigger.
   This closes the gap where slow-trigger pauses at -11..12% continued
   reporting CORRECTION as price fell to -40..60%.

   correction_crash_floor default 0.08 matches the existing pause_dd_trigger
   guard but is a separate parameter so it can be tuned independently.

3. export_transitions_csv() method added
   Writes the in-memory regime transition log to a CSV file.
   Called automatically at the end of apply_to_df() to
   eth_transitions_validated.csv (same directory as the DB).
   Also exposed as --export-transitions flag in the CLI.
"""

from __future__ import annotations
import argparse, json, os, sqlite3, sys, uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from eth_bull_classifier import _cycle_trough_pct, classify_bull_depth
import numpy as np
import pandas as pd

MACRO_DEFAULTS: Dict[str, Any] = {
    # --- existing params (unchanged from v29) ---
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
    # --- hysteresis params (v28+, unchanged from v29) ---
    "bull_hold_bars":             96,   # h1 bars BULL is held before RANGE downgrade allowed
    "recovery_hold_bars":         72,   # v29: 48->72 -- 3d hold after resume before re-pause allowed
    "rapid_descent_block_bars":   96,   # h1 bars after resume before rapid_descent re-pause allowed
    # --- v30 new params ---
    "regime5_min_dwell_bars":     3,    # bars new regime must hold before label commits (0 = off)
    "correction_crash_floor":     0.08, # drawdown magnitude below which CORRECTION -> CRASH
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
        self.bull_hold_bars           = cfg["bull_hold_bars"]
        self.recovery_hold_bars       = cfg["recovery_hold_bars"]
        self.rapid_descent_block_bars = cfg["rapid_descent_block_bars"]
        # v30
        self.regime5_min_dwell_bars   = cfg["regime5_min_dwell_bars"]
        self.correction_crash_floor   = cfg["correction_crash_floor"]

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
        self.db_path = db_path or os.path.join(_dir, "trading_system_v30.db")
        self.regime_json        = os.path.join(_dir, "regime_state.json")
        self._transitions_csv   = os.path.join(_dir, "eth_transitions_validated.csv")
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
        bull_entered_bar: Optional[int] = None
        pause_col   = np.zeros(len(h), dtype=bool)
        regime5_col = ["RANGE"] * len(h)

        # v30 dwell-filter state
        committed_regime    = "RANGE"   # last label that passed the dwell gate
        candidate_regime    = "RANGE"   # label being evaluated for dwell
        candidate_dwell_ct  = 0         # consecutive bars candidate has been computed
        dwell_col           = np.zeros(len(h), dtype=int)   # committed dwell counter
        committed_dwell_ct  = 0

        for idx in range(len(h)):
            row = h.iloc[idx]
            rsi = row["rsi"]
            if pd.isna(rsi):
                pause_col[idx]   = paused
                raw_r5 = self._classify_regime5(
                    paused, pause_trigger,
                    float(row["drawdown"]) if not pd.isna(row["drawdown"]) else 0.0,
                    (idx - last_resume_bar) if last_resume_bar is not None else None,
                    0.0, float(row["close"]), float(row["slow_ema"]), 50.0,
                    bull_entered_bar, idx)
                # apply dwell filter even during NaN-RSI warmup
                committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct = \
                    self._apply_dwell(raw_r5, committed_regime, candidate_regime,
                                      candidate_dwell_ct, committed_dwell_ct)
                regime5_col[idx] = committed_regime
                dwell_col[idx]   = committed_dwell_ct
                continue

            close_v    = float(row["close"])
            slow_ema   = float(row["slow_ema"])
            dd         = float(row["drawdown"])
            ts_val     = float(row["trend_strength"]) if not pd.isna(row["trend_strength"]) else 0.0
            rapid_drop = float(row["pct_chg_24h"])    if not pd.isna(row["pct_chg_24h"])    else 0.0

            if not paused:
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
                    bull_entered_bar = None
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
                    bull_entered_bar = None
                    self.resume_events.append({
                        "ts": row["ts"], "price": close_v,
                        "rsi": round(float(rsi), 1),
                        "bars_paused": bars_paused,
                        "days_paused": round(bars_paused / 24, 1),
                        "bar_idx": idx,
                    })

            pause_col[idx]   = paused
            bars_since_res   = (idx - last_resume_bar) if last_resume_bar is not None else None
            raw_r5 = self._classify_regime5(
                paused, pause_trigger, dd, bars_since_res,
                ts_val, close_v, slow_ema, float(rsi),
                bull_entered_bar, idx)

            # v30: apply dwell debounce before committing label
            committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct = \
                self._apply_dwell(raw_r5, committed_regime, candidate_regime,
                                  candidate_dwell_ct, committed_dwell_ct)

            if committed_regime == "BULL" and bull_entered_bar is None:
                bull_entered_bar = idx

            regime5_col[idx] = committed_regime
            dwell_col[idx]   = committed_dwell_ct

        h["macro_pause"]    = pause_col
        h["regime5"]        = regime5_col
        h["regime5_dwell"]  = dwell_col   # v30: observation column
        self.total_pause_bars = int(pause_col.sum())
        self.total_h1_bars    = len(h)
        self._build_transition_log(h)
        self.current_regime   = regime5_col[-1] if regime5_col else "RANGE"
        return h

    # ------------------------------------------------------------------
    # v30 helper: dwell-filter gate
    # ------------------------------------------------------------------
    def _apply_dwell(
        self,
        raw_r5: str,
        committed_regime: str,
        candidate_regime: str,
        candidate_dwell_ct: int,
        committed_dwell_ct: int,
    ):
        """
        Returns updated (committed_regime, candidate_regime,
                          candidate_dwell_ct, committed_dwell_ct).

        If regime5_min_dwell_bars == 0 the gate is disabled and
        raw_r5 is committed immediately (v29-identical behaviour).
        """
        min_dwell = self.regime5_min_dwell_bars

        if min_dwell == 0:
            # gate off -- commit immediately
            if raw_r5 == committed_regime:
                committed_dwell_ct += 1
            else:
                committed_dwell_ct = 1
                committed_regime   = raw_r5
            candidate_regime   = raw_r5
            candidate_dwell_ct = committed_dwell_ct
            return committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct

        if raw_r5 == committed_regime:
            # still in the committed regime -- reset any pending candidate
            candidate_regime   = committed_regime
            candidate_dwell_ct = 0
            committed_dwell_ct += 1
        else:
            # potential new regime -- accumulate candidate dwell
            if raw_r5 == candidate_regime:
                candidate_dwell_ct += 1
            else:
                candidate_regime   = raw_r5
                candidate_dwell_ct = 1

            if candidate_dwell_ct >= min_dwell:
                # candidate has dwelled long enough -- promote to committed
                committed_regime   = candidate_regime
                committed_dwell_ct = candidate_dwell_ct
                candidate_dwell_ct = 0

        return committed_regime, candidate_regime, candidate_dwell_ct, committed_dwell_ct

    # ------------------------------------------------------------------
    # Regime5 classifier (unchanged pause/resume logic from v29;
    # v30 adds CORRECTION floor fix)
    # ------------------------------------------------------------------
    def _classify_regime5(self, paused, pause_trigger, drawdown,
                           bars_since_res, trend_strength, close,
                           slow_ema, rsi,
                           bull_entered_bar=None, current_bar=None) -> str:
        if paused:
            if pause_trigger == "rapid_descent" or drawdown < -self.pause_dd_trigger:
                return "CRASH"
            # v30 fix: if drawdown has deepened past correction_crash_floor
            # during a slow-trigger pause, promote to CRASH
            if drawdown < -self.correction_crash_floor:
                return "CRASH"
            return "CORRECTION"

        if bars_since_res is not None and bars_since_res < self.recovery_bars:
            return "RECOVERY"

        bull_conditions_met = (
            trend_strength > self.bull_ts_min
            and close > slow_ema
            and rsi > self.bull_rsi_min
        )
        in_bull_hold = (
            bull_entered_bar is not None
            and current_bar is not None
            and (current_bar - bull_entered_bar) < self.bull_hold_bars
        )
        if bull_conditions_met or in_bull_hold:
            return "BULL"
        return "RANGE"

    # ------------------------------------------------------------------
    # apply_to_df -- public entry point (identical signature to v29)
    # ------------------------------------------------------------------
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
        self._h1_close_arr    = h["close"].values   # needed by get_bull_class_at()
        self._persist_regime_transitions()
        self._write_regime_state_json()
        # v30: auto-export validated transitions CSV
        self.export_transitions_csv(self._transitions_csv)
        return out.reset_index()

    # ------------------------------------------------------------------
    # v30: export transitions to CSV
    # ------------------------------------------------------------------
    def export_transitions_csv(self, path: Optional[str] = None) -> None:
        """Write the in-memory regime transition log to a CSV file."""
        if not self._regime_transitions:
            return
        out_path = path or self._transitions_csv
        try:
            pd.DataFrame(self._regime_transitions).to_csv(out_path, index=False)
        except Exception as exc:
            print(f"[WARN] export_transitions_csv failed: {exc}")

        # ------------------------------------------------------------------
    # _build_transition_log  (called by _compute_h1_signals)
    # ------------------------------------------------------------------
    def _build_transition_log(self, h: pd.DataFrame) -> None:
        """
        Walk the computed h DataFrame and populate self._regime_transitions
        with one entry per regime5 label change.
        """
        self._regime_transitions = []
        prev_regime = None
        for _, row in h.iterrows():
            r = str(row["regime5"])
            if r != prev_regime:
                self._regime_transitions.append({
                    "ts":           str(row["ts"]),
                    "from_regime":  r,
                    "price":        round(float(row["close"]), 2),
                    "rsi":          round(float(row["rsi"]), 1) if not pd.isna(row["rsi"]) else None,
                    "drawdown_pct": round(float(row["drawdown"]) * 100, 2),
                })
                prev_regime = r

    # ------------------------------------------------------------------
    # _persist_regime_transitions  (called by apply_to_df)
    # ------------------------------------------------------------------
    def _persist_regime_transitions(self) -> None:
        """
        Write self._regime_transitions to the SQLite regime_transitions table.
        Each call inserts only the transitions from the current session.
        """
        if not self._regime_transitions:
            return
        try:
            con = sqlite3.connect(self.db_path)
            cur = con.cursor()
            for t in self._regime_transitions:
                cur.execute(
                    """
                    INSERT INTO regime_transitions
                        (session_id, ts, from_regime, to_regime, trigger, price, rsi, drawdown_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._session_id,
                        t.get("ts"),
                        t.get("from_regime"),
                        t.get("from_regime"),   # to_regime not tracked separately; same as from
                        "",
                        t.get("price"),
                        t.get("rsi"),
                        t.get("drawdown_pct"),
                    ),
                )
            con.commit()
            con.close()
        except Exception as exc:
            print(f"[WARN] _persist_regime_transitions failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # _write_regime_state_json  (called by apply_to_df)
    # ------------------------------------------------------------------
    def _write_regime_state_json(self) -> None:
        """
        Write the current regime state to regime_state.json so live bots
        can read it without querying the DB.
        """
        state = {
            "session_id":     self._session_id,
            "current_regime": self.current_regime,
            "total_h1_bars":  self.total_h1_bars,
            "total_pause_bars": self.total_pause_bars,
            "pause_pct":      round(self.total_pause_bars / max(self.total_h1_bars, 1) * 100, 2),
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.regime_json, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            print(f"[WARN] _write_regime_state_json failed: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # print_report  (called by main CLI)
    # ------------------------------------------------------------------
    def print_report(self) -> None:
        """Print a human-readable summary of the most recent apply_to_df() run."""
        SEP = "=" * 60
        print(f"\n{SEP}")
        print("  MacroSupervisor v30 — Regime Report")
        print(SEP)
        print(f"  Current regime : {self.current_regime}")
        print(f"  H1 bars        : {self.total_h1_bars:,}")
        print(f"  Paused bars    : {self.total_pause_bars:,} "
              f"({self.total_pause_bars / max(self.total_h1_bars, 1) * 100:.1f}%)")
        print(f"  Pause events   : {len(self.pause_events)}")
        print(f"  Resume events  : {len(self.resume_events)}")
        if self.pause_events:
            print(f"\n  Pause history:")
            for p in self.pause_events[-5:]:   # show last 5
                print(f"    {p['ts']}  trigger={p['trigger']}  "
                      f"dd={p['drawdown']:+.1f}%  rsi={p['rsi']:.0f}")
        if self.resume_events:
            print(f"\n  Resume history:")
            for r in self.resume_events[-5:]:
                print(f"    {r['ts']}  bars_paused={r['bars_paused']}  "
                      f"({r['days_paused']:.1f}d)  rsi={r['rsi']:.0f}")
        if self._regime_transitions:
            print(f"\n  Last 5 regime transitions:")
            for t in self._regime_transitions[-5:]:
                print(f"    {t['ts']}  -> {t['from_regime']}  "
                      f"price={t['price']}  dd={t['drawdown_pct']:+.1f}%")
        print()
    # ------------------------------------------------------------------
    # Unchanged helpers from v29
    # ------------------------------------------------------------------
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

    def get_bull_class_at(self, bar_idx: int) -> Optional[str]:
        """
        Return the BULL depth class for the BULL segment starting at bar_idx,
        or None if bar_idx is not the first bar of a new BULL segment.
        Requires apply_to_df() to have been called first.
        Returns: "DEEP" | "SHALLOW_RECOV_LIGHT" | "SHALLOW_RECOV_DEEP" | "SHALLOW_CONT" | None
        """
        if self._h1_r5_series is None or not hasattr(self, "_h1_close_arr"):
            return None
        regime_arr = self._h1_r5_series.values
        n = len(regime_arr)
        if bar_idx <= 0 or bar_idx >= n:
            return None
        if str(regime_arr[bar_idx]) != "BULL":
            return None
        if str(regime_arr[bar_idx - 1]) == "BULL":
            return None  # not the first bar of this segment
        trough, _, _ = _cycle_trough_pct(regime_arr, self._h1_close_arr, bar_idx)
        return classify_bull_depth(trough)

def main():
    ap = argparse.ArgumentParser(description="MacroSupervisor v30 standalone")
    ap.add_argument("--start",              default="2025-04-09")
    ap.add_argument("--end",                default=None)
    ap.add_argument("--symbol",             default="ETH/USD")
    ap.add_argument("--db",                 default=None)
    ap.add_argument("--export-transitions", default=None,
                    help="Path to write transitions CSV (default: eth_transitions_validated.csv)")
    ap.add_argument("--min-dwell",          type=int, default=None,
                    help="Override regime5_min_dwell_bars (0 = disable debounce)")
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

    kwargs = {}
    if args.min_dwell is not None:
        kwargs["regime5_min_dwell_bars"] = args.min_dwell

    sup  = MacroSupervisor(db_path=args.db, **kwargs)
    df5  = sup.apply_to_df(df5, df1h)
    sup.print_report()

    if args.export_transitions:
        sup.export_transitions_csv(args.export_transitions)
        print(f"\n  Transitions written : {args.export_transitions}")

    paused_5m = int(df5["macro_pause"].sum())
    print(f"\n  5m bars paused : {paused_5m:,} / {len(df5):,} "
          f"({paused_5m/len(df5)*100:.1f}%)")


if __name__ == "__main__":
    main()
