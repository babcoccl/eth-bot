#!/usr/bin/env python3
"""
eth_helpers_v2.py  —  Updated data-fetch + indicator utilities
============================================================
Supports MacroSupervisor v31 (Orchestrator Edition).

Exports
-------
IND                 : dict   — indicator hyper-parameters
fetch_ohlcv()       : Coinbase public REST candle fetch (disk-cached)
prepare_indicators(): join 5m features + h1 regime_h1 column (uses v31)
"""

import os, sys, hashlib, warnings, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from eth_macrosupervisor_v31 import MacroSupervisor
from eth_bull_classifier import (
    _cycle_trough_pct,
    classify_bull_depth,
)

warnings.filterwarnings("ignore")

IND = {
    "rsi_period":   14,
    "atr_period":   14,
    "bb_window":    20,
    "bb_std":       2.0,
    "zscore_window": 24,
    "vol_window":   20,
    "regime_fast":  20,
    "regime_slow":  50,
}

_CACHE_DIR = Path(__file__).parent / "ohlcv_cache"
_CACHE_DIR.mkdir(exist_ok=True)

def _cache_key(symbol: str, timeframe: str, since_dt: datetime, until_dt: datetime) -> str:
    raw = f"{symbol}|{timeframe}|{since_dt.date()}|{until_dt.date()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _is_historical(until_dt: datetime) -> bool:
    now_ts = datetime.now(timezone.utc).timestamp()
    return (now_ts - until_dt.timestamp()) > 86400

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_bollinger(close, window=20, std_mult=2.0):
    mid   = close.rolling(window).mean()
    std   = close.rolling(window).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw    = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, bw

def calc_zscore(close, window=24):
    m = close.rolling(window).mean()
    s = close.rolling(window).std().replace(0, np.nan)
    return (close - m) / s

def fetch_ohlcv(symbol, timeframe, since_dt, until_dt, use_cache=True):
    # This just redirects to the original fetch_ohlcv in eth_helpers to avoid code duplication
    from eth_helpers import fetch_ohlcv as original_fetch
    return original_fetch(symbol, timeframe, since_dt, until_dt, use_cache)

def prepare_indicators(df5, df1h, min_dwell=3, conviction=1.0):
    """
    Join 5m OHLCV with computed indicators and h1 regime label (v31).
    """
    d = df5.copy()
    d["rsi"]      = calc_rsi(d["close"], IND["rsi_period"])
    d["rsi_prev"] = d["rsi"].shift(1)
    d["atr"]      = calc_atr(d["high"], d["low"], d["close"], IND["atr_period"])
    d["atr_pct"]  = d["atr"] / d["close"].replace(0, np.nan)
    MACRO_DD_WINDOW = 90 * 288
    d["rolling_90d_high"] = d["close"].rolling(MACRO_DD_WINDOW, min_periods=1).max()
    d["macro_dd_pct"]     = (d["close"] - d["rolling_90d_high"]) / d["rolling_90d_high"]
    d["zscore"]   = calc_zscore(d["close"], IND["zscore_window"])

    bb_u, bb_m, bb_l, bw = calc_bollinger(d["close"], IND["bb_window"], IND["bb_std"])
    d["bb_upper"] = bb_u
    d["bb_mid"]   = bb_m
    d["bb_lower"] = bb_l
    d["bw_pct"]   = bw

    d["fast_ma"]       = d["close"].ewm(span=IND["regime_fast"], adjust=False).mean()
    d["slow_ma"]       = d["close"].ewm(span=IND["regime_slow"], adjust=False).mean()
    d["trend_strength"]= (d["fast_ma"] - d["slow_ma"]) / d["slow_ma"].replace(0, np.nan)
    d["vol_ma"]        = d["volume"].rolling(IND["vol_window"]).mean()
    d["vol_ratio"]     = d["volume"] / d["vol_ma"].replace(0, np.nan)
    d["candle_body"]   = (d["close"] - d["open"]).abs()
    d["lower_wick"]    = d[["open", "close"]].min(axis=1) - d["low"]

    # Use MacroSupervisor v31
    sup = MacroSupervisor(regime5_min_dwell_bars=min_dwell)
    sup.advisor_bridge_enabled = False # Disable live file read for backtest
    sup.conviction_score = conviction # Set conviction for backtest
    h   = sup._compute_h1_signals(df1h)
    
    # Join regime5 onto 5m bars
    h_regime = h[["ts", "macro_pause", "regime5"]].set_index("ts")
    result = d.set_index("ts").join(h_regime, how="left")
    result["macro_pause"] = result["macro_pause"].ffill().fillna(False).astype(bool)
    result["regime5"]     = result["regime5"].ffill().fillna("RANGE")
    
    # Join bull class
    regime_arr = h["regime5"].values
    close_arr  = h["close"].values
    n          = len(regime_arr)
    bull_class_col = [""] * n
    for i in range(1, n):
        if str(regime_arr[i]) == "BULL" and str(regime_arr[i - 1]) != "BULL":
            trough, _, _ = _cycle_trough_pct(regime_arr, close_arr, i)
            bull_class_col[i] = classify_bull_depth(trough)
        elif str(regime_arr[i]) == "BULL":
            bull_class_col[i] = bull_class_col[i - 1]

    h1_bull = df1h[["ts"]].copy()
    h1_bull["bull_class_h1"] = bull_class_col
    result = result.join(h1_bull.set_index("ts")[["bull_class_h1"]], how="left")
    result["bull_class_h1"] = result["bull_class_h1"].ffill().fillna("")
    
    return result.reset_index()
