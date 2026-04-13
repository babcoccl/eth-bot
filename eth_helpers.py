#!/usr/bin/env python3
"""
eth_helpers.py  —  Shared data-fetch + indicator utilities
============================================================
FROZEN after v28.  Do NOT version this file.
If indicator logic must change: rename to eth_helpers_v2.py,
update ARCHITECTURE.md and eth_helpers.pyi simultaneously.

Exports
-------
IND                 : dict   — indicator hyper-parameters
fetch_ohlcv()       : Coinbase public REST candle fetch
calc_rsi()          : Wilder RSI
calc_atr()          : EWM ATR
calc_bollinger()    : Bollinger Bands + bandwidth
calc_zscore()       : rolling z-score
calc_regime()       : h1 regime (UPTREND / DOWNTREND / RANGE)
prepare_indicators(): join 5m features + h1 regime_h1 column
"""

import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Indicator hyper-parameters (never changed since v22) ─────────────────────
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


# ── Pure indicator math ───────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
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


def calc_regime(close_h1, fast=20, slow=50):
    fast_ma = close_h1.ewm(span=fast, adjust=False).mean()
    slow_ma = close_h1.ewm(span=slow, adjust=False).mean()
    regime  = pd.Series("RANGE", index=close_h1.index)
    regime[close_h1 > slow_ma * 1.002] = "UPTREND"
    regime[close_h1 < slow_ma * 0.998] = "DOWNTREND"
    return regime, fast_ma, slow_ma


# ── Coinbase public REST fetch ────────────────────────────────────────────────

def fetch_ohlcv(symbol, timeframe, since_dt, until_dt):
    import time as _time
    TF_MAP = {"1m": 60, "5m": 300, "15m": 900,
              "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
    gran = TF_MAP.get(timeframe)
    if gran is None:
        print(f"[ERROR] Unsupported timeframe: {timeframe}.", file=sys.stderr)
        sys.exit(1)
    try:
        import requests as _req
    except ImportError:
        print("[ERROR] pip install requests", file=sys.stderr)
        sys.exit(1)

    product  = symbol.replace("/", "-")
    base_url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    headers  = {"User-Agent": "eth-strategy-backtest/28"}
    MAX_BARS = 300
    window   = gran * MAX_BARS
    now_ts   = int(datetime.now(timezone.utc).timestamp())
    since_ts = int(since_dt.timestamp())
    until_ts = min(int(until_dt.timestamp()), now_ts)
    all_bars = []
    cursor   = since_ts

    while cursor < until_ts:
        end_ts  = min(cursor + window, until_ts)
        start_s = datetime.fromtimestamp(cursor,  tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_s   = datetime.fromtimestamp(end_ts,  tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for attempt in range(4):
            try:
                r = _req.get(base_url,
                             params={"granularity": gran, "start": start_s, "end": end_s},
                             headers=headers, timeout=30)
                r.raise_for_status()
                bars = r.json()
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"[ERROR] Coinbase fetch failed: {exc}", file=sys.stderr)
                    sys.exit(1)
                _time.sleep(2 ** attempt)
        if isinstance(bars, dict) and "message" in bars:
            print(f"[ERROR] Coinbase API: {bars['message']}", file=sys.stderr)
            sys.exit(1)
        if not bars:
            cursor = end_ts
            continue
        for b in sorted(bars, key=lambda b: b[0]):
            ts_bar = int(b[0])
            if since_ts <= ts_bar <= until_ts:
                all_bars.append([
                    ts_bar * 1000,
                    float(b[3]), float(b[2]),
                    float(b[1]), float(b[4]), float(b[5]),
                ])
        cursor = end_ts
        _time.sleep(0.12)

    if not all_bars:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = (df[(df["ts"] >= pd.Timestamp(since_dt)) & (df["ts"] <= pd.Timestamp(until_dt))]
          .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_indicators(df5, df1h):
    """
    Join 5m OHLCV with computed indicators and h1 regime label.

    Returns DataFrame with columns:
        ts, open, high, low, close, volume,
        rsi, rsi_prev, atr, atr_pct, zscore,
        bb_upper, bb_mid, bb_lower, bw_pct,
        fast_ma, slow_ma, trend_strength,
        vol_ma, vol_ratio, candle_body, lower_wick,
        regime_h1  (UPTREND | DOWNTREND | RANGE, forward-filled from 1h)
    NaN rows on rsi/bb_upper/zscore are dropped.
    """
    d = df5.copy()
    d["rsi"]      = calc_rsi(d["close"], IND["rsi_period"])
    d["rsi_prev"] = d["rsi"].shift(1)
    d["atr"]      = calc_atr(d["high"], d["low"], d["close"], IND["atr_period"])
    d["atr_pct"]  = d["atr"] / d["close"].replace(0, np.nan)
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

    h1 = df1h.copy()
    h1["regime_h1"], _, _ = calc_regime(h1["close"], IND["regime_fast"], IND["regime_slow"])
    h1 = h1[["ts", "regime_h1"]].set_index("ts")

    d = d.set_index("ts").join(h1, how="left")
    d["regime_h1"] = d["regime_h1"].ffill().fillna("RANGE")
    result = d.reset_index().dropna(subset=["rsi", "bb_upper", "zscore"]).reset_index(drop=True)

    for col in ["rsi", "trend_strength", "atr_pct"]:
        null_count = result[col].isna().sum()
        if null_count > 0:
            print(f"[WARN] {null_count} NaN in '{col}' after prepare_indicators "
                  f"(first at row {result[col].isna().idxmax()})")
    return result
