#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
from eth_bot_interface import BotInterface, BotStatus, Position, Lot

warnings.filterwarnings("ignore")

PRESETS = {
    "dcb_v1": {
        "base_qty":           0.05,
        "lookback_bars":      12 * 24 * 3, # 3 days of 5m bars
        "min_drop_pct":       0.05,        # Crash must be at least 5%
        "vol_ratio_max":      0.90,        # Volume must be relatively weak on the bounce
        "fib_entry_low":      0.382,
        "fib_entry_high":     0.500,
        "fib_stop":           0.786,
        "max_hold_bars":      12 * 24 * 2, # 2 days time stop
        "fee_pct":            0.00065,
    },
    "dcb_v1_tuned": {
        "base_qty":           0.05,
        "lookback_bars":      12 * 24 * 2, # 2 days
        "min_drop_pct":       0.04,        # 4% minimum drop
        "vol_ratio_max":      0.85,        # strict volume filter
        "fib_entry_low":      0.382,
        "fib_entry_high":     0.500,       # Strict user requirement
        "fib_stop":           0.786,
        "max_hold_bars":      12 * 24 * 2, # 2 days time stop
        "fee_pct":            0.00065,
    },
    "dcb_v2_optimized": {
        "base_qty":           0.05,
        "lookback_bars":      12 * 24 * 2, # Optimal: 2 days
        "min_drop_pct":       0.03,        # Optimal: 3% minimum drop
        "vol_ratio_max":      0.85,        # Optimal: 0.85 volume filter
        "fib_entry_low":      0.382,
        "fib_entry_high":     0.500,       # Strict user requirement
        "fib_stop":           0.786,
        "max_hold_bars":      12 * 24 * 2, # 2 days time stop
        "fee_pct":            0.00065,
    }
}

class RecoveryBot(BotInterface):
    """
    RECOVERY/DOWNTREND regime specialist — Dead Cat Bounce short strategy.
    """
    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol   = symbol
        self._active   = True
        self._position = Position(symbol=symbol, side="SHORT")
        self._cash     = 0.0
        self._capital  = 0.0
        self._realized_pnl = 0.0
        self._trade_count  = 0
        self._equity_curve = []
        self._trades       = []
        self._cumulative   = 0.0
        self._last_trade_ts = None

    @property
    def bot_id(self) -> str:
        return f"recoverybot_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["RECOVERY", "DOWNTREND"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id            = self.bot_id,
            symbol            = self._symbol,
            capital_allocated = self._capital,
            capital_deployed  = self._position.qty * self._position.avg_entry,
            capital_available = self._cash,
            open_side         = self._position.side,
            open_qty          = self._position.qty, 
            open_avg_entry    = self._position.avg_entry,
            unrealized_pnl    = 0.0,
            realized_pnl      = self._realized_pnl,
            trade_count       = self._trade_count,
            active            = self._active,
            supported_regimes = self.supported_regimes,
        )

    def _reset(self, capital: float) -> None:
        self._cash         = float(capital)
        self._capital      = float(capital)
        self._position     = Position(symbol=self._symbol, side="SHORT")
        self._trades       = []
        self._equity_curve = []
        self._cumulative   = 0.0
        self._last_trade_ts = None

    def _sell_short(self, i, ts, regime_h1, rsi, fill_price, reason, base_qty, fee_pct):
        # To short, we need margin. We assume 1x leverage, meaning we need cash >= short value.
        sv = base_qty * fill_price
        if sv > self._cash:
            return False
            
        fee = sv * fee_pct
        self._cash -= fee # Deduct fee from cash
        self._position.qty = base_qty
        self._position.avg_entry = fill_price
        self._position.entry_bar = i
        self._position.lots = [Lot(qty=base_qty, price=fill_price, fee=fee, ts=ts, row_idx=len(self._trades))]
        self._last_trade_ts = ts
        
        self._trades.append({
            "ts": ts, "side": "SELL_SHORT", "reason": reason,
            "regime_h1": str(regime_h1),
            "price": fill_price, "qty": base_qty, "fee": fee,
            "rsi": float(rsi),
            "pnl": 0.0, "pnl_after_fees": 0.0, "win": float("nan"),
            "bars_held": float("nan"), "exit_price": float("nan"),
        })
        return True

    def _buy_to_cover(self, i, ts, regime_h1, rsi, fill_price, reason, fee_pct):
        p = self._position
        if not p.is_open:
            return False
            
        lot = p.lots[0]
        
        # PnL logic for Short: (Entry - Exit) * Qty
        gross_pnl = (lot.price - fill_price) * p.qty
        exit_fee = (p.qty * fill_price) * fee_pct
        net_pnl = gross_pnl - lot.fee - exit_fee
        
        self._cash += net_pnl
        self._cumulative += net_pnl
        if reason != "end_of_period":
            self._realized_pnl += net_pnl
            self._trade_count += 1
            
        bh = i - p.entry_bar
        entry_row = self._trades[lot.row_idx]
        entry_row.update({
            "pnl": net_pnl, "pnl_after_fees": net_pnl,
            "win": 1.0 if net_pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": fill_price,
        })
        
        self._trades.append({
            "ts": ts, "side": "BUY_COVER", "reason": reason,
            "regime_h1": str(regime_h1),
            "price": fill_price, "qty": p.qty, "fee": exit_fee,
            "rsi": float(rsi),
            "pnl": net_pnl, "pnl_after_fees": net_pnl,
            "win": 1.0 if net_pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": float("nan"),
        })
        p.reset()
        return True

    def run_backtest(self, df: pd.DataFrame, preset: dict, capital: float, preset_name: str) -> tuple:
        self._reset(capital)
        p = preset
        fee_pct      = p["fee_pct"]
        base_qty     = p["base_qty"]
        lookback     = p["lookback_bars"]
        min_drop     = p["min_drop_pct"]
        vol_max      = p["vol_ratio_max"]
        f_low        = p["fib_entry_low"]
        f_high       = p["fib_entry_high"]
        f_stop       = p["fib_stop"]
        max_hold     = p["max_hold_bars"]

        # We need rolling max high over lookback bars
        df["rolling_high"] = df["high"].rolling(window=lookback, min_periods=1).max()

        # Extract columns to numpy arrays for speed
        ts_arr      = df["ts"].values
        regime_arr  = df["regime5"].values.astype(str) if "regime5" in df.columns else np.full(len(df), "RANGE")
        low_arr     = df["low"].values.astype(float)
        high_arr    = df["high"].values.astype(float)
        close_arr   = df["close"].values.astype(float)
        open_arr    = df["open"].values.astype(float)
        roll_hi_arr = df["rolling_high"].values.astype(float)

        # Optional columns
        regime_h1_arr = df["regime_h1"].values.astype(str) if "regime_h1" in df.columns else np.full(len(df), "")
        rsi_arr       = df["rsi"].values.astype(float) if "rsi" in df.columns else np.full(len(df), np.nan)
        vol_ratio_arr = df["vol_ratio"].values.astype(float) if "vol_ratio" in df.columns else np.full(len(df), 1.0)

        for i in range(len(df)):
            ts      = ts_arr[i]
            regime  = regime_arr[i]
            low     = low_arr[i]
            high    = high_arr[i]
            close   = close_arr[i]
            open_p  = open_arr[i]

            regime_h1 = regime_h1_arr[i]
            rsi       = rsi_arr[i]

            unreal = self._position.unrealized_pnl(close, fee_pct)
            self._equity_curve.append(self._cash + unreal)

            if self._position.is_open:
                bh = i - self._position.entry_bar
                
                if high >= getattr(self, "_active_fib_stop", 999999):
                    self._buy_to_cover(i, ts, regime_h1, rsi, getattr(self, "_active_fib_stop", high), "stop_loss_fib", fee_pct)
                    continue
                    
                if low <= getattr(self, "_active_macro_low", 0):
                    self._buy_to_cover(i, ts, regime_h1, rsi, getattr(self, "_active_macro_low", low), "target_macro_low", fee_pct)
                    continue
                    
                if bh >= max_hold:
                    self._buy_to_cover(i, ts, regime_h1, rsi, close, "time_stop", fee_pct)
                    continue
                    
                continue

            if i < lookback:
                continue

            if regime not in self.supported_regimes:
                continue

            # Optimization: only recalculate if current high is in a reasonable range to be a fib
            # Using rolling_high is fast.
            macro_peak = roll_hi_arr[i]
            # Find when that peak occurred in the window
            window_high = high_arr[i-lookback:i+1]
            peak_idx = window_high.argmax() + (i - lookback)
            if peak_idx >= i:
                continue # Peak is right now
                
            macro_low = low_arr[peak_idx:i+1].min()
            
            drop_pct = (macro_peak - macro_low) / macro_peak
            if drop_pct < min_drop:
                continue
                
            drop_val = macro_peak - macro_low
            fib_382 = macro_low + drop_val * f_low
            fib_500 = macro_low + drop_val * f_high
            fib_786 = macro_low + drop_val * f_stop
            
            # Entry condition: Retest 0.382 - 0.50 zone
            if high >= fib_382 and close <= fib_500:
                if close < open_p: # Close red
                    vol_r = vol_ratio_arr[i]
                    if vol_r <= vol_max: # Weak volume
                        if self._sell_short(i, ts, regime_h1, rsi, close, "dcb_short", base_qty, fee_pct):
                            self._active_macro_low = macro_low
                            self._active_fib_stop  = fib_786

        if self._position.is_open:
            last_idx = len(df) - 1
            self._buy_to_cover(last_idx, ts_arr[last_idx], regime_h1_arr[last_idx], rsi_arr[last_idx], close_arr[last_idx], "end_of_period", fee_pct)

        return self._build_result(capital, preset_name)

    def _build_result(self, capital: float, preset_name: str) -> tuple:
        if not self._trades:
            return pd.DataFrame(), {}
        tdf     = pd.DataFrame(self._trades)
        eq      = np.array(self._equity_curve)
        peak_eq = np.maximum.accumulate(eq)
        max_dd  = float(((eq - peak_eq) / peak_eq).min()) * 100
        ret_s   = np.diff(eq) / eq[:-1]
        sharpe  = (ret_s.mean() / ret_s.std() * np.sqrt(105_120)
                   if ret_s.std() > 0 else 0.0)
                   
        covers = tdf[tdf["side"] == "BUY_COVER"]
        real   = covers[covers["reason"] != "end_of_period"]
        wins   = real[real["pnl_after_fees"] > 0]
        
        psl_r  = real[real["reason"] == "stop_loss_fib"]
        tgt_r  = real[real["reason"] == "target_macro_low"]
        tst_r  = real[real["reason"] == "time_stop"]
        
        return tdf, {
            "preset":          preset_name,
            "trades":          len(real),
            "win_rate":        len(wins) / len(real) * 100 if len(real) > 0 else 0,
            "realized_pnl":    tdf["pnl_after_fees"].sum() if "pnl_after_fees" in tdf.columns else 0.0,
            "total_return":    self._cumulative / capital * 100,
            "final_equity":    capital + self._cumulative,
            "max_drawdown":    max_dd,
            "sharpe":          sharpe,
            "fees":            tdf["fee"].sum(),
            "target_fires":    len(tgt_r),
            "target_pnl":      float(tgt_r["pnl_after_fees"].sum()) if len(tgt_r) > 0 else 0.0,
            "psl_fires":       len(psl_r),
            "psl_pnl":         float(psl_r["pnl_after_fees"].sum()) if len(psl_r) > 0 else 0.0,
            "time_stop_fires": len(tst_r),
            "time_stop_pnl":   float(tst_r["pnl_after_fees"].sum()) if len(tst_r) > 0 else 0.0,
            "time_stop_pct":   len(tst_r) / len(real) * 100 if len(real) > 0 else 0.0,
            "avg_bars_held":   float(real["bars_held"].mean()) if len(real) > 0 else 0,
        }
