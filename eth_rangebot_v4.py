#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
from eth_bot_interface import BotInterface, BotStatus, Position, Lot
from eth_persistence_v1 import BotStateStore

warnings.filterwarnings("ignore")

PRESETS = {
    "grid_v1": {
        "base_qty":           0.05,
        "grid_levels":        10,
        "grid_step_bps":      40,       # 0.4% spacing
        "pos_stop_loss_pct":  0.06,     # 6% catastrophic stop
        "drift_threshold":    0.0015,   # Pause grid if trend_strength > 0.15%
        "fee_pct":            0.00065,
    },
    "grid_v1_tight": {
        "base_qty":           0.05,
        "grid_levels":        15,
        "grid_step_bps":      20,       # 0.2% spacing
        "pos_stop_loss_pct":  0.05,
        "fee_pct":            0.00065,
    },
}

class RangeBot(BotInterface):
    """
    RANGE/RECOVERY regime specialist — Grid Trading architecture.

    Instead of trying to snipe a single bottom entry, this bot deploys a ladder
    of buy and sell limit orders (simulated via intra-bar highs and lows) to
    capture sideways price oscillations.
    """
    def __init__(self, symbol: str = "ETH-USD"):
        self._symbol   = symbol
        self._active   = True
        self._position = Position(symbol=symbol)
        self._cash     = 0.0
        self._capital  = 0.0
        self._realized_pnl = 0.0
        self._trade_count  = 0
        
        # Grid state
        self._grid_active = False
        self._base_price  = 0.0
        self._grid_step   = 0.0
        self._buy_levels  = []
        self._sell_levels = []
        self._equity_curve = []
        self._trades       = []
        self._cumulative   = 0.0
        self._store = BotStateStore(self.bot_id)

    def save_to_disk(self):
        extra = {
            "grid_active": self._grid_active,
            "base_price":  self._base_price,
            "grid_step":   self._grid_step,
            "buy_levels":  self._buy_levels,
            "sell_levels": self._sell_levels
        }
        self._store.save(self._cash, self._capital, self._position, self._trades, extra_state=extra)

    def load_from_disk(self):
        state = self._store.load()
        if state:
            self._cash = state["cash"]
            self._capital = state["capital"]
            self._position = state["position"]
            self._trades = state.get("trades", [])
            
            extra = state.get("extra_state", {})
            self._grid_active = extra.get("grid_active", False)
            self._base_price  = extra.get("base_price", 0.0)
            self._grid_step   = extra.get("grid_step", 0.0)
            self._buy_levels  = extra.get("buy_levels", [])
            self._sell_levels = extra.get("sell_levels", [])
            
            print(f"[INFO] {self.bot_id} re-hydrated state from disk.")
            return True
        return False

    @property
    def bot_id(self) -> str:
        return f"rangebot_{self._symbol.lower().replace('-', '_')}"

    @property
    def supported_regimes(self) -> list:
        return ["RANGE", "RECOVERY"]

    def get_status(self) -> BotStatus:
        return BotStatus(
            bot_id            = self.bot_id,
            symbol            = self._symbol,
            capital_allocated = self._capital,
            capital_deployed  = self._position.cost_basis,
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

    def get_recent_trades(self, n=5) -> list:
        return self._trades[-n:] if self._trades else []

    def get_state_summary(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "equity": self._cash + self._position.qty * self._position.avg_entry if self._position.is_open else self._cash,
            "realized_pnl": self._realized_pnl,
            "active_position": self._position.is_open,
            "grid_levels": len(self._buy_levels) + len(self._sell_levels),
        }

    def _reset(self, capital: float) -> None:
        self._cash         = float(capital)
        self._capital      = float(capital)
        self._position     = Position(symbol=self._symbol)
        self._trades       = []
        self._equity_curve = []
        self._cumulative   = 0.0
        self._grid_active  = False
        self._buy_levels   = []
        self._sell_levels  = []

    def run_backtest(self, df: pd.DataFrame, preset: dict, capital: float, preset_name: str, supervisor=None) -> tuple:
        self._reset(capital)
        p = preset
        fee_pct      = p["fee_pct"]
        base_qty     = p["base_qty"]
        grid_levels  = p["grid_levels"]
        grid_bps     = p["grid_step_bps"]
        psl_pct      = p.get("pos_stop_loss_pct", 0.05)

        for i in range(len(df)):
            row    = df.iloc[i]
            ts     = row["ts"]
            regime = str(row.get("regime5", "RANGE"))
            low    = float(row["low"])
            high   = float(row["high"])
            close  = float(row["close"])

            self._equity_curve.append(self._cash + self._position.qty * close)

            # Check catastrophic stop loss FIRST
            if self._position.is_open:
                unreal = (close - self._position.avg_entry) / self._position.avg_entry if self._position.avg_entry > 0 else 0
                if unreal < -psl_pct:
                    self._sell_all(i, df, close, "pos_stop_loss", fee_pct, supervisor=supervisor)
                    self._grid_active = False
                    self._buy_levels = []
                    self._sell_levels = []
                    continue

            # Regime transitions
            if regime not in self.supported_regimes:
                if self._grid_active:
                    if self._position.is_open:
                        self._sell_all(i, df, close, "regime_exit", fee_pct, supervisor=supervisor)
                    self._grid_active = False
                    self._buy_levels = []
                    self._sell_levels = []
                continue

            # ── DRIFT FILTER ──────────────────────────────────────────────────
            # Even in RANGE regime, if trend_strength is too high, it indicates
            # a persistent drift that can blow out a grid.
            drift_thresh = p.get("drift_threshold", 0.0015)
            trend_str    = float(row.get("trend_strength", 0.0))
            if abs(trend_str) > drift_thresh:
                if self._grid_active:
                    if self._position.is_open:
                        self._sell_all(i, df, close, "drift_exit", fee_pct, supervisor=supervisor)
                    self._grid_active = False
                    self._buy_levels = []
                    self._sell_levels = []
                continue
            # ──────────────────────────────────────────────────────────────────

            # Initialize or re-anchor grid
            if not self._grid_active:
                self._grid_active = True
                self._base_price  = close
                self._grid_step   = close * (grid_bps / 10000.0)
                self._buy_levels  = [self._base_price - self._grid_step * j for j in range(1, grid_levels + 1)]
                self._sell_levels = []
            elif not self._sell_levels and self._buy_levels:
                # We hold no inventory. If price drifts UP and leaves our buy ladder behind, trail the grid up.
                if close > self._buy_levels[0] + self._grid_step * 2:
                    self._base_price  = close
                    self._buy_levels  = [self._base_price - self._grid_step * j for j in range(1, grid_levels + 1)]

            # Simulate intra-bar limit order fills
            # 1. Check buys
            for bl in sorted(self._buy_levels, reverse=True): # highest buys hit first
                if low <= bl:
                    if self._buy_lot(i, df, bl, "grid_buy", base_qty, fee_pct, supervisor=supervisor):
                        self._buy_levels.remove(bl)
                        self._sell_levels.append(bl + self._grid_step)
            
            # 2. Check sells
            for sl in sorted(self._sell_levels): # lowest sells hit first
                if high >= sl:
                    if self._sell_lot(i, df, sl, "grid_sell", fee_pct, supervisor=supervisor):
                        self._sell_levels.remove(sl)
                        self._buy_levels.append(sl - self._grid_step)

        if self._position.is_open:
            self._sell_all(len(df)-1, df, float(df.iloc[-1]["close"]), "end_of_period", fee_pct)

        return self._build_result(capital, preset_name)

    def _buy_lot(self, i, df, fill_price, reason, base_qty, fee_pct, supervisor=None) -> bool:
        row = df.iloc[i]
        bv  = base_qty * fill_price
        
        # Risk Check (v32)
        if supervisor:
            allowed_bv = supervisor.request_allocation(self.bot_id, bv)
            if allowed_bv < bv:
                if allowed_bv < 1.0: # Too small to trade
                    return False
                # Adjust base_qty to match allowed budget
                base_qty = allowed_bv / fill_price
                bv  = base_qty * fill_price

        if bv > self._cash:
            return False # insufficient cash for this grid level

        fee = bv * fee_pct
        self._cash -= (bv + fee)
        
        # update position
        old_cost = self._position.qty * self._position.avg_entry
        self._position.qty += base_qty
        self._position.avg_entry = (old_cost + bv) / self._position.qty
        
        lot = Lot(qty=base_qty, price=fill_price, fee=fee, ts=row["ts"], row_idx=len(self._trades))
        self._position.lots.append(lot)
        
        # Update Supervisor (v32)
        if supervisor:
            supervisor.update_bot_status_realtime(self.bot_id, self._position.cost_basis)

        self._trades.append({
            "ts": row["ts"], "side": "BUY", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": base_qty, "fee": fee,
            "rsi": float(row.get("rsi", float("nan"))),
            "bb_lower": float(row.get("bb_lower", float("nan"))),
            "bw_pct":   float(row.get("bw_pct",   float("nan"))),
            "bars_held": float("nan"), "exit_price": float("nan"),
        })
        self.save_to_disk()
        return True

    def _sell_lot(self, i, df, fill_price, reason, fee_pct, supervisor=None) -> bool:
        if not self._position.lots:
            return False
        
        row = df.iloc[i]
        lot = self._position.lots.pop(0) # FIFO
        
        sell_val = lot.qty * fill_price
        sell_fee = sell_val * fee_pct
        pnl = sell_val - sell_fee - (lot.qty * lot.price + lot.fee)
        
        self._cash += (sell_val - sell_fee)
        self._position.qty -= lot.qty
        if self._position.qty < 1e-9:
            self._position.avg_entry = 0.0
            self._position.qty = 0.0
            
        self._cumulative += pnl
        self._realized_pnl += pnl
        self._trade_count += 1
        
        # Update Supervisor (v32)
        if supervisor:
            supervisor.update_bot_status_realtime(self.bot_id, self._position.cost_basis)

        bh = i - lot.row_idx # approx bars held since that buy
        buy_row = self._trades[lot.row_idx]
        buy_row.update({
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": fill_price,
        })
        
        self._trades.append({
            "ts": row["ts"], "side": "SELL", "reason": reason,
            "regime_h1": str(row.get("regime_h1", "")),
            "price": fill_price, "qty": lot.qty, "fee": sell_fee,
            "rsi": float(row.get("rsi", float("nan"))),
            "bb_lower": float("nan"), "bw_pct": float("nan"),
            "pnl": pnl, "pnl_after_fees": pnl,
            "win": 1.0 if pnl > 0 else 0.0,
            "bars_held": bh, "exit_price": float("nan"),
        })
        return True

    def _sell_all(self, i, df, fill_price, reason, fee_pct, supervisor=None):
        while self._position.lots:
            self._sell_lot(i, df, fill_price, reason, fee_pct, supervisor=supervisor)

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
        
        sells  = tdf[tdf["side"] == "SELL"]
        real   = sells[~sells["reason"].isin(["end_of_period", "regime_exit"])]
        wins   = real[real["pnl_after_fees"] > 0]
        
        psl_r  = sells[sells["reason"] == "pos_stop_loss"]
        tgt_r  = sells[sells["reason"] == "grid_sell"]
        tst_r  = sells[sells["reason"] == "time_stop"] # shouldn't exist anymore but keep for compat
        
        rd_r   = tdf[tdf["reason"] == "grid_buy"]
        
        return tdf, {
            "preset":          preset_name,
            "trades":          len(real),
            "win_rate":        len(wins) / len(real) * 100 if len(real) > 0 else 0,
            "realized_pnl":    tdf["pnl_after_fees"].sum(),
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
            "time_stop_pct":   0.0,
            "range_dip_fires": len(rd_r),
            "lv_scalp_fires":  0,
            "eop_pnl":         float(sells[sells["reason"].isin(["end_of_period", "regime_exit"])]["pnl_after_fees"].sum()) if len(sells) > 0 else 0.0,
            "avg_bars_held":   float(real["bars_held"].mean()) if len(real) > 0 else 0,
        }
