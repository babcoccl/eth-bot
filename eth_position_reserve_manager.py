#!/usr/bin/env python3
"""
eth_position_reserve_manager.py  —  Position Reserve Manager
=============================================================
Manages tranche exits from CrashAccumulator's accumulated ETH position.
Called by MacroSupervisor on each price tick AFTER the CRASH window closes.

Philosophy: "Bird in hand is worth two in the bush."
  - Don't hold for a single large exit. Ladder out in tranches.
  - Lock gains aggressively at +10%. Let the position run to +15% and +20%
    only if regime and momentum still support it.
  - Override conditions exit remaining position IMMEDIATELY rather than
    waiting for the next tranche level.

Tranche schedule (applied to avg_entry cost basis):
  T1 : price > avg_entry * 1.10  -> sell 33% of remaining qty  (guaranteed +10%)
  T2 : price > avg_entry * 1.15  -> sell 33% of remaining qty  (consolidate +15%)
  T3 : price > avg_entry * 1.20  -> sell remaining qty          (full +20% target)

Override conditions (sell ALL remaining immediately):
  OVR-1 : Regime shifts CRASH while unrealized_pct > +5%
           (crash incoming, protect any gain)
  OVR-2 : Regime is CORRECTION and unrealized_pct > +8%
           (corrections eat gains quickly — don't wait for T2)
  OVR-3 : Days since accumulation_end > 90 and unrealized_pct > +7%
           (time-value pressure: free capital for next crash)
  OVR-4 : unrealized_pct crosses below +3% after having been above +10%
           (reversion detected — take what's left)

Phase 4 (LLM Orchestrator) integration:
  The LLM Orchestrator can override any tranche decision with:
    override_hold(reason)   -> skip T1 sell this tick (max 3 consecutive)
    override_sell(reason)   -> sell entire position now
  This allows the Orchestrator to say:
    "RSI=72, BULL regime momentum strong — skip T1, let it run to +15%"
  or:
    "Macro news incoming, exit now at +12% rather than wait for T2"

Usage:
  mgr = PositionReserveManager(avg_entry=2384.0, qty=0.148, fee_pct=0.00065)
  mgr.on_tick(price=2623.0, regime="RECOVERY", days_held=12)
  if mgr.should_sell:
      qty_to_sell = mgr.sell_qty
      reason      = mgr.sell_reason
      mgr.record_sell(qty_to_sell, price=2623.0)
"""

from dataclasses import dataclass, field
from typing import Optional


TRANCHE_SCHEDULE = [
    {"level": 0.10, "pct_of_remaining": 0.33, "name": "T1"},
    {"level": 0.15, "pct_of_remaining": 0.33, "name": "T2"},
    {"level": 0.20, "pct_of_remaining": 1.00, "name": "T3"},  # sell all remaining
]

OVERRIDES = {
    "OVR-1": {"desc": "Regime->CRASH while unrealized>+5%",       "threshold": 0.05},
    "OVR-2": {"desc": "Regime=CORRECTION and unrealized>+8%",     "threshold": 0.08},
    "OVR-3": {"desc": "Days>90 held and unrealized>+7%",          "threshold": 0.07},
    "OVR-4": {"desc": "Reversion: was>+10% now<+3%",              "threshold": 0.03},
}


@dataclass
class SellSignal:
    should_sell: bool = False
    qty: float = 0.0
    reason: str = ""
    tranche: str = ""
    unrealized_pct: float = 0.0
    unrealized_usd: float = 0.0


class PositionReserveManager:
    """
    Laddered exit manager for CrashAccumulator's accumulated position.
    Stateful: tracks which tranches have fired, peak unrealized, days held.
    """

    def __init__(self, avg_entry: float, qty: float, fee_pct: float = 0.00065):
        self.avg_entry       = avg_entry
        self.remaining_qty   = qty
        self.initial_qty     = qty
        self.fee_pct         = fee_pct

        self._tranches_fired  = set()   # {"T1", "T2", "T3"}
        self._peak_unrealized = 0.0     # for OVR-4 reversion detection
        self._days_held       = 0
        self._prev_regime     = None
        self._llm_hold_count  = 0       # consecutive LLM override holds
        self._sells           = []      # history

    @property
    def is_done(self) -> bool:
        return self.remaining_qty <= 0.0001

    @property
    def realized_pnl(self) -> float:
        return sum(s["pnl"] for s in self._sells)

    def on_tick(self, price: float, regime: str, days_held: int) -> SellSignal:
        """
        Evaluate tranche and override conditions at current price + regime.
        Returns SellSignal with should_sell=True if action needed.
        """
        if self.is_done:
            return SellSignal()

        self._days_held = days_held
        unrealized_pct = (price - self.avg_entry) / self.avg_entry
        unrealized_usd = (price - self.avg_entry) * self.remaining_qty
        self._peak_unrealized = max(self._peak_unrealized, unrealized_pct)

        # ── OVERRIDE checks (priority over tranche schedule) ─────────────────
        # OVR-1: regime just flipped to CRASH
        if regime == "CRASH" and self._prev_regime != "CRASH":
            if unrealized_pct > OVERRIDES["OVR-1"]["threshold"]:
                self._prev_regime = regime
                return SellSignal(
                    should_sell=True, qty=self.remaining_qty, reason="OVR-1",
                    tranche="OVR", unrealized_pct=unrealized_pct,
                    unrealized_usd=unrealized_usd
                )

        # OVR-2: correction with meaningful gain — bird in hand
        if regime == "CORRECTION" and unrealized_pct > OVERRIDES["OVR-2"]["threshold"]:
            return SellSignal(
                should_sell=True, qty=self.remaining_qty, reason="OVR-2",
                tranche="OVR", unrealized_pct=unrealized_pct,
                unrealized_usd=unrealized_usd
            )

        # OVR-3: time pressure — held > 90 days with decent gain
        if days_held > 90 and unrealized_pct > OVERRIDES["OVR-3"]["threshold"]:
            return SellSignal(
                should_sell=True, qty=self.remaining_qty, reason="OVR-3",
                tranche="OVR", unrealized_pct=unrealized_pct,
                unrealized_usd=unrealized_usd
            )

        # OVR-4: reversion — was above +10%, now below +3%
        if (self._peak_unrealized > 0.10
                and unrealized_pct < OVERRIDES["OVR-4"]["threshold"]):
            if self.remaining_qty > 0:
                return SellSignal(
                    should_sell=True, qty=self.remaining_qty, reason="OVR-4",
                    tranche="OVR", unrealized_pct=unrealized_pct,
                    unrealized_usd=unrealized_usd
                )

        # ── TRANCHE schedule ──────────────────────────────────────────────────
        for tranche in TRANCHE_SCHEDULE:
            if tranche["name"] in self._tranches_fired:
                continue
            if unrealized_pct >= tranche["level"]:
                # LLM hold override (max 3 consecutive)
                if self._llm_hold_count < 3:
                    qty_to_sell = self.remaining_qty * tranche["pct_of_remaining"]
                    self._tranches_fired.add(tranche["name"])
                    self._prev_regime = regime
                    return SellSignal(
                        should_sell=True, qty=qty_to_sell,
                        reason=f"tranche_{tranche['name']}",
                        tranche=tranche["name"],
                        unrealized_pct=unrealized_pct,
                        unrealized_usd=unrealized_usd
                    )

        self._prev_regime = regime
        return SellSignal()

    def record_sell(self, qty: float, price: float) -> float:
        """Execute a sell. Returns realized PnL for this tranche."""
        if qty <= 0:
            return 0.0
        sell_val = qty * price
        fee      = sell_val * self.fee_pct
        pnl      = (price - self.avg_entry) * qty - fee
        self.remaining_qty -= qty
        self._sells.append({
            "qty": qty, "price": price, "fee": fee, "pnl": pnl,
            "unrealized_pct": (price - self.avg_entry) / self.avg_entry,
        })
        return pnl

    def llm_hold(self):
        """LLM Orchestrator: skip this tranche sell for one tick."""
        self._llm_hold_count += 1

    def llm_sell(self) -> SellSignal:
        """LLM Orchestrator: force immediate full exit."""
        price_est = 0.0
        return SellSignal(
            should_sell=True, qty=self.remaining_qty,
            reason="llm_override_sell", tranche="LLM"
        )

    def status(self, price: float) -> dict:
        unrealized_pct = (price - self.avg_entry) / self.avg_entry
        return {
            "avg_entry":       self.avg_entry,
            "remaining_qty":   self.remaining_qty,
            "pct_remaining":   self.remaining_qty / self.initial_qty,
            "tranches_fired":  list(self._tranches_fired),
            "unrealized_pct":  unrealized_pct,
            "unrealized_usd":  unrealized_pct * self.avg_entry * self.remaining_qty,
            "peak_unrealized": self._peak_unrealized,
            "realized_pnl":    self.realized_pnl,
            "days_held":       self._days_held,
        }
