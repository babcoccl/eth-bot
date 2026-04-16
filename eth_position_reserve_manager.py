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

Bull-class-aware sizing (v2)
-----------------------------
The MacroSupervisor v30 classifies each BULL entry as DEEP / MID / SHALLOW
based on the preceding cycle trough depth. This manager now accepts an
optional bull_class parameter and adjusts:
  - Tranche profit targets (DEEP entries swing wider; SHALLOW tighter)
  - Override thresholds (MID exits faster given 0% historical win rate)
  - Recommended alloc_pct (fraction of available capital to deploy)
  - Recommended stop_loss (absolute drawdown from entry-peak to hard exit)

Per-class defaults (aligned with backtest findings v30.2):
  DEEP    : alloc=70%, stop=-20%, T1=+12%, T2=+20%, T3=+30%
  MID     : alloc=30%, stop=-12%, T1=+8%,  T2=+12%, T3=+18%  (caution)
  SHALLOW : alloc=45%, stop=-15%, T1=+10%, T2=+15%, T3=+20%  (stop raised from -10%)
  UNKNOWN : alloc=40%, stop=-15%, T1=+10%, T2=+15%, T3=+20%  (safe default)

Tranche schedule (applied to avg_entry cost basis):
  T1 : price > avg_entry * (1 + t1_target)  -> sell 33% of remaining qty
  T2 : price > avg_entry * (1 + t2_target)  -> sell 33% of remaining qty
  T3 : price > avg_entry * (1 + t3_target)  -> sell remaining qty

Override conditions (sell ALL remaining immediately):
  OVR-1 : Regime shifts CRASH while unrealized_pct > ovr1_threshold
  OVR-2 : Regime is CORRECTION and unrealized_pct > ovr2_threshold
  OVR-3 : Days since accumulation_end > 90 and unrealized_pct > ovr3_threshold
  OVR-4 : unrealized_pct crosses below ovr4_threshold after having been above T1

Phase 4 (LLM Orchestrator) integration:
  The LLM Orchestrator can override any tranche decision with:
    override_hold(reason)   -> skip T1 sell this tick (max 3 consecutive)
    override_sell(reason)   -> sell entire position now

Usage:
  mgr = PositionReserveManager(
      avg_entry=2384.0, qty=0.148, fee_pct=0.00065, bull_class="DEEP"
  )
  mgr.on_tick(price=2623.0, regime="RECOVERY", days_held=12)
  if mgr.should_sell:
      qty_to_sell = mgr.sell_qty
      reason      = mgr.sell_reason
      mgr.record_sell(qty_to_sell, price=2623.0)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Bull-class configuration
# ---------------------------------------------------------------------------

@dataclass
class BullClassConfig:
    """Per-class sizing and exit parameters."""
    bull_class:     str
    alloc_pct:      float   # fraction of available capital to deploy (0-1)
    stop_loss:      float   # hard stop from entry-peak, e.g. -0.20 = -20%
    t1_target:      float   # T1 tranche profit target, e.g. 0.12 = +12%
    t2_target:      float   # T2 tranche profit target
    t3_target:      float   # T3 tranche profit target (sell all remaining)
    ovr1_threshold: float   # OVR-1 (CRASH regime) minimum gain to trigger
    ovr2_threshold: float   # OVR-2 (CORRECTION regime) minimum gain to trigger
    ovr3_threshold: float   # OVR-3 (time pressure >90 days) minimum gain
    ovr4_threshold: float   # OVR-4 (reversion) floor after peak


BULL_CLASS_CONFIGS: Dict[str, BullClassConfig] = {
    # DEEP: Large recovery swings expected. Deploy more capital, wider stops,
    # higher tranche targets. Historical: 50% win rate, +25.9% total.
    "DEEP": BullClassConfig(
        bull_class="DEEP",
        alloc_pct=0.70,
        stop_loss=-0.20,
        t1_target=0.12,
        t2_target=0.20,
        t3_target=0.30,
        ovr1_threshold=0.06,
        ovr2_threshold=0.10,
        ovr3_threshold=0.08,
        ovr4_threshold=0.04,
    ),
    # MID: Historically weak (0% win rate, 2 trades). Deploy less, exit faster.
    # Tight stop, aggressive OVR thresholds, lower tranche targets.
    "MID": BullClassConfig(
        bull_class="MID",
        alloc_pct=0.30,
        stop_loss=-0.12,
        t1_target=0.08,
        t2_target=0.12,
        t3_target=0.18,
        ovr1_threshold=0.04,
        ovr2_threshold=0.06,
        ovr3_threshold=0.05,
        ovr4_threshold=0.02,
    ),
    # SHALLOW: Momentum trades. Moderate size, standard stop, standard targets.
    # Historical: 56% win rate, +118.6% total (bulk of PnL).
    # stop_loss raised from -0.10 to -0.15 (v30.2): tighter stop was cutting
    # winners that needed -10% to -14% drawdown room before recovering.
    "SHALLOW": BullClassConfig(
        bull_class="SHALLOW",
        alloc_pct=0.45,
        stop_loss=-0.15,
        t1_target=0.10,
        t2_target=0.15,
        t3_target=0.20,
        ovr1_threshold=0.05,
        ovr2_threshold=0.08,
        ovr3_threshold=0.07,
        ovr4_threshold=0.03,
    ),
    # UNKNOWN: Safe default when bull_class is not provided.
    "UNKNOWN": BullClassConfig(
        bull_class="UNKNOWN",
        alloc_pct=0.40,
        stop_loss=-0.15,
        t1_target=0.10,
        t2_target=0.15,
        t3_target=0.20,
        ovr1_threshold=0.05,
        ovr2_threshold=0.08,
        ovr3_threshold=0.07,
        ovr4_threshold=0.03,
    ),
}


# ---------------------------------------------------------------------------
# Signal and manager
# ---------------------------------------------------------------------------

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

    Pass bull_class="DEEP"/"MID"/"SHALLOW" to enable class-aware sizing.
    If omitted, falls back to "UNKNOWN" (original conservative defaults).
    """

    def __init__(
        self,
        avg_entry: float,
        qty: float,
        fee_pct: float = 0.00065,
        bull_class: Optional[str] = None,
    ):
        self.avg_entry     = avg_entry
        self.remaining_qty = qty
        self.initial_qty   = qty
        self.fee_pct       = fee_pct
        self.bull_class    = (bull_class or "UNKNOWN").upper()
        self.cfg: BullClassConfig = BULL_CLASS_CONFIGS.get(
            self.bull_class, BULL_CLASS_CONFIGS["UNKNOWN"]
        )

        # Build tranche schedule from config
        self._tranche_schedule = [
            {"level": self.cfg.t1_target, "pct_of_remaining": 0.33, "name": "T1"},
            {"level": self.cfg.t2_target, "pct_of_remaining": 0.33, "name": "T2"},
            {"level": self.cfg.t3_target, "pct_of_remaining": 1.00, "name": "T3"},
        ]

        self._tranches_fired  = set()
        self._peak_unrealized = 0.0
        self._days_held       = 0
        self._prev_regime     = None
        self._llm_hold_count  = 0
        self._sells           = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self.remaining_qty <= 0.0001

    @property
    def realized_pnl(self) -> float:
        return sum(s["pnl"] for s in self._sells)

    @property
    def recommended_stop_loss(self) -> float:
        """Hard stop from entry-peak recommended for this bull class."""
        return self.cfg.stop_loss

    @property
    def recommended_alloc_pct(self) -> float:
        """Fraction of available capital recommended for this bull class."""
        return self.cfg.alloc_pct

    # ------------------------------------------------------------------
    # Core tick evaluation
    # ------------------------------------------------------------------

    def on_tick(self, price: float, regime: str, days_held: int) -> SellSignal:
        """
        Evaluate tranche and override conditions at current price + regime.
        Returns SellSignal with should_sell=True if action needed.
        """
        if self.is_done:
            return SellSignal()

        self._days_held       = days_held
        unrealized_pct        = (price - self.avg_entry) / self.avg_entry
        unrealized_usd        = (price - self.avg_entry) * self.remaining_qty
        self._peak_unrealized = max(self._peak_unrealized, unrealized_pct)
        cfg                   = self.cfg

        # ── OVERRIDE checks (priority over tranche schedule) ─────────────────

        # OVR-1: regime just flipped to CRASH
        if regime == "CRASH" and self._prev_regime != "CRASH":
            if unrealized_pct > cfg.ovr1_threshold:
                self._prev_regime = regime
                return SellSignal(
                    should_sell=True, qty=self.remaining_qty, reason="OVR-1",
                    tranche="OVR", unrealized_pct=unrealized_pct,
                    unrealized_usd=unrealized_usd,
                )

        # OVR-2: correction with meaningful gain
        if regime == "CORRECTION" and unrealized_pct > cfg.ovr2_threshold:
            return SellSignal(
                should_sell=True, qty=self.remaining_qty, reason="OVR-2",
                tranche="OVR", unrealized_pct=unrealized_pct,
                unrealized_usd=unrealized_usd,
            )

        # OVR-3: time pressure — held > 90 days with decent gain
        if days_held > 90 and unrealized_pct > cfg.ovr3_threshold:
            return SellSignal(
                should_sell=True, qty=self.remaining_qty, reason="OVR-3",
                tranche="OVR", unrealized_pct=unrealized_pct,
                unrealized_usd=unrealized_usd,
            )

        # OVR-4: reversion — was above T1 target, now below ovr4_threshold
        if (self._peak_unrealized > cfg.t1_target
                and unrealized_pct < cfg.ovr4_threshold
                and self.remaining_qty > 0):
            return SellSignal(
                should_sell=True, qty=self.remaining_qty, reason="OVR-4",
                tranche="OVR", unrealized_pct=unrealized_pct,
                unrealized_usd=unrealized_usd,
            )

        # ── TRANCHE schedule ──────────────────────────────────────────────────
        for tranche in self._tranche_schedule:
            if tranche["name"] in self._tranches_fired:
                continue
            if unrealized_pct >= tranche["level"]:
                if self._llm_hold_count < 3:
                    qty_to_sell = self.remaining_qty * tranche["pct_of_remaining"]
                    self._tranches_fired.add(tranche["name"])
                    self._prev_regime = regime
                    return SellSignal(
                        should_sell=True, qty=qty_to_sell,
                        reason=f"tranche_{tranche['name']}",
                        tranche=tranche["name"],
                        unrealized_pct=unrealized_pct,
                        unrealized_usd=unrealized_usd,
                    )

        self._prev_regime = regime
        return SellSignal()

    # ------------------------------------------------------------------
    # Record keeping
    # ------------------------------------------------------------------

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
        return SellSignal(
            should_sell=True, qty=self.remaining_qty,
            reason="llm_override_sell", tranche="LLM",
        )

    def status(self, price: float) -> dict:
        unrealized_pct = (price - self.avg_entry) / self.avg_entry
        return {
            "bull_class":        self.bull_class,
            "avg_entry":         self.avg_entry,
            "remaining_qty":     self.remaining_qty,
            "pct_remaining":     self.remaining_qty / self.initial_qty,
            "tranches_fired":    list(self._tranches_fired),
            "unrealized_pct":    unrealized_pct,
            "unrealized_usd":    unrealized_pct * self.avg_entry * self.remaining_qty,
            "peak_unrealized":   self._peak_unrealized,
            "realized_pnl":      self.realized_pnl,
            "days_held":         self._days_held,
            "recommended_stop":  self.cfg.stop_loss,
            "recommended_alloc": self.cfg.alloc_pct,
        }
