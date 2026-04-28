#!/usr/bin/env python3
import json
import os
from typing import Any, Dict
from eth_bot_interface import Position, Lot

class BotStateStore:
    """
    Handles serialization and de-serialization of bot-specific state.
    Ensures that cash, capital, and open positions survive a script restart.
    """
    def __init__(self, bot_id: str, storage_dir: str = ".bot_state"):
        self.bot_id = bot_id
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self.state_file = os.path.join(storage_dir, f"{bot_id}.json")

    def save(self, cash: float, capital: float, position: Position, trades: list, extra_state: dict = None):
        """Serializes bot state to disk."""
        
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return super().default(obj)

        state = {
            "cash": cash,
            "capital": capital,
            "position": self._serialize_position(position),
            "trades": trades,
            "extra_state": extra_state or {},
            "last_updated": os.times()[4]
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=4, cls=CustomEncoder)

    def load(self) -> Dict[str, Any]:
        """Loads bot state from disk if available."""
        if not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            
            # De-serialize position
            pos_data = state["position"]
            pos = Position(
                symbol=pos_data["symbol"],
                side=pos_data["side"],
                qty=pos_data["qty"],
                avg_entry=pos_data["avg_entry"],
                peak_price=pos_data["peak_price"],
                dca_count=pos_data["dca_count"],
                entry_bar=pos_data["entry_bar"],
                entry_regime=pos_data["entry_regime"],
                bull_class=pos_data["bull_class"]
            )
            # De-serialize lots
            for l in pos_data["lots"]:
                pos.lots.append(Lot(
                    qty=l["qty"],
                    price=l["price"],
                    fee=l["fee"],
                    ts=l["ts"],
                    row_idx=l["row_idx"]
                ))
            
            state["position"] = pos
            return state
        except Exception as e:
            print(f"[ERROR] Failed to load bot state for {self.bot_id}: {e}")
            return None

    def _serialize_position(self, pos: Position) -> Dict[str, Any]:
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "qty": pos.qty,
            "avg_entry": pos.avg_entry,
            "peak_price": pos.peak_price,
            "dca_count": pos.dca_count,
            "entry_bar": pos.entry_bar,
            "entry_regime": pos.entry_regime,
            "bull_class": pos.bull_class,
            "lots": [
                {
                    "qty": l.qty,
                    "price": l.price,
                    "fee": l.fee,
                    "ts": str(l.ts),
                    "row_idx": l.row_idx
                } for l in pos.lots
            ]
        }
