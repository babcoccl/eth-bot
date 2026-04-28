#!/usr/bin/env python3
"""
eth_coinbase_executor.py — Order Execution Layer
==================================================
Wraps the Coinbase Advanced Trade REST API for order placement.
Supports both live and paper trading modes.

Paper Trading Mode:
  When PAPER_TRADING=true, orders are NOT sent to Coinbase.
  Instead, they are logged to a local journal file and filled
  immediately at the current market price. This allows the full
  bot fleet to run against live market data without risking capital.

Live Mode:
  Orders are placed via the official coinbase-advanced-py SDK.
  Fill confirmation comes through the WebSocket user channel,
  NOT by polling this module.

Usage:
  executor = OrderExecutor(api_key, api_secret, paper_mode=True)
  result = executor.buy_market("ETH-USD", quote_size=100.0, current_price=2500.0)
"""

from __future__ import annotations
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("eth_executor")

# ── Journal for paper trades ──────────────────────────────────────────────────
_JOURNAL_DIR = Path(__file__).parent / ".paper_trades"
_JOURNAL_DIR.mkdir(exist_ok=True)


@dataclass
class OrderResult:
    """Standardized order result returned by the executor."""
    success:         bool
    order_id:        str
    client_order_id: str
    product_id:      str
    side:            str        # BUY or SELL
    order_type:      str        # MARKET or LIMIT
    requested_size:  float      # quote_size for buy, base_size for sell
    fill_price:      float      # Actual or simulated fill price
    fill_qty:        float      # Actual or simulated fill quantity
    fee:             float      # Estimated fee
    paper:           bool       # True if this was a paper trade
    error:           str = ""   # Error message if success=False
    ts:              str = ""   # ISO timestamp


class OrderExecutor:
    """
    Unified order execution interface.
    
    In paper mode:  simulates fills at current market price.
    In live mode:   places orders via Coinbase Advanced Trade API.
    """

    # Coinbase Advanced Trade fee tier (maker/taker)
    # Adjust based on your actual fee tier
    DEFAULT_FEE_PCT = 0.006  # 0.6% taker fee (conservative estimate)

    def __init__(
        self,
        api_key:    str = "",
        api_secret: str = "",
        paper_mode: bool = True,
        fee_pct:    float = None,
    ):
        self._api_key    = api_key
        self._api_secret = api_secret
        self._paper_mode = paper_mode
        self._fee_pct    = fee_pct or self.DEFAULT_FEE_PCT
        self._rest_client = None
        self._journal_path = _JOURNAL_DIR / f"journal_{datetime.now().strftime('%Y%m%d')}.jsonl"

        if not paper_mode:
            self._init_rest_client()

        mode_str = "PAPER" if paper_mode else "LIVE"
        logger.info(f"OrderExecutor initialized in {mode_str} mode.")

    def _init_rest_client(self) -> None:
        """Initialize the official Coinbase REST client."""
        try:
            from coinbase.rest import RESTClient
            self._rest_client = RESTClient(
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
            logger.info("Coinbase REST client initialized.")
        except ImportError:
            logger.error(
                "coinbase-advanced-py not installed. "
                "Run: pip install coinbase-advanced-py"
            )
            raise

    # ── Market Orders ─────────────────────────────────────────────────────

    def buy_market(
        self,
        product_id:    str,
        quote_size:    float,
        current_price: float,
    ) -> OrderResult:
        """
        Place a market BUY order.
        
        Args:
            product_id:    e.g. "ETH-USD"
            quote_size:    amount in USD to spend
            current_price: latest known price (for paper fills)
        """
        client_order_id = str(uuid.uuid4())

        if self._paper_mode:
            return self._paper_fill(
                client_order_id, product_id, "BUY", "MARKET",
                quote_size, current_price,
            )

        # Live order
        try:
            order = self._rest_client.market_order_buy(
                client_order_id=client_order_id,
                product_id=product_id,
                quote_size=str(round(quote_size, 2)),
            )
            return self._parse_order_response(order, client_order_id, product_id, "BUY", quote_size)
        except Exception as e:
            logger.error(f"Live BUY order failed: {e}")
            return OrderResult(
                success=False, order_id="", client_order_id=client_order_id,
                product_id=product_id, side="BUY", order_type="MARKET",
                requested_size=quote_size, fill_price=0, fill_qty=0,
                fee=0, paper=False, error=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
            )

    def sell_market(
        self,
        product_id:    str,
        base_size:     float,
        current_price: float,
    ) -> OrderResult:
        """
        Place a market SELL order.
        
        Args:
            product_id:    e.g. "ETH-USD"
            base_size:     amount of base currency (ETH) to sell
            current_price: latest known price (for paper fills)
        """
        client_order_id = str(uuid.uuid4())

        if self._paper_mode:
            return self._paper_fill(
                client_order_id, product_id, "SELL", "MARKET",
                base_size, current_price,
            )

        # Live order
        try:
            order = self._rest_client.market_order_sell(
                client_order_id=client_order_id,
                product_id=product_id,
                base_size=str(round(base_size, 8)),
            )
            return self._parse_order_response(order, client_order_id, product_id, "SELL", base_size)
        except Exception as e:
            logger.error(f"Live SELL order failed: {e}")
            return OrderResult(
                success=False, order_id="", client_order_id=client_order_id,
                product_id=product_id, side="SELL", order_type="MARKET",
                requested_size=base_size, fill_price=0, fill_qty=0,
                fee=0, paper=False, error=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
            )

    # ── Limit Orders ──────────────────────────────────────────────────────

    def buy_limit(
        self,
        product_id:    str,
        base_size:     float,
        limit_price:   float,
    ) -> OrderResult:
        """Place a limit BUY order (GTC — Good Till Cancelled)."""
        client_order_id = str(uuid.uuid4())

        if self._paper_mode:
            # Paper limit orders fill immediately at the limit price
            return self._paper_fill(
                client_order_id, product_id, "BUY", "LIMIT",
                base_size * limit_price, limit_price,
            )

        try:
            order = self._rest_client.limit_order_gtc_buy(
                client_order_id=client_order_id,
                product_id=product_id,
                base_size=str(round(base_size, 8)),
                limit_price=str(round(limit_price, 2)),
            )
            return self._parse_order_response(order, client_order_id, product_id, "BUY", base_size * limit_price)
        except Exception as e:
            logger.error(f"Live LIMIT BUY failed: {e}")
            return OrderResult(
                success=False, order_id="", client_order_id=client_order_id,
                product_id=product_id, side="BUY", order_type="LIMIT",
                requested_size=base_size, fill_price=0, fill_qty=0,
                fee=0, paper=False, error=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
            )

    def sell_limit(
        self,
        product_id:    str,
        base_size:     float,
        limit_price:   float,
    ) -> OrderResult:
        """Place a limit SELL order (GTC — Good Till Cancelled)."""
        client_order_id = str(uuid.uuid4())

        if self._paper_mode:
            return self._paper_fill(
                client_order_id, product_id, "SELL", "LIMIT",
                base_size, limit_price,
            )

        try:
            order = self._rest_client.limit_order_gtc_sell(
                client_order_id=client_order_id,
                product_id=product_id,
                base_size=str(round(base_size, 8)),
                limit_price=str(round(limit_price, 2)),
            )
            return self._parse_order_response(order, client_order_id, product_id, "SELL", base_size)
        except Exception as e:
            logger.error(f"Live LIMIT SELL failed: {e}")
            return OrderResult(
                success=False, order_id="", client_order_id=client_order_id,
                product_id=product_id, side="SELL", order_type="LIMIT",
                requested_size=base_size, fill_price=0, fill_qty=0,
                fee=0, paper=False, error=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
            )

    # ── Portfolio Sync ────────────────────────────────────────────────────

    def get_balance(self, currency: str = "USD") -> float:
        """Get available balance for a currency."""
        if self._paper_mode:
            logger.info(f"Paper mode: balance query for {currency} (returning 0 — use local state)")
            return 0.0

        try:
            accounts = self._rest_client.get_accounts()
            for acct in accounts.get("accounts", []):
                if acct.get("currency") == currency:
                    return float(acct.get("available_balance", {}).get("value", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")
            return 0.0

    # ── Paper Trading ─────────────────────────────────────────────────────

    def _paper_fill(
        self,
        client_order_id: str,
        product_id:      str,
        side:            str,
        order_type:      str,
        size:            float,
        price:           float,
    ) -> OrderResult:
        """Simulate an immediate fill at the given price."""
        paper_order_id = f"PAPER-{client_order_id[:8]}"

        if side == "BUY":
            fill_qty = size / price  # size is quote_size (USD)
            fee = size * self._fee_pct
        else:
            fill_qty = size          # size is base_size (ETH)
            fee = (size * price) * self._fee_pct

        result = OrderResult(
            success         = True,
            order_id        = paper_order_id,
            client_order_id = client_order_id,
            product_id      = product_id,
            side            = side,
            order_type      = order_type,
            requested_size  = size,
            fill_price      = price,
            fill_qty        = fill_qty,
            fee             = fee,
            paper           = True,
            ts              = datetime.now(timezone.utc).isoformat(),
        )

        # Write to journal
        self._write_journal(result)

        logger.info(
            f"PAPER {side} {fill_qty:.6f} {product_id} @ ${price:.2f} "
            f"(fee: ${fee:.2f}) [{paper_order_id}]"
        )
        return result

    def _write_journal(self, result: OrderResult) -> None:
        """Append a paper trade to the daily journal file."""
        try:
            entry = {
                "ts":              result.ts,
                "order_id":        result.order_id,
                "client_order_id": result.client_order_id,
                "product_id":      result.product_id,
                "side":            result.side,
                "order_type":      result.order_type,
                "fill_price":      result.fill_price,
                "fill_qty":        result.fill_qty,
                "fee":             result.fee,
                "requested_size":  result.requested_size,
            }
            with open(self._journal_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write paper journal: {e}")

    # ── Response Parsing ──────────────────────────────────────────────────

    def _parse_order_response(
        self, response, client_order_id, product_id, side, size
    ) -> OrderResult:
        """Parse the Coinbase SDK order response into an OrderResult."""
        try:
            # The SDK returns a dict-like object
            order_data = response if isinstance(response, dict) else response.__dict__
            
            order_id = order_data.get("order_id", "")
            success = order_data.get("success", True)
            error_msg = order_data.get("error_response", {}).get("message", "")

            return OrderResult(
                success         = bool(success) and not error_msg,
                order_id        = order_id,
                client_order_id = client_order_id,
                product_id      = product_id,
                side            = side,
                order_type      = "MARKET",
                requested_size  = size,
                fill_price      = 0,     # Actual fill comes via WebSocket
                fill_qty        = 0,     # Actual fill comes via WebSocket
                fee             = 0,     # Actual fee comes via WebSocket
                paper           = False,
                error           = error_msg,
                ts              = datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            return OrderResult(
                success=False, order_id="", client_order_id=client_order_id,
                product_id=product_id, side=side, order_type="MARKET",
                requested_size=size, fill_price=0, fill_qty=0,
                fee=0, paper=False, error=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
            )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    executor = OrderExecutor(paper_mode=True)

    # Simulate a buy
    result = executor.buy_market("ETH-USD", quote_size=100.0, current_price=2500.0)
    print(f"\nBUY result:  {result}")

    # Simulate a sell
    result = executor.sell_market("ETH-USD", base_size=0.04, current_price=2510.0)
    print(f"SELL result: {result}")

    print(f"\nJournal written to: {executor._journal_path}")
