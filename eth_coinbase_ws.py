#!/usr/bin/env python3
"""
eth_coinbase_ws.py — Coinbase Advanced Trade WebSocket Manager
===============================================================
Maintains persistent WebSocket connections to Coinbase for:
  1. Market Data (public)  — real-time ticker prices
  2. User Channel (private) — order fill notifications

Architecture:
  - Each connection runs in its own asyncio task
  - Events are pushed into a shared asyncio.Queue
  - Auto-reconnects on disconnect with exponential backoff
  - Heartbeat subscription keeps connections alive
  - JWT tokens auto-refresh every 90 seconds (2-min expiry)

Usage:
  queue = asyncio.Queue()
  ws = CoinbaseWSManager(api_key, api_secret, queue)
  await ws.start(["ETH-USD"])
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("eth_ws")


# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickEvent:
    """Real-time price update from the market data stream."""
    product_id: str
    price:      float
    best_bid:   float
    best_ask:   float
    volume_24h: float
    ts:         datetime
    raw:        dict = field(default_factory=dict, repr=False)


@dataclass
class FillEvent:
    """Order fill notification from the user channel."""
    order_id:     str
    product_id:   str
    side:         str        # BUY or SELL
    fill_price:   float
    fill_qty:     float
    fee:          float
    client_order_id: str
    ts:           datetime
    raw:          dict = field(default_factory=dict, repr=False)


@dataclass
class OrderUpdateEvent:
    """Order status change (open, cancelled, etc.)."""
    order_id:     str
    product_id:   str
    status:       str        # OPEN, FILLED, CANCELLED, EXPIRED, FAILED
    side:         str
    client_order_id: str
    ts:           datetime
    raw:          dict = field(default_factory=dict, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CoinbaseWSManager:
    """
    Manages two WebSocket connections to Coinbase:
      - Market stream (public):  ticker prices
      - User stream (private):   fill/order notifications
    
    All events are pushed into a shared asyncio.Queue for the
    Live Engine to consume.
    """

    MARKET_WS_URL = "wss://advanced-trade-ws.coinbase.com"
    USER_WS_URL   = "wss://advanced-trade-ws-user.coinbase.com"

    def __init__(
        self,
        api_key:    str,
        api_secret: str,
        event_queue: asyncio.Queue,
        paper_mode:  bool = False,
    ):
        self._api_key    = api_key
        self._api_secret = api_secret
        self._queue      = event_queue
        self._paper_mode = paper_mode
        self._running    = False
        self._tasks: List[asyncio.Task] = []
        self._product_ids: List[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self, product_ids: List[str]) -> None:
        """Start both WebSocket connections."""
        self._product_ids = product_ids
        self._running = True

        logger.info(f"Starting WebSocket manager for {product_ids}")

        # Always start market data stream
        self._tasks.append(
            asyncio.create_task(self._run_market_stream(), name="market_ws")
        )

        # Only start user stream if we have credentials and not in paper mode
        if self._api_key and self._api_secret and not self._paper_mode:
            self._tasks.append(
                asyncio.create_task(self._run_user_stream(), name="user_ws")
            )
        else:
            logger.info("User stream disabled (paper mode or no credentials)")

    async def stop(self) -> None:
        """Gracefully shutdown all WebSocket connections."""
        logger.info("Shutting down WebSocket manager...")
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("WebSocket manager stopped.")

    # ── Market Data Stream (Public) ───────────────────────────────────────

    async def _run_market_stream(self) -> None:
        """Persistent connection to the public market data stream."""
        backoff = 1
        while self._running:
            try:
                import websockets
                async with websockets.connect(
                    self.MARKET_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("Market WebSocket connected.")
                    backoff = 1  # Reset on successful connect

                    # Subscribe to ticker + heartbeats
                    sub_msg = {
                        "type": "subscribe",
                        "product_ids": self._product_ids,
                        "channel": "ticker",
                    }
                    await ws.send(json.dumps(sub_msg))

                    hb_msg = {
                        "type": "subscribe",
                        "product_ids": self._product_ids,
                        "channel": "heartbeats",
                    }
                    await ws.send(json.dumps(hb_msg))

                    logger.info(f"Subscribed to ticker for {self._product_ids}")

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_msg)
                            await self._handle_market_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from market stream: {raw_msg[:100]}")

            except asyncio.CancelledError:
                logger.info("Market stream task cancelled.")
                return
            except Exception as e:
                if not self._running:
                    return
                logger.error(f"Market stream error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _handle_market_message(self, data: dict) -> None:
        """Parse market data messages and emit TickEvents."""
        channel = data.get("channel")
        if channel != "ticker":
            return

        events = data.get("events", [])
        for event in events:
            tickers = event.get("tickers", [])
            for ticker in tickers:
                try:
                    tick = TickEvent(
                        product_id = ticker.get("product_id", ""),
                        price      = float(ticker.get("price", 0)),
                        best_bid   = float(ticker.get("best_bid", 0)),
                        best_ask   = float(ticker.get("best_ask", 0)),
                        volume_24h = float(ticker.get("volume_24_h", 0)),
                        ts         = datetime.now(timezone.utc),
                        raw        = ticker,
                    )
                    await self._queue.put(tick)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse ticker: {e}")

    # ── User Stream (Private / Authenticated) ─────────────────────────────

    async def _run_user_stream(self) -> None:
        """Persistent connection to the authenticated user stream."""
        backoff = 1
        while self._running:
            try:
                jwt_token = self._generate_jwt()
                import websockets
                async with websockets.connect(
                    self.USER_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("User WebSocket connected.")
                    backoff = 1

                    # Subscribe to user channel with JWT auth
                    sub_msg = {
                        "type": "subscribe",
                        "product_ids": self._product_ids,
                        "channel": "user",
                        "jwt": jwt_token,
                    }
                    await ws.send(json.dumps(sub_msg))

                    # Also subscribe to heartbeats for keepalive
                    hb_msg = {
                        "type": "subscribe",
                        "product_ids": self._product_ids,
                        "channel": "heartbeats",
                        "jwt": jwt_token,
                    }
                    await ws.send(json.dumps(hb_msg))

                    logger.info("Subscribed to user channel.")

                    # JWT refresh task — re-subscribe every 90s
                    refresh_task = asyncio.create_task(
                        self._jwt_refresh_loop(ws),
                        name="jwt_refresh"
                    )

                    try:
                        async for raw_msg in ws:
                            if not self._running:
                                break
                            try:
                                data = json.loads(raw_msg)
                                await self._handle_user_message(data)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from user stream")
                    finally:
                        refresh_task.cancel()

            except asyncio.CancelledError:
                logger.info("User stream task cancelled.")
                return
            except Exception as e:
                if not self._running:
                    return
                logger.error(f"User stream error: {e}. Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _jwt_refresh_loop(self, ws) -> None:
        """Re-subscribe with a fresh JWT every 90 seconds."""
        while self._running:
            await asyncio.sleep(90)
            try:
                jwt_token = self._generate_jwt()
                sub_msg = {
                    "type": "subscribe",
                    "product_ids": self._product_ids,
                    "channel": "user",
                    "jwt": jwt_token,
                }
                await ws.send(json.dumps(sub_msg))
                logger.debug("JWT token refreshed.")
            except Exception as e:
                logger.warning(f"JWT refresh failed: {e}")

    async def _handle_user_message(self, data: dict) -> None:
        """Parse user channel messages and emit FillEvent / OrderUpdateEvent."""
        channel = data.get("channel")
        if channel != "user":
            return

        events = data.get("events", [])
        for event in events:
            event_type = event.get("type", "")
            orders = event.get("orders", [])

            for order in orders:
                status = order.get("status", "").upper()
                order_id = order.get("order_id", "")
                product_id = order.get("product_id", "")
                side = order.get("order_side", "").upper()
                client_order_id = order.get("client_order_id", "")

                # Emit fill event for completed orders
                if status == "FILLED":
                    try:
                        fill = FillEvent(
                            order_id        = order_id,
                            product_id      = product_id,
                            side            = "BUY" if "BUY" in side else "SELL",
                            fill_price      = float(order.get("average_filled_price", 0)),
                            fill_qty        = float(order.get("filled_size", 0)),
                            fee             = float(order.get("total_fees", 0)),
                            client_order_id = client_order_id,
                            ts              = datetime.now(timezone.utc),
                            raw             = order,
                        )
                        await self._queue.put(fill)
                        logger.info(f"FILL: {fill.side} {fill.fill_qty} {fill.product_id} @ ${fill.fill_price:.2f}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse fill: {e}")
                else:
                    # Emit generic order update
                    update = OrderUpdateEvent(
                        order_id        = order_id,
                        product_id      = product_id,
                        status          = status,
                        side            = "BUY" if "BUY" in side else "SELL",
                        client_order_id = client_order_id,
                        ts              = datetime.now(timezone.utc),
                        raw             = order,
                    )
                    await self._queue.put(update)

    # ── JWT Generation ────────────────────────────────────────────────────

    def _generate_jwt(self) -> str:
        """
        Generate a JWT token for Coinbase Advanced Trade WebSocket auth.
        Uses the official coinbase-advanced-py SDK if available,
        otherwise falls back to manual JWT construction.
        """
        try:
            from coinbase import jwt_generator
            return jwt_generator.build_ws_jwt(self._api_key, self._api_secret)
        except ImportError:
            pass

        # Manual JWT fallback using PyJWT
        try:
            import jwt as pyjwt
            now = int(time.time())
            payload = {
                "sub": self._api_key,
                "iss": "coinbase-cloud",
                "aud": ["cdp_service"],
                "nbf": now,
                "exp": now + 120,
            }
            # The API secret from Coinbase is an EC private key in PEM format
            token = pyjwt.encode(payload, self._api_secret, algorithm="ES256")
            return token
        except ImportError:
            logger.error(
                "Neither coinbase-advanced-py nor PyJWT is installed. "
                "Run: pip install coinbase-advanced-py"
            )
            raise RuntimeError("No JWT library available")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

async def _test_market_stream():
    """Quick test: connect to public market data and print 10 ticks."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    queue = asyncio.Queue()
    ws = CoinbaseWSManager("", "", queue, paper_mode=True)
    await ws.start(["ETH-USD"])

    count = 0
    while count < 10:
        event = await asyncio.wait_for(queue.get(), timeout=30)
        if isinstance(event, TickEvent):
            print(f"  TICK: {event.product_id} ${event.price:.2f} "
                  f"bid={event.best_bid:.2f} ask={event.best_ask:.2f}")
            count += 1

    await ws.stop()
    print(f"\nReceived {count} ticks. WebSocket manager working correctly.")


if __name__ == "__main__":
    asyncio.run(_test_market_stream())
