#!/usr/bin/env python3
"""
src/eth_trading/utils/trading_utils.py  —  Shared trading utility functions
=========================================================================
"""

import pandas as pd
from typing import List

def find_support_levels(df_warm: pd.DataFrame, n_levels: int = 3) -> List[float]:
    """
    Volume-profile support identification from warmup period.
    Bins price range into 20 buckets, returns top-N volume-weighted price nodes.

    Parameters
    ----------
    df_warm : pd.DataFrame
        DataFrame containing historical OHLCV data for warmup.
    n_levels : int, default 3
        Number of top volume nodes to return as support levels.

    Returns
    -------
    List[float]
        List of prices corresponding to identified support levels.
    """
    if df_warm is None or len(df_warm) < 20:
        return []
    try:
        lo, hi = df_warm["low"].min(), df_warm["high"].max()
        if hi <= lo:
            return []
        buckets = 20
        bucket_size = (hi - lo) / buckets
        vol_profile = {}
        for _, row in df_warm.iterrows():
            mid    = (row["high"] + row["low"]) / 2
            vol    = row.get("volume", 1.0)
            bucket = int((mid - lo) / bucket_size)
            bucket = max(0, min(buckets - 1, bucket))
            price_node = lo + (bucket + 0.5) * bucket_size
            vol_profile[price_node] = vol_profile.get(price_node, 0) + vol
        mid_price = (lo + hi) / 2
        support_candidates = {p: v for p, v in vol_profile.items() if p < mid_price}
        sorted_levels = sorted(support_candidates, key=lambda p: -support_candidates[p])
        return sorted_levels[:n_levels]
    except Exception:
        return []
