import pandas as pd
import numpy as np
import pytest
from eth_helpers import calc_bollinger

def test_calc_bollinger_happy_path():
    # Create a simple series
    close = pd.Series([float(i) for i in range(100, 120)])
    window = 10
    # The new implementation returns only bandwidth
    bw = calc_bollinger(close, window=window)

    # Check length
    assert len(bw) == 20

    # First (window-1) should be NaN
    assert bw.iloc[:window-1].isna().all()
    assert not bw.iloc[window-1:].isna().any()

    # Verify values at index 10
    # New implementation use ddof=0
    idx = 10
    subset = close.iloc[idx-window+1:idx+1]
    ma = subset.mean()
    std = subset.std(ddof=0)
    upper = ma + (std * 2.0)
    lower = ma - (std * 2.0)
    expected_bw = (upper - lower) / ma

    assert np.isclose(bw.iloc[idx], expected_bw)

def test_calc_bollinger_custom_params():
    np.random.seed(42)
    close = pd.Series(np.random.randn(50) + 100)
    window = 15
    std_mult = 1.5
    bw = calc_bollinger(close, window=window, std_mult=std_mult)

    assert np.isnan(bw.iloc[window-2])
    assert not np.isnan(bw.iloc[window-1])

    idx = 25
    subset = close.iloc[idx-window+1:idx+1]
    ma = subset.mean()
    std = subset.std(ddof=0)
    upper = ma + (std * std_mult)
    lower = ma - (std * std_mult)
    expected_bw = (upper - lower) / ma

    assert np.isclose(bw.iloc[idx], expected_bw)

def test_calc_bollinger_insufficient_data():
    close = pd.Series([100, 101, 102])
    bw = calc_bollinger(close, window=20)
    assert bw.isna().all()

def test_calc_bollinger_zero_division():
    # If ma is zero, bw should handle it (replace(0, np.nan))
    close = pd.Series([0.0] * 25)
    bw = calc_bollinger(close, window=20)

    assert (bw.iloc[19:].isna()).all()

def test_calc_bollinger_constant_value():
    close = pd.Series([100.0] * 25)
    bw = calc_bollinger(close, window=20)

    assert (bw.iloc[19:] == 0).all()
