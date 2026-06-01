from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.dividend_low_vol_etf import BacktestConfig, backtest, performance


def test_backtest_runs_on_synthetic_daily_data():
    dates = pd.bdate_range("2020-01-01", periods=820)
    close = pd.Series(range(820), index=dates, dtype=float) / 100 + 1
    df = pd.DataFrame({"close": close})

    result = backtest(df, BacktestConfig(price_percentile_window=60, ma_window=20))

    assert {"buyhold", "strategy", "position", "price_pct", "ma200"}.issubset(result.columns)
    assert result["strategy"].notna().all()
    assert result["position"].between(0.2, 1.6).all()

    metrics = performance(result["strategy"])
    assert set(metrics) == {"CAGR(%)", "MaxDD(%)", "Sharpe", "FinalNAV"}
