from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.dividend_low_vol_etf import BacktestConfig, backtest, generate_demo_data, performance


def test_backtest_runs_on_demo_daily_data():
    bars = generate_demo_data(date(2020, 1, 1), date(2024, 1, 31))

    result = backtest(bars, BacktestConfig(price_percentile_window=60, ma_window=20))

    assert len(result) > 800
    assert {"buyhold", "strategy", "position", "price_pct", "ma200"}.issubset(result[0])
    assert all(row["strategy"] > 0 for row in result)
    assert all(0.2 <= row["position"] <= 1.6 for row in result)

    metrics = performance(result, "strategy")
    assert set(metrics) == {"CAGR(%)", "MaxDD(%)", "Sharpe", "FinalNAV"}
