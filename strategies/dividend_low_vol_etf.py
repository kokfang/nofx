"""Dividend low-volatility ETF monthly timing backtest.

This module provides a runnable MVP backtest for 512890 (红利低波 ETF) using
AkShare daily ETF prices.  It intentionally uses a trailing price percentile as
an easily available valuation proxy, so it can run without paid valuation data.

Examples:
    python strategies/dividend_low_vol_etf.py
    python strategies/dividend_low_vol_etf.py --start 2019-01-01 --no-plot
    python strategies/dividend_low_vol_etf.py --csv data/512890.csv --plot-file output/nav.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ETF_CODE = "512890"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the ETF timing strategy backtest."""

    symbol: str = ETF_CODE
    start: str | None = None
    end: str | None = None
    price_percentile_window: int = 750
    ma_window: int = 200
    initial_position: float = 1.0
    risk_free_daily_return: float = 0.0


def _normalize_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize AkShare/CSV data to a date-indexed close-price DataFrame."""

    if raw.empty:
        raise ValueError("No price data was returned or loaded.")

    column_candidates = {
        "date": ("date", "日期", "交易日期", "Date"),
        "close": ("close", "收盘", "收盘价", "Close"),
    }

    rename_map: dict[str, str] = {}
    for target, candidates in column_candidates.items():
        for column in candidates:
            if column in raw.columns:
                rename_map[column] = target
                break
        else:
            raise ValueError(
                f"Missing required {target!r} column. Available columns: {list(raw.columns)}"
            )

    df = raw.rename(columns=rename_map)[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.set_index("date")

    if df.empty:
        raise ValueError("No valid rows remained after normalizing price data.")
    if (df["close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    return df


def load_data(symbol: str = ETF_CODE, csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load ETF daily close prices from AkShare or a local CSV file.

    CSV mode is useful for repeatable tests and for environments where AkShare
    cannot reach Eastmoney. The CSV may use either English columns (date, close)
    or AkShare-style Chinese columns (日期, 收盘).
    """

    if csv_path is not None:
        return _normalize_price_frame(pd.read_csv(csv_path))

    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "AkShare is required when --csv is not provided. Install with: "
            "python -m pip install akshare"
        ) from exc

    raw = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
    return _normalize_price_frame(raw)


def filter_date_range(
    df: pd.DataFrame, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Apply optional inclusive start/end date filters."""

    filtered = df.copy()
    if start:
        filtered = filtered.loc[pd.Timestamp(start) :]
    if end:
        filtered = filtered.loc[: pd.Timestamp(end)]
    if filtered.empty:
        raise ValueError("No data remains after applying the requested date range.")
    return filtered


def latest_percentile_rank(values: Iterable[float]) -> float:
    """Return percentile rank of the latest value within a rolling window."""

    series = pd.Series(values)
    return float(series.rank(pct=True).iloc[-1])


def calc_factors(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Calculate trend and valuation-proxy factors."""

    result = df.copy()
    result["ma200"] = result["close"].rolling(config.ma_window).mean()
    result["price_pct"] = (
        result["close"]
        .rolling(config.price_percentile_window, min_periods=config.price_percentile_window)
        .apply(latest_percentile_rank, raw=False)
    )
    return result


def get_position(price_pct: float, close: float, ma200: float) -> float:
    """Map valuation and trend scores to a target position."""

    if pd.isna(price_pct):
        valuation_score = 60
    elif price_pct < 0.1:
        valuation_score = 100
    elif price_pct < 0.2:
        valuation_score = 90
    elif price_pct < 0.4:
        valuation_score = 80
    elif price_pct < 0.6:
        valuation_score = 60
    elif price_pct < 0.8:
        valuation_score = 40
    else:
        valuation_score = 20

    if pd.isna(ma200):
        trend_score = 60
    else:
        diff = close / ma200 - 1
        if diff > 0.1:
            trend_score = 100
        elif diff > 0:
            trend_score = 80
        elif diff > -0.1:
            trend_score = 60
        else:
            trend_score = 20

    score = valuation_score * 0.8 + trend_score * 0.2

    if score < 30:
        return 0.2
    if score < 40:
        return 0.4
    if score < 50:
        return 0.6
    if score < 60:
        return 0.8
    if score < 70:
        return 1.0
    if score < 80:
        return 1.2
    if score < 90:
        return 1.4
    return 1.6


def generate_position(df: pd.DataFrame, initial_position: float = 1.0) -> pd.Series:
    """Generate daily positions with rebalancing on actual month-end trading days."""

    if df.empty:
        raise ValueError("Cannot generate positions for an empty data set.")

    month_end_dates = df.groupby(df.index.to_period("M")).tail(1).index
    positions = pd.Series(index=df.index, dtype="float64")
    current_position = initial_position

    for dt, row in df.iterrows():
        if dt in month_end_dates:
            current_position = get_position(row["price_pct"], row["close"], row["ma200"])
        positions.loc[dt] = current_position

    return positions


def backtest(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Run the monthly timing backtest."""

    result = calc_factors(df, config)
    result["ret"] = result["close"].pct_change().fillna(0.0)
    result["position"] = generate_position(result, config.initial_position)
    result["cash_position"] = 1.0 - result["position"]
    result["strategy_ret"] = (
        result["position"].shift(1).fillna(config.initial_position) * result["ret"]
        + result["cash_position"].shift(1).fillna(0.0) * config.risk_free_daily_return
    )
    result["buyhold"] = (1 + result["ret"]).cumprod()
    result["strategy"] = (1 + result["strategy_ret"]).cumprod()
    return result


def performance(nav: pd.Series) -> dict[str, float]:
    """Calculate CAGR, maximum drawdown and annualized Sharpe ratio."""

    clean_nav = nav.dropna()
    if len(clean_nav) < 2:
        raise ValueError("At least two NAV observations are required for performance metrics.")

    years = max((clean_nav.index[-1] - clean_nav.index[0]).days / 365.25, 1 / 365.25)
    cagr = (clean_nav.iloc[-1] / clean_nav.iloc[0]) ** (1 / years) - 1
    drawdown = clean_nav / clean_nav.cummax() - 1
    daily_ret = clean_nav.pct_change().dropna()
    sharpe = np.nan
    if daily_ret.std(ddof=0) > 0:
        sharpe = daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "CAGR(%)": round(cagr * 100, 2),
        "MaxDD(%)": round(float(drawdown.min()) * 100, 2),
        "Sharpe": round(float(sharpe), 2) if not np.isnan(sharpe) else np.nan,
        "FinalNAV": round(float(clean_nav.iloc[-1]), 4),
    }


def save_or_show_plot(result: pd.DataFrame, plot_file: str | Path | None, no_plot: bool) -> None:
    """Render the NAV chart, either to a file or an interactive window."""

    if no_plot and plot_file is None:
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result["buyhold"], label="Buy & Hold")
    plt.plot(result.index, result["strategy"], label="Enhanced")
    plt.title(f"Dividend Low Vol ETF Timing Backtest ({ETF_CODE})")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if plot_file is not None:
        output_path = Path(plot_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    elif not no_plot:  # pragma: no cover - interactive path
        plt.show()

    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a monthly timing backtest for dividend low-volatility ETF 512890."
    )
    parser.add_argument("--symbol", default=ETF_CODE, help="ETF code for AkShare. Default: 512890")
    parser.add_argument("--csv", help="Optional local CSV path with date/close or 日期/收盘 columns.")
    parser.add_argument("--start", help="Inclusive start date, e.g. 2019-01-01.")
    parser.add_argument("--end", help="Inclusive end date, e.g. 2025-12-31.")
    parser.add_argument("--plot-file", help="Save NAV chart to this path instead of only showing it.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting unless --plot-file is set.")
    parser.add_argument("--export-csv", help="Optional path to export the full backtest result CSV.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = BacktestConfig(symbol=args.symbol, start=args.start, end=args.end)
    prices = load_data(symbol=config.symbol, csv_path=args.csv)
    prices = filter_date_range(prices, config.start, config.end)
    result = backtest(prices, config)

    print("========== Data Range ==========")
    print(f"{result.index[0].date()} -> {result.index[-1].date()} ({len(result)} rows)")
    print()
    print("========== Buy & Hold ==========")
    print(performance(result["buyhold"]))
    print()
    print("========== Enhanced ==========")
    print(performance(result["strategy"]))

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(export_path, index_label="date")
        print(f"\nBacktest details exported to: {export_path}")

    save_or_show_plot(result, args.plot_file, args.no_plot)
    if args.plot_file:
        print(f"NAV chart saved to: {args.plot_file}")


if __name__ == "__main__":
    main()
