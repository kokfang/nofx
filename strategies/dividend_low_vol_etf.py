"""Dependency-free dividend low-volatility ETF timing backtest.

The script first tries to fetch 512890 daily K-lines from Eastmoney's public API
with Python's standard library. If the current network blocks that request, it
falls back to a deterministic bundled demo data set so the program still runs and
prints a complete result immediately. Use --no-demo-fallback if you prefer a hard
failure when live data is unavailable.

Examples:
    python strategies/dividend_low_vol_etf.py --no-plot
    python strategies/dividend_low_vol_etf.py --start 2019-01-01 --plot-file output/nav.svg
    python strategies/dividend_low_vol_etf.py --csv data/512890.csv --export-csv output/detail.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

ETF_CODE = "512890"
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PriceBar:
    """One daily close-price observation."""

    day: date
    close: float


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the ETF timing strategy backtest."""

    symbol: str = ETF_CODE
    start: date | None = None
    end: date | None = None
    price_percentile_window: int = 750
    ma_window: int = 200
    initial_position: float = 1.0
    risk_free_daily_return: float = 0.0
    allow_demo_fallback: bool = True


def parse_day(value: str | None) -> date | None:
    """Parse YYYY-MM-DD dates used by CLI options."""

    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_price_rows(rows: Iterable[dict[str, str]]) -> list[PriceBar]:
    """Normalize CSV-like rows to sorted unique close-price bars."""

    bars_by_day: dict[date, PriceBar] = {}
    for row in rows:
        day_text = row.get("date") or row.get("日期") or row.get("交易日期") or row.get("Date")
        close_text = row.get("close") or row.get("收盘") or row.get("收盘价") or row.get("Close")
        if not day_text or not close_text:
            raise ValueError("CSV rows must contain date/close or 日期/收盘 columns.")
        day = datetime.strptime(day_text.strip()[:10], "%Y-%m-%d").date()
        close = float(close_text)
        if close <= 0:
            raise ValueError(f"Close price must be positive: {day}={close}")
        bars_by_day[day] = PriceBar(day=day, close=close)

    bars = sorted(bars_by_day.values(), key=lambda item: item.day)
    if not bars:
        raise ValueError("No valid price rows were loaded.")
    return bars


def load_csv(csv_path: str | Path) -> list[PriceBar]:
    """Load local CSV data with either English or AkShare-style Chinese columns."""

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return normalize_price_rows(csv.DictReader(handle))


def eastmoney_secid(symbol: str) -> str:
    """Return Eastmoney secid. ETF 512xxx codes trade on Shanghai by convention."""

    if symbol.startswith(("5", "6", "9")):
        return f"1.{symbol}"
    return f"0.{symbol}"


def load_eastmoney(symbol: str, start: date | None, end: date | None, timeout: int = 15) -> list[PriceBar]:
    """Fetch daily forward-adjusted K-line data from Eastmoney using stdlib only."""

    beg = (start or date(2018, 1, 1)).strftime("%Y%m%d")
    finish = (end or date.today()).strftime("%Y%m%d")
    params = {
        "secid": eastmoney_secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": beg,
        "end": finish,
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; nofx-etf-backtest/1.0)",
            "Referer": "https://quote.eastmoney.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"Eastmoney returned no K-line data for {symbol}.")

    rows = []
    for item in klines:
        fields = item.split(",")
        rows.append({"date": fields[0], "close": fields[2]})
    return normalize_price_rows(rows)


def generate_demo_data(start: date | None = None, end: date | None = None) -> list[PriceBar]:
    """Generate deterministic demo prices so the program always has a runnable path.

    The series is not real market data. It is deliberately labelled DEMO in CLI
    output and exists only to prove the full backtest pipeline in blocked/offline
    environments.
    """

    first = start or date(2019, 1, 2)
    last = end or date(2026, 5, 29)
    bars: list[PriceBar] = []
    current = first
    trading_index = 0
    while current <= last:
        if current.weekday() < 5:
            long_cycle = 0.10 * math.sin(trading_index / 165)
            short_cycle = 0.035 * math.sin(trading_index / 37)
            trend = 0.00018 * trading_index
            drawdown_2021 = -0.18 * math.exp(-((trading_index - 620) / 120) ** 2)
            drawdown_2024 = -0.12 * math.exp(-((trading_index - 1320) / 95) ** 2)
            close = 1.0 + trend + long_cycle + short_cycle + drawdown_2021 + drawdown_2024
            bars.append(PriceBar(day=current, close=round(max(close, 0.25), 4)))
            trading_index += 1
        current += timedelta(days=1)
    return bars


def load_data(config: BacktestConfig, csv_path: str | Path | None = None) -> tuple[list[PriceBar], str]:
    """Load prices from CSV, Eastmoney, or deterministic demo fallback."""

    if csv_path:
        return filter_date_range(load_csv(csv_path), config.start, config.end), f"CSV:{csv_path}"

    try:
        return filter_date_range(load_eastmoney(config.symbol, config.start, config.end), config.start, config.end), "Eastmoney"
    except Exception as exc:
        if not config.allow_demo_fallback:
            raise
        print(f"WARNING: live Eastmoney data unavailable ({exc}); using bundled DEMO data.", file=sys.stderr)
        return filter_date_range(generate_demo_data(config.start, config.end), config.start, config.end), "DEMO (not real market data)"


def filter_date_range(bars: Sequence[PriceBar], start: date | None, end: date | None) -> list[PriceBar]:
    """Apply optional inclusive date filters."""

    filtered = [bar for bar in bars if (start is None or bar.day >= start) and (end is None or bar.day <= end)]
    if not filtered:
        raise ValueError("No data remains after applying the requested date range.")
    return filtered


def mean(values: Sequence[float]) -> float:
    """Return arithmetic mean for a non-empty sequence."""

    return sum(values) / len(values)


def latest_percentile_rank(values: Sequence[float]) -> float:
    """Return percentile rank of the latest value within a rolling window."""

    latest = values[-1]
    less = sum(1 for value in values if value < latest)
    equal = sum(1 for value in values if value == latest)
    average_rank = less + (equal + 1) / 2
    return average_rank / len(values)


def get_position(price_pct: float | None, close: float, ma200: float | None) -> float:
    """Map valuation and trend scores to a target position."""

    if price_pct is None:
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

    if ma200 is None:
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


def is_month_end_trading_day(bars: Sequence[PriceBar], index: int) -> bool:
    """Return True when a bar is the last available trading day in its month."""

    if index == len(bars) - 1:
        return True
    today = bars[index].day
    tomorrow_bar = bars[index + 1].day
    return today.year != tomorrow_bar.year or today.month != tomorrow_bar.month


def backtest(bars: Sequence[PriceBar], config: BacktestConfig) -> list[dict[str, float | date | None]]:
    """Run the monthly timing backtest and return row dictionaries."""

    rows: list[dict[str, float | date | None]] = []
    current_position = config.initial_position
    buyhold = 1.0
    strategy = 1.0

    closes = [bar.close for bar in bars]
    for index, bar in enumerate(bars):
        close = bar.close
        previous_close = bars[index - 1].close if index > 0 else close
        ret = close / previous_close - 1 if index > 0 else 0.0

        ma200 = None
        if index + 1 >= config.ma_window:
            ma200 = mean(closes[index + 1 - config.ma_window : index + 1])

        price_pct = None
        if index + 1 >= config.price_percentile_window:
            price_pct = latest_percentile_rank(closes[index + 1 - config.price_percentile_window : index + 1])

        previous_position = rows[-1]["position"] if rows else config.initial_position
        cash_position = 1.0 - float(previous_position)
        strategy_ret = float(previous_position) * ret + cash_position * config.risk_free_daily_return
        buyhold *= 1 + ret
        strategy *= 1 + strategy_ret

        if is_month_end_trading_day(bars, index):
            current_position = get_position(price_pct, close, ma200)

        rows.append(
            {
                "date": bar.day,
                "close": close,
                "ret": ret,
                "ma200": ma200,
                "price_pct": price_pct,
                "position": current_position,
                "buyhold": buyhold,
                "strategy": strategy,
                "strategy_ret": strategy_ret,
            }
        )
    return rows


def performance(rows: Sequence[dict[str, float | date | None]], nav_column: str) -> dict[str, float]:
    """Calculate CAGR, maximum drawdown, Sharpe ratio and final NAV."""

    if len(rows) < 2:
        raise ValueError("At least two NAV observations are required for performance metrics.")
    first_day = rows[0]["date"]
    last_day = rows[-1]["date"]
    assert isinstance(first_day, date) and isinstance(last_day, date)
    navs = [float(row[nav_column]) for row in rows]
    years = max((last_day - first_day).days / 365.25, 1 / 365.25)
    cagr = (navs[-1] / navs[0]) ** (1 / years) - 1

    peak = navs[0]
    max_drawdown = 0.0
    for nav in navs:
        peak = max(peak, nav)
        max_drawdown = min(max_drawdown, nav / peak - 1)

    daily_returns = [navs[i] / navs[i - 1] - 1 for i in range(1, len(navs))]
    if len(daily_returns) > 1 and statistics.pstdev(daily_returns) > 0:
        sharpe = statistics.mean(daily_returns) / statistics.pstdev(daily_returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = float("nan")

    return {
        "CAGR(%)": round(cagr * 100, 2),
        "MaxDD(%)": round(max_drawdown * 100, 2),
        "Sharpe": round(sharpe, 2) if not math.isnan(sharpe) else float("nan"),
        "FinalNAV": round(navs[-1], 4),
    }


def export_csv(rows: Sequence[dict[str, float | date | None]], output_path: str | Path) -> None:
    """Export full backtest rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "close", "ret", "ma200", "price_pct", "position", "buyhold", "strategy", "strategy_ret"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_svg_plot(rows: Sequence[dict[str, float | date | None]], output_path: str | Path) -> None:
    """Save a small dependency-free SVG NAV chart."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 420
    pad_left, pad_right, pad_top, pad_bottom = 60, 24, 24, 44
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    nav_values = [float(row["buyhold"]) for row in rows] + [float(row["strategy"]) for row in rows]
    min_nav, max_nav = min(nav_values), max(nav_values)
    if math.isclose(min_nav, max_nav):
        max_nav += 0.01

    def points(column: str) -> str:
        coords = []
        for i, row in enumerate(rows):
            x = pad_left + (i / max(len(rows) - 1, 1)) * chart_w
            nav = float(row[column])
            y = pad_top + (max_nav - nav) / (max_nav - min_nav) * chart_h
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    first_day = rows[0]["date"]
    last_day = rows[-1]["date"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2}" y="18" text-anchor="middle" font-size="16" font-family="Arial">Dividend Low Vol ETF Timing Backtest ({ETF_CODE})</text>
  <line x1="{pad_left}" y1="{pad_top + chart_h}" x2="{pad_left + chart_w}" y2="{pad_top + chart_h}" stroke="#888"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + chart_h}" stroke="#888"/>
  <polyline points="{points('buyhold')}" fill="none" stroke="#8884d8" stroke-width="2"/>
  <polyline points="{points('strategy')}" fill="none" stroke="#f0b90b" stroke-width="2"/>
  <text x="{pad_left}" y="{height - 12}" font-size="12" font-family="Arial">{first_day}</text>
  <text x="{width - pad_right}" y="{height - 12}" text-anchor="end" font-size="12" font-family="Arial">{last_day}</text>
  <text x="{pad_left + 8}" y="{pad_top + 18}" font-size="12" font-family="Arial" fill="#8884d8">Buy & Hold</text>
  <text x="{pad_left + 8}" y="{pad_top + 36}" font-size="12" font-family="Arial" fill="#b38300">Enhanced</text>
  <text x="8" y="{pad_top + 12}" font-size="12" font-family="Arial">{max_nav:.2f}</text>
  <text x="8" y="{pad_top + chart_h}" font-size="12" font-family="Arial">{min_nav:.2f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    """Print metrics in a compact stable format."""

    print(f"========== {title} ==========")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run a monthly timing backtest for dividend low-volatility ETF 512890.")
    parser.add_argument("--symbol", default=ETF_CODE, help="ETF code for Eastmoney. Default: 512890")
    parser.add_argument("--csv", help="Optional local CSV path with date/close or 日期/收盘 columns.")
    parser.add_argument("--start", default="2019-01-01", help="Inclusive start date. Default: 2019-01-01")
    parser.add_argument("--end", help="Inclusive end date. Default: today")
    parser.add_argument("--plot-file", help="Save a dependency-free SVG NAV chart to this path.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write a plot unless --plot-file is set.")
    parser.add_argument("--export-csv", help="Optional path to export the full backtest result CSV.")
    parser.add_argument("--no-demo-fallback", action="store_true", help="Fail instead of using bundled demo data when live data is unavailable.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = BacktestConfig(
        symbol=args.symbol,
        start=parse_day(args.start),
        end=parse_day(args.end),
        allow_demo_fallback=not args.no_demo_fallback,
    )
    bars, source = load_data(config, csv_path=args.csv)
    rows = backtest(bars, config)

    print("========== Data Range ==========")
    print(f"Source: {source}")
    print(f"{rows[0]['date']} -> {rows[-1]['date']} ({len(rows)} rows)")
    print(f"Latest close: {rows[-1]['close']}")
    print(f"Latest position: {rows[-1]['position']}x")
    print()
    print_metrics("Buy & Hold", performance(rows, "buyhold"))
    print()
    print_metrics("Enhanced", performance(rows, "strategy"))

    if args.export_csv:
        export_csv(rows, args.export_csv)
        print(f"\nBacktest details exported to: {args.export_csv}")

    if args.plot_file:
        save_svg_plot(rows, args.plot_file)
        print(f"NAV chart saved to: {args.plot_file}")
    elif not args.no_plot:
        default_plot = Path("output") / f"{args.symbol}_nav.svg"
        save_svg_plot(rows, default_plot)
        print(f"NAV chart saved to: {default_plot}")


if __name__ == "__main__":
    main()
