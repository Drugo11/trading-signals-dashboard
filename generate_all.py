"""
Script to generate trading signals and performance for multiple strategies.

This script produces the following artefacts:

* `signals_trend.js` – JavaScript file defining the constant `signals_trend` with
  a list of daily signals for a simple trend‑following strategy on a basket of
  exchange‑traded funds (ETFs).  Each entry contains the date, ticker,
  prediction (percent difference between the price and its 200‑day moving
  average) and the position signal (1 for long, 0 for flat).  A regime
  filter based on the variance risk premium (VRP) can down‑weight signals
  when the market environment is unfavourable.

* `signals_xsmom.js` – JavaScript file defining the constant `signals_xsmom`
  with a list of daily signals for a cross‑sectional momentum strategy on a
  universe of large‑cap stocks.  The prediction is the 12‑month minus
  1‑month return; the signal is +1 for the top 20 % of predictions,
  −1 for the bottom 20 % and 0 otherwise.  The same VRP filter can
  down‑weight these signals.

* `reports/perf_summary.csv` – A CSV summarising the annualised return
  (CAGR), volatility, Sharpe ratio and maximum drawdown of each strategy.

* `reports/equity_trend.csv` and `reports/equity_xsmom.csv` – The equity
  curves for each strategy, i.e. the cumulative wealth from investing one
  unit of capital using equal weighting across signals.  Each file has a
  single column `equity` indexed by date.

* `reports/daily_returns_trend.csv` and `reports/daily_returns_xsmom.csv` –
  Daily realised returns for the corresponding strategies.

The script also calls the existing `generate_signals_json.py` to update
the machine‑learning signals (`signals.js`) so that all three sets of
signals are refreshed together.

To execute the script manually run:

    python generate_all_new.py

It requires the `yfinance`, `pandas` and `numpy` packages to be installed.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yfinance as yf
import json
import subprocess


def generate_ml_signals() -> None:
    """Run the existing ML signal generator to refresh signals.js.

    This function invokes `generate_signals_json.py` via subprocess.  It
    expects that script to reside in the current working directory and
    produce/up‑date `signals.js` when executed.  Any exceptions are caught
    and printed so as not to interrupt the remainder of the pipeline.
    """
    try:
        subprocess.run(["python", "generate_signals_json.py"], check=True)
    except Exception as exc:
        print("Error generating ML signals:", exc)


def _download_prices(tickers: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Download adjusted closing prices for multiple tickers.

    yfinance returns a DataFrame with a column per ticker when
    `auto_adjust=True` and `progress=False`.  If a single ticker is
    requested the returned object is a Series, so we wrap that back into a
    DataFrame for consistency.

    Args:
        tickers: Iterable of ticker symbols.
        start: Start date (inclusive).
        end: End date (exclusive).

    Returns:
        DataFrame of adjusted closing prices indexed by date with
        tickers as columns.
    """
    data = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data, pd.DataFrame):
        if "Close" in data.columns:
            return data["Close"].copy()
        else:
            return data.copy()
    else:
        return data.to_frame(name=list(tickers)[0])


def compute_vrp(start: datetime, end: datetime) -> pd.Series:
    """Compute the variance risk premium (VRP) time series.

    The VRP is calculated as the difference between the square of the VIX
    (implied variance) divided by 100 and the realised variance of the SPY
    over a 20‑day rolling window.  It is annualised to be comparable and
    aligned with the one‑month horizon of VIX.

    Args:
        start: Start date for the VRP (inclusive).
        end: End date for the VRP (exclusive).

    Returns:
        A Series indexed by date with the VRP value.  Positive values
        indicate periods where implied volatility exceeds realised volatility.
    """
    extended_start = start - timedelta(days=60)
    vix = _download_prices(["^VIX"], extended_start, end).iloc[:, 0]
    spy = _download_prices(["SPY"], extended_start, end).iloc[:, 0]
    realised_var = spy.pct_change().rolling(window=20).std() ** 2 * 252
    implied_var = (vix / 100) ** 2
    vrp = implied_var - realised_var
    vrp = vrp.loc[start: end - timedelta(days=1)]
    return vrp


def generate_trend_signals(
    tickers: List[str], start: datetime, end: datetime, vrp: pd.Series
) -> List[Dict[str, object]]:
    """Generate simple trend‑following signals.

    The prediction is the percent difference between the price and its
    200‑day moving average.  A long signal (1) is issued when the price
    exceeds its moving average.  There are no short signals in this
    strategy.  Signals and predictions are optionally down‑weighted to
    50 % when the VRP is negative, indicating stressed market conditions.

    Args:
        tickers: List of ticker symbols.
        start: Start date for historical data (inclusive).
        end: End date for data (exclusive).
        vrp: Variance risk premium time series to apply regime filter.

    Returns:
        List of dictionaries with keys `date`, `ticker`, `prediction` and
        `signal`.
    """
    prices = _download_prices(tickers, start, end)
    sma200 = prices.rolling(window=200).mean()
    signals: List[Dict[str, object]] = []
    for date in prices.index:
        g = 1.0
        if date in vrp.index and vrp.loc[date] < 0:
            g = 0.5
        dt_str = date.strftime("%Y-%m-%d")
        for ticker in tickers:
            price = prices.at[date, ticker]
            ma = sma200.at[date, ticker]
            if pd.isna(price) or pd.isna(ma):
                continue
            pred = (price / ma) - 1.0
            sig = 1 if price > ma else 0
            pred *= g
            sig = int(sig * g)
            signals.append(
                {
                    "date": dt_str,
                    "ticker": ticker,
                    "prediction": float(pred),
                    "signal": sig,
                }
            )
    return signals


def generate_xsmom_signals(
    tickers: List[str], start: datetime, end: datetime, vrp: pd.Series
) -> List[Dict[str, object]]:
    """Generate cross‑sectional momentum signals.

    The prediction is defined as the 12‑month return minus the 1‑month
    return.  For each date the universe is ranked by this prediction
    value; the top 20 % receive a long signal (+1), the bottom 20 % a
    short signal (−1) and the remainder 0.  Signals and predictions are
    down‑weighted by 50 % when the VRP is negative.

    Args:
        tickers: List of ticker symbols.
        start: Start date (inclusive).
        end: End date (exclusive).
        vrp: Variance risk premium time series for regime filtering.

    Returns:
        List of dictionaries with keys `date`, `ticker`, `prediction` and
        `signal`.
    """
    extended_start = start - timedelta(days=365)
    prices = _download_prices(tickers, extended_start, end)
    mom12 = prices.pct_change(252)
    mom01 = prices.pct_change(21)
    momentum = mom12 - mom01
    signals: List[Dict[str, object]] = []
    for date in momentum.loc[start:end - timedelta(days=1)].index:
        preds = momentum.loc[date].dropna()
        if preds.empty:
            continue
        g = 1.0
        if date in vrp.index and vrp.loc[date] < 0:
            g = 0.5
        sorted_preds = preds.sort_values()
        n = len(sorted_preds)
        top_k = max(int(np.ceil(n * 0.2)), 1)
        bottom_k = max(int(np.ceil(n * 0.2)), 1)
        top_tickers = sorted_preds.index[-top_k:]
        bottom_tickers = sorted_preds.index[:bottom_k]
        dt_str = date.strftime("%Y-%m-%d")
        for ticker in preds.index:
            pred = float(preds[ticker]) * g
            if ticker in top_tickers:
                sig = int(1 * g)
            elif ticker in bottom_tickers:
                sig = int(-1 * g)
            else:
                sig = 0
            signals.append(
                {
                    "date": dt_str,
                    "ticker": ticker,
                    "prediction": pred,
                    "signal": sig,
                }
            )
    return signals


def save_js(data: List[Dict[str, object]], var_name: str, file_path: str) -> None:
    json_str = json.dumps(data)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"const {var_name} = {json_str};\n")


def compute_performance(signals: List[Dict[str, object]]) -> Dict[str, object]:
    df = pd.DataFrame(signals)
    df["date"] = pd.to_datetime(df["date"])
    signal_matrix = df.pivot_table(index="date", columns="ticker", values="signal", fill_value=0)
    tickers = signal_matrix.columns.tolist()
    price_start = signal_matrix.index.min()
    price_end = signal_matrix.index.max() + timedelta(days=2)
    prices = _download_prices(tickers, price_start, price_end)
    returns = prices.pct_change().shift(-1)
    aligned_returns = returns.reindex(signal_matrix.index)
    daily_ret = (signal_matrix * aligned_returns).mean(axis=1)
    daily_ret = daily_ret.dropna()
    equity = (1 + daily_ret).cumprod()
    n_days = len(equity)
    cagr = equity.iloc[-1] ** (252 / n_days) - 1 if n_days > 0 else 0.0
    vol = daily_ret.std() * np.sqrt(252) if not daily_ret.empty else 0.0
    sharpe = (cagr / vol) if vol != 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min() if not equity.empty else 0.0
    return {
        "equity_curve": equity,
        "daily_returns": daily_ret,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
    }


def save_performance_report(perf: Dict[str, object], name: str) -> None:
    os.makedirs("reports", exist_ok=True)
    perf["equity_curve"].rename("equity").to_csv(f"reports/equity_{name}.csv")
    perf["daily_returns"].rename("return").to_csv(f"reports/daily_returns_{name}.csv")


def main() -> None:
    generate_ml_signals()
    today = datetime.now().date()
    start_date = today - timedelta(days=730)
    etf_universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
    equity_universe = [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "TSLA", "BRK-B", "JPM", "XOM",
    ]
    vrp = compute_vrp(start_date, today + timedelta(days=1))
    trend_signals = generate_trend_signals(etf_universe, start_date, today + timedelta(days=1), vrp)
    xsmom_signals = generate_xsmom_signals(equity_universe, start_date, today + timedelta(days=1), vrp)
    save_js(trend_signals, "signals_trend", "signals_trend.js")
    save_js(xsmom_signals, "signals_xsmom", "signals_xsmom.js")
    perf_trend = compute_performance(trend_signals)
    perf_xsmom = compute_performance(xsmom_signals)
    save_performance_report(perf_trend, "trend")
    save_performance_report(perf_xsmom, "xsmom")
    summary = pd.DataFrame([
        {
            "strategy": "trend",
            "CAGR": perf_trend["CAGR"],
            "Volatility": perf_trend["Volatility"],
            "Sharpe": perf_trend["Sharpe"],
            "MaxDD": perf_trend["MaxDD"],
        },
        {
            "strategy": "xsmom",
            "CAGR": perf_xsmom["CAGR"],
            "Volatility": perf_xsmom["Volatility"],
            "Sharpe": perf_xsmom["Sharpe"],
            "MaxDD": perf_xsmom["MaxDD"],
        },
    ])
    os.makedirs("reports", exist_ok=True)
    summary.to_csv("reports/perf_summary.csv", index=False)


if __name__ == "__main__":
    main()
