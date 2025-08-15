import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import subprocess


def generate_ml_signals():
    """Call the existing generate_signals_json script to update signals.js for ML signals."""
    try:
        # Run the script using subprocess. It will create/update signals.js
        subprocess.run(["python", "generate_signals_json.py"], check=True)
    except Exception as exc:
        print("Error generating ML signals:", exc)


def generate_trend_signals(tickers, start, end):
    """Generate trend-following signals based on price vs. a 200-day moving average.

    For each date and ticker, compute the percent difference between the price and
    its 200-day moving average as a prediction. Generate a long signal (1) if
    the price is above the moving average and 0 otherwise. No short signals
    are generated for this simple trend strategy.

    Args:
        tickers: List of tickers to process.
        start: Start date for historical data.
        end: End date for historical data.

    Returns:
        List of dictionaries with keys date, ticker, prediction, and signal.
    """
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    sma = data.rolling(window=200).mean()
    signals = []
    for date in data.index:
        dt_str = date.strftime("%Y-%m-%d")
        for ticker in tickers:
            price = data.loc[date, ticker]
            ma = sma.loc[date, ticker]
            # Skip if we don't have enough history for the moving average
            if pd.isna(price) or pd.isna(ma):
                continue
            pred = (price / ma) - 1.0
            sig = 1 if price > ma else 0
            signals.append(
                {
                    "date": dt_str,
                    "ticker": ticker,
                    "prediction": float(pred),
                    "signal": int(sig),
                }
            )
    return signals


def generate_xsmom_signals(tickers, start, end):
    """Generate cross-sectional momentum signals using 12-month minus 1-month returns.

    Momentum is calculated daily as (12m return – 1m return). On each date, all
    available tickers are ranked by this value. Signals are assigned as
    follows: long (1) for the top 20% of tickers by momentum, short (‑1)
    for the bottom 20%, and 0 for the middle 60%.

    Args:
        tickers: List of tickers to process.
        start: Start date for historical data.
        end: End date for historical data.

    Returns:
        List of dictionaries with keys date, ticker, prediction, and signal.
    """
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    # Calculate 12-month and 1-month returns using trading days (~252 and 21 days)
    mom12 = data / data.shift(252) - 1.0
    mom1 = data / data.shift(21) - 1.0
    momentum = mom12 - mom1
    signals = []
    for date in momentum.index:
        dt_str = date.strftime("%Y-%m-%d")
        row = momentum.loc[date].dropna()
        if row.empty:
            continue
        q80 = row.quantile(0.8)
        q20 = row.quantile(0.2)
        for ticker in tickers:
            value = momentum.at[date, ticker]
            if pd.isna(value):
                continue
            pred = float(value)
            if value >= q80:
                sig = 1
            elif value <= q20:
                sig = -1
            else:
                sig = 0
            signals.append(
                {
                    "date": dt_str,
                    "ticker": ticker,
                    "prediction": pred,
                    "signal": int(sig),
                }
            )
    return signals


def save_js(filename: str, var_name: str, data_list):
    """Save a list of dictionaries to a JavaScript file.

    The file will define a single variable with the given name and assign the
    provided data list to it. This makes it easy to load the data in a
    browser environment.
    """
    with open(filename, "w") as f:
        f.write(f"const {var_name} = ")
        json.dump(data_list, f)
        f.write(";")


def compute_performance(signals, data):
    """Compute daily strategy returns given signals and price data.

    For each date, actual returns are computed from price data and then
    multiplied by the corresponding signals. The daily strategy return is the
    average of these position returns. When no signals are present for a date
    the date is skipped.
    """
    returns = data.pct_change().dropna()
    records = []
    for date in returns.index:
        dt_str = date.strftime("%Y-%m-%d")
        # collect signals for this date
        day_signals = [s for s in signals if s["date"] == dt_str]
        if not day_signals:
            continue
        rets = []
        for s in day_signals:
            ticker = s["ticker"]
            sig = s["signal"]
            if sig == 0:
                continue
            if ticker not in returns.columns:
                continue
            r = returns.loc[date, ticker]
            if pd.isna(r):
                continue
            rets.append(sig * r)
        if rets:
            day_ret = float(np.mean(rets))
            records.append({"date": dt_str, "return": day_ret})
    df = pd.DataFrame(records)
    if not df.empty:
        df.set_index("date", inplace=True)
    return df


def save_performance_report(trend_returns: pd.DataFrame, xsmom_returns: pd.DataFrame):
    """Save a summary CSV and equity curves for the strategies.

    A summary CSV containing CAGR, volatility, Sharpe ratio and maximum
    drawdown is written to ``reports/perf_summary.csv``. Daily equity curves
    for each strategy are stored in ``reports/equity_trend.csv`` and
    ``reports/equity_xsmom.csv``.
    """
    def calc_metrics(df):
        if df.empty:
            return 0.0, 0.0, 0.0, 0.0
        cum = (1 + df["return"]).cumprod()
        total_days = len(df)
        cagr = cum.iloc[-1] ** (252 / total_days) - 1
        vol = df["return"].std() * np.sqrt(252)
        # Sharpe ratio: annualized return divided by annualized volatility
        if df["return"].std() != 0:
            sharpe = (df["return"].mean() * 252) / (df["return"].std() * np.sqrt(252))
        else:
            sharpe = 0.0
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        maxdd = drawdown.min()
        return float(cagr), float(vol), float(sharpe), float(maxdd)

    cagr_t, vol_t, sharpe_t, maxdd_t = calc_metrics(trend_returns)
    cagr_x, vol_x, sharpe_x, maxdd_x = calc_metrics(xsmom_returns)
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(
        [
            {
                "strategy": "Trend ETF",
                "CAGR": cagr_t,
                "Vol": vol_t,
                "Sharpe": sharpe_t,
                "MaxDD": maxdd_t,
            },
            {
                "strategy": "Momentum 12-1",
                "CAGR": cagr_x,
                "Vol": vol_x,
                "Sharpe": sharpe_x,
                "MaxDD": maxdd_x,
            },
        ]
    ).to_csv("reports/perf_summary.csv", index=False)
    # save equity curves
    trend_returns.to_csv("reports/equity_trend.csv")
    xsmom_returns.to_csv("reports/equity_xsmom.csv")


def main():
    # Generate ML signals first so signals.js is up to date
    generate_ml_signals()
    # Determine lookback period (two years)
    today = datetime.today().date()
    start_date = today - timedelta(days=730)
    end_date = today
    # Define universes
    etfs = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
    largecaps = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "GOOG",
        "TSLA",
        "BRK-B",
        "JPM",
        "V",
        "UNH",
        "JNJ",
        "XOM",
        "PG",
        "MA",
        "HD",
        "ABBV",
        "MRK",
        "PEP",
        "LLY",
        "BAC",
        "KO",
        "AVGO",
        "PFE",
        "CSCO",
        "TMO",
        "COST",
        "CVX",
        "WMT",
        "ABT",
        "AMGN",
        "ACN",
        "INTC",
        "QCOM",
        "MCD",
        "DHR",
        "BMY",
        "TXN",
        "LOW",
        "NEE",
        "ORCL",
        "LIN",
        "NKE",
        "RTX",
        "UPS",
        "MMM",
        "HON",
        "MS",
        "GS",
    ]
    # Generate strategy signals
    trend_signals = generate_trend_signals(etfs, start_date, end_date)
    xsmom_signals = generate_xsmom_signals(largecaps, start_date, end_date)
    # Save JS files for front-end
    save_js("signals_trend.js", "signals_trend", trend_signals)
    save_js("signals_xsmom.js", "signals_xsmom", xsmom_signals)
    # Fetch data for performance evaluation
    data_trend = yf.download(etfs, start=start_date, end=end_date, progress=False)["Adj Close"]
    data_lc = yf.download(largecaps, start=start_date, end=end_date, progress=False)["Adj Close"]
    # Compute daily returns for each strategy
    trend_returns = compute_performance(trend_signals, data_trend)
    xsmom_returns = compute_performance(xsmom_signals, data_lc)
    # Save performance summary and equity curves
    save_performance_report(trend_returns, xsmom_returns)


if __name__ == "__main__":
    main()
