"""
trading_system.py
==================

This module contains a simple but extensible framework for building a
prototype cross‑sectional machine learning trading strategy.  It is
intended as a starting point for experimenting with ideas such as
feature engineering, regime detection, meta‑labeling and walk‑forward
evaluation.  The default implementation focuses on a small universe of
liquid equities, but can be customised to any set of assets.  If
network access or the `yfinance` package is unavailable, the module
generates synthetic price series so that the pipeline can still be
executed end‑to‑end.

The high level workflow is:

1. Fetch or simulate historical price data for the selected tickers.
2. Compute a handful of technical features (momentum and volatility).
3. Build a long‑format data frame where each row corresponds to a
   (date, ticker) pair with associated features and future returns.
4. Split the data into sequential train/test folds using
   `TimeSeriesSplit` and fit a gradient boosting model on each train
   window.
5. Convert the model predictions into long/short positions (long the
   top quantile, short the bottom quantile) and apply a simple
   volatility targeting scheme.
6. Aggregate returns across tickers within each day to produce a
   strategy equity curve and compute summary statistics.

This module is deliberately simple: it does not include data cleansing,
purging or embargo procedures to prevent leakage, nor does it model
transaction costs in detail.  These features can be added later as
needed.  The code is annotated with docstrings to aid in
understanding and extension.

Example usage (run in a Python interpreter):

    from trading_system.trading_system import run_backtest

    # define the universe and run the pipeline
    equity_curve, stats = run_backtest(
        tickers=["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
        start="2018-01-01",
        horizon=5,
        n_splits=6,
        quantile=0.1,
        target_vol=0.10,
    )
    print(stats)

    # Equity curve is a pandas Series indexed by date
    equity_curve.plot(title="Strategy Equity Curve")

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore
    warnings.warn(
        "XGBoost is not installed. The pipeline will not work until you"
        " install the `xgboost` package."
    )

# Attempt to import pandas_datareader for macro data.  If unavailable, we'll fall back
# to synthetic macro series later.  Use alias pdr to avoid confusion with pd.
try:
    from pandas_datareader import data as pdr  # type: ignore
except ImportError:  # pragma: no cover
    pdr = None  # type: ignore


def fetch_price_data(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    end: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Fetch adjusted closing prices for a list of tickers.

    The function attempts to download data using the `yfinance` package.
    If `yfinance` is not installed or network access fails, it falls
    back to generating synthetic price series using a geometric Brownian
    motion model.  The synthetic data allows the rest of the pipeline
    to run for demonstration purposes.

    Parameters
    ----------
    tickers : Iterable[str]
        Collection of ticker symbols to download.
    start : str, optional
        Start date (inclusive) for the price history, by default
        "2018-01-01".
    end : str, optional
        End date (exclusive) for the price history.  If ``None``, the
        current date is used.
    seed : int, optional
        Random seed for synthetic data generation, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns for each ticker.  Dates
        correspond to business days (Monday–Friday).  If real data is
        downloaded the values represent adjusted closing prices.  If
        synthetic data is generated, the units are arbitrary but
        follow a realistic lognormal random walk.
    """
    tickers = list(tickers)
    # Attempt to use yfinance if available
    try:
        import yfinance as yf  # type: ignore
        # Use auto_adjust=True so dividends/splits are accounted for
        df = (
            yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=True
            )
            .loc[:, (slice(None), "Close")]
        )
        # When multiple tickers are downloaded, yfinance returns a
        # multi‑index column: (ticker, 'Close'); flatten it
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df = df.dropna(how="all")
        if not df.empty:
            return df
    except Exception as exc:
        warnings.warn(
            f"Failed to fetch data via yfinance ({exc}). Falling back to synthetic data."
        )

    # Generate synthetic GBM data when yfinance fails
    rng = np.random.default_rng(seed)
    # Define trading dates (business days) from start to end
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    # Random drift and volatility parameters per ticker
    drift = rng.uniform(0.0001, 0.0005, len(tickers))
    vol = rng.uniform(0.01, 0.03, len(tickers))
    prices: Dict[str, List[float]] = {}
    for i, tkr in enumerate(tickers):
        # Start price drawn between 50 and 150
        price = 100.0 * rng.lognormal(0, 0.2)
        series = []
        for _ in range(n):
            # Geometric Brownian Motion: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            # With dt=1 day
            z = rng.normal()
            price *= np.exp((drift[i] - 0.5 * vol[i] ** 2) + vol[i] * z)
            series.append(price)
        prices[tkr] = series
    synthetic_df = pd.DataFrame(prices, index=dates)
    return synthetic_df


def fetch_macro_data(
    start: str = "2018-01-01",
    end: Optional[str] = None,
    series: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Fetch macroeconomic time series from FRED or generate synthetic data.

    This helper retrieves daily (or lower frequency) macro variables from the
    Federal Reserve's FRED database via ``pandas_datareader``.  If fetching
    fails or the library is unavailable, it generates synthetic random
    series so the rest of the pipeline can run without external
    dependencies.

    Parameters
    ----------
    start : str
        Start date for the macro series, as an ISO format string.
    end : str, optional
        End date for the macro series.  If None, uses the current date.
    series : Dict[str, str], optional
        Mapping from FRED series codes to column names to use in the
        returned DataFrame.  A default mapping is used if this is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by business date with columns corresponding
        to macro variables.  Missing values are forward‑filled.  When
        synthetic data is generated, a random walk is used.
    """
    # Default FRED series mapping
    if series is None:
        series = {
            "FEDFUNDS": "fed_funds",
            "CPIAUCSL": "cpi",
            "UNRATE": "unemployment",
        }
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    try:
        if pdr is not None:
            frames = []
            for code, name in series.items():
                df = pdr.DataReader(code, "fred", start, end_ts)
                df = df.rename(columns={code: name})
                frames.append(df)
            macro_df = pd.concat(frames, axis=1)
            macro_df = macro_df.sort_index().ffill()
            return macro_df
    except Exception:
        # proceed to synthetic fallback
        pass
    # Synthetic fallback
    date_range = pd.bdate_range(start=start, end=end_ts)
    rng = np.random.default_rng(42)
    macro_df = pd.DataFrame(index=date_range)
    for _, name in series.items():
        noise = rng.normal(0, 0.02, len(date_range))
        macro_df[name] = np.cumsum(noise)
    return macro_df


def build_dataset(
    price_df: pd.DataFrame,
    horizon: int = 5,
    macro_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct a long‑form data set with features, future returns and optional macro data.

    For each ticker, a set of technical features is computed.  These
    features are combined into a MultiIndex DataFrame with a top level
    representing the feature name and a second level for the ticker.
    The result is reshaped into a long format suitable for
    cross‑sectional learning: columns include 'Date', 'Ticker', feature
    values, optional macro variables and 'y' (the forward return over
    the specified horizon).

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide DataFrame of price history with tickers as columns and
        datetime index.
    horizon : int, optional
        Number of business days over which to compute future returns,
        by default 5.
    macro_df : pd.DataFrame, optional
        DataFrame of macroeconomic features indexed by date.  Columns
        represent different macro variables.  These columns will be
        merged onto the long‑format data and treated as additional
        features.  If ``None``, no macro data is used.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        A tuple containing the long‑format DataFrame and a list of
        feature column names (including macro names if provided).  The
        DataFrame has columns ['Date', 'Ticker', *features, 'y'].
    """
    # Technical features: momentum over 5 and 20 days, volatility over 20
    # days, relative volatility (20d / 60d)
    returns = price_df.pct_change()
    feature_dict: Dict[str, pd.DataFrame] = {
        "mom_5": price_df.pct_change(5),
        "mom_20": price_df.pct_change(20),
        "vol_20": returns.rolling(20).std(),
        "rvol": returns.rolling(20).std() / returns.rolling(60).std(),
    }
    feat = pd.concat(feature_dict, axis=1)
    target = price_df.shift(-horizon).pct_change(horizon)
    df = feat.dropna().stack(level=1).reset_index()
    df = df.rename(columns={"level_0": "Date", "level_1": "Ticker"})
    feature_cols: List[str] = list(feature_dict.keys())
    tgt = target.stack().reset_index()
    tgt = tgt.rename(columns={"level_0": "Date", "level_1": "Ticker", 0: "y"})
    df = df.merge(tgt, on=["Date", "Ticker"], how="inner")
    # Include macro features if provided
    if macro_df is not None and not macro_df.empty:
        macro_df_local = macro_df.copy()
        macro_df_local.index.name = "Date"
        macro_df_local = macro_df_local.sort_index().reset_index()
        macro_df_local = macro_df_local.ffill()
        df = df.merge(macro_df_local, on="Date", how="left")
        macro_cols = [c for c in macro_df_local.columns if c != "Date"]
        df = df.dropna(subset=macro_cols)
        feature_cols.extend(macro_cols)
    df = df.dropna(subset=["y"] + feature_cols)
    return df, feature_cols


def time_series_cv(
    df: pd.DataFrame, n_splits: int = 5
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test indices for time series cross‑validation.

    The splitter respects the chronological order of the data.  Each fold
    uses an expanding training window and a fixed test window.  The
    indices refer to rows in the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to split.  Must be sorted by date.
    n_splits : int, optional
        Number of folds, by default 5.

    Yields
    ------
    Iterable[Tuple[np.ndarray, np.ndarray]]
        Pairs of train/test index arrays.
    """
    # Ensure the data is sorted by Date
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(df_sorted):
        yield train_idx, test_idx


@dataclass
class BacktestResult:
    """Container for the results of a backtest."""

    equity_curve: pd.Series
    daily_returns: pd.Series
    raw_predictions: pd.DataFrame
    features: List[str]
    summary: Dict[str, float]


def run_backtest(
    tickers: Iterable[str],
    start: str = "2018-01-01",
    end: Optional[str] = None,
    horizon: int = 5,
    n_splits: int = 6,
    quantile: float = 0.1,
    target_vol: float = 0.10,
    max_depth: int = 4,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    subsample: float = 0.7,
    colsample_bytree: float = 0.7,
    random_state: int = 42,
    include_macro: bool = True,
    macro_series: Optional[Dict[str, str]] = None,
) -> BacktestResult:
    """Execute the full pipeline and return backtest results.

    Parameters
    ----------
    tickers : Iterable[str]
        List of ticker symbols to include in the universe.
    start : str, optional
        Start date for the price history, by default "2018-01-01".
    end : str, optional
        End date for the price history.  If ``None``, uses the current
        date.
    horizon : int, optional
        Forward return horizon (in business days) for the target
        variable, by default 5.
    n_splits : int, optional
        Number of walk‑forward folds, by default 6.
    quantile : float, optional
        Top/bottom quantile threshold for determining long/short
        positions, by default 0.1 (top 10% longs, bottom 10% shorts).
    target_vol : float, optional
        Annualised volatility target for position sizing, by default
        10%.
    max_depth, n_estimators, learning_rate, subsample, colsample_bytree,
    random_state : model hyperparameters
        Parameters for the XGBoost regressor.

    Returns
    -------
    BacktestResult
        An object containing the equity curve, raw predictions and
        summary statistics.

    Raises
    ------
    ValueError
        If the xgboost package is not available.
    """
    if XGBRegressor is None:
        raise ValueError(
            "XGBoost is required for this backtest. Please install the `xgboost` package."
        )
    # 1. Fetch price data
    price_df = fetch_price_data(tickers, start=start, end=end)
    # 2. Optionally fetch macroeconomic data
    macro_df = None
    if include_macro:
        try:
            macro_df = fetch_macro_data(start=start, end=end, series=macro_series)
        except Exception:
            macro_df = None
    # 3. Build dataset including macro features
    df, feature_cols = build_dataset(price_df, horizon=horizon, macro_df=macro_df)
    # Sort by date for reproducibility
    df = df.sort_values("Date").reset_index(drop=True)
    # Prepare container for predictions
    all_preds = []  # type: List[pd.DataFrame]
    all_returns = []  # type: List[pd.DataFrame]
    # 3. Walk‑forward cross validation
    for fold_idx, (train_idx, test_idx) in enumerate(time_series_cv(df, n_splits=n_splits)):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        # Extract features and target
        X_train = train_df[feature_cols]
        y_train = train_df["y"]
        X_test = test_df[feature_cols]
        # Fit model
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        # Predict on test set
        preds = model.predict(X_test)
        # Create a copy of test_df to hold predictions
        test_pred_df = test_df.copy()
        test_pred_df["pred"] = preds
        all_preds.append(test_pred_df)
    # Concatenate all predictions and sort by date
    pred_df = pd.concat(all_preds).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    # 4. Generate positions based on quantiles
    daily_positions = []
    unique_dates = pred_df["Date"].unique()
    # Use volatility estimate from training returns of each fold
    # For simplicity, compute a global volatility estimate from the entire dataset
    # (More refined approaches could compute per‑fold estimates)
    overall_vol = df["y"].std() * np.sqrt(252)
    if overall_vol == 0 or np.isnan(overall_vol):
        overall_vol = 1.0  # avoid division by zero
    leverage = target_vol / overall_vol
    for d in unique_dates:
        day_slice = pred_df[pred_df["Date"] == d]
        # Determine thresholds for long and short
        q_hi = day_slice["pred"].quantile(1.0 - quantile)
        q_lo = day_slice["pred"].quantile(quantile)
        # Assign weights: 1 for long, -1 for short, 0 otherwise
        weights = np.where(
            day_slice["pred"] >= q_hi,
            1.0,
            np.where(day_slice["pred"] <= q_lo, -1.0, 0.0),
        )
        # Construct DataFrame with weights and future returns
        day_positions = day_slice[["Date", "Ticker", "y"]].copy()
        day_positions["w"] = weights
        # Apply leverage (volatility targeting)
        day_positions["strategy_ret"] = day_positions["y"] * day_positions["w"] * leverage
        daily_positions.append(day_positions)
    # Combine daily positions
    pos_df = pd.concat(daily_positions).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    # 5. Aggregate returns per day across tickers (equal weight per position)
    # We assume equal weight per non‑zero position; zeros contribute nothing
    # Compute the average return across all tickers (including zeros) by grouping by date
    daily_ret = pos_df.groupby("Date")["strategy_ret"].mean()
    # 6. Compute equity curve
    equity_curve = (1.0 + daily_ret).cumprod()
    # Give the equity curve a descriptive name rather than inheriting
    # the original column name (e.g. 'strategy_ret')
    equity_curve.name = "equity_curve"
    # Compute summary statistics
    n_days = len(daily_ret)
    total_return = equity_curve.iloc[-1] - 1.0
    cagr = equity_curve.iloc[-1] ** (252 / n_days) - 1.0 if n_days > 0 else np.nan
    # Daily sharpe ratio using sqrt(252)
    sharpe = (
        (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        if daily_ret.std() > 0
        else np.nan
    )
    # Maximum drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    max_drawdown = drawdown.min()
    summary = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "annual_volatility_target": float(target_vol),
        "quantile": float(quantile),
    }
    # Wrap results in dataclass
    result = BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_ret,
        raw_predictions=pred_df,
        features=feature_cols,
        summary=summary,
    )
    return result


def main() -> None:
    """Run a basic demonstration when executed as a script."""
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
    result = run_backtest(tickers, start="2018-01-01", horizon=5, n_splits=6)
    # Print summary statistics
    print("Backtest summary statistics:")
    for k, v in result.summary.items():
        print(f"  {k}: {v:.4f}")
    # Show the tail of the equity curve
    print("\nEquity curve tail:")
    print(result.equity_curve.tail())


if __name__ == "__main__":  # pragma: no cover
    main()