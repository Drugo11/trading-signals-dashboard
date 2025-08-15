"""
generate_signals_json.py
------------------------

Utility script to generate a JSON file of trading signals from the
machine‑learning backtest.  The script uses the `run_backtest`
function defined in ``trading_system.py`` to compute predictions for
the specified universe of tickers.  It then classifies each
(date, ticker) pair as Long (+1), Short (–1) or Flat (0) based on
quantile thresholds and saves the results to ``signals.json``.

To update the signals, run this script from the repository root:

```
python trading_system/generate_signals_json.py
```

The resulting file ``signals.json`` will be created in the same
directory.  You can load it into the HTML app to visualise the
signals.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np

# Import run_backtest from the sibling module.  Adjust sys.path so that
# Python can locate the trading_system package when this script is
# executed directly (rather than as part of the package).
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
# Ensure the parent directory is on sys.path so 'trading_system' can be found
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

from trading_system.trading_system import run_backtest


def classify_signals(
    raw_predictions,
    quantile: float,
) -> List[Dict[str, Any]]:
    """Convert continuous predictions into discrete trading signals.

    Parameters
    ----------
    raw_predictions : pd.DataFrame
        DataFrame with columns ['Date', 'Ticker', 'pred'].
    quantile : float
        Fraction of the distribution to use for long/short
        thresholds (e.g. 0.1 means top 10% long and bottom 10%
        short).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with keys ``date``, ``ticker``,
        ``prediction`` and ``signal`` (1 for long, –1 for short, 0
        otherwise).
    """
    signals: List[Dict[str, Any]] = []
    # Ensure sorted by date so unique() returns in order
    df = raw_predictions.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    dates = df["Date"].unique()
    for d in dates:
        day_slice = df[df["Date"] == d]
        q_hi = day_slice["pred"].quantile(1.0 - quantile)
        q_lo = day_slice["pred"].quantile(quantile)
        for row in day_slice.itertuples():
            if row.pred >= q_hi:
                sig = 1
            elif row.pred <= q_lo:
                sig = -1
            else:
                sig = 0
            signals.append(
                {
                    "date": str(row.Date.date()),
                    "ticker": row.Ticker,
                    "prediction": float(row.pred),
                    "signal": int(sig),
                }
            )
    return signals


def main(
    tickers: Iterable[str] = (
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
    ),
    start: str = "2018-01-01",
    horizon: int = 5,
    n_splits: int = 6,
    quantile: float = 0.1,
    output_file: Path = Path(__file__).with_name("signals.json"),
) -> None:
    """Generate signals and write them to a JSON file.

    Parameters
    ----------
    tickers : Iterable[str], optional
        Universe of tickers, by default the same as used in the demo.
    start : str, optional
        Start date for the backtest, by default "2018-01-01".
    horizon : int, optional
        Forward return horizon, by default 5.
    n_splits : int, optional
        Number of walk‑forward folds, by default 6.
    quantile : float, optional
        Quantile threshold for classification, by default 0.1.
    output_file : Path, optional
        Where to write the JSON output, by default ``signals.json`` in
        the script directory.
    """
    result = run_backtest(
        tickers=tickers,
        start=start,
        horizon=horizon,
        n_splits=n_splits,
        quantile=quantile,
    )
    signals = classify_signals(result.raw_predictions[["Date", "Ticker", "pred"]], quantile)
    # Write JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)
    # In addition to the JSON file, emit a JavaScript file for the web app.
    # The JS file exports the signals as a constant so that it can be loaded via a <script> tag.
    js_output_file = Path(__file__).with_name("signals.js")
    with open(js_output_file, "w", encoding="utf-8") as f_js:
        f_js.write("const signals = ")
        # Write the JSON array without indentation to reduce file size
        json.dump(signals, f_js)
        f_js.write(";")
    print(f"Signals written to {output_file} and {js_output_file}")


if __name__ == "__main__":  # pragma: no cover
    main()