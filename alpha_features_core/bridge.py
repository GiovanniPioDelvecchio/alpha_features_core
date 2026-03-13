from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional


import alpha_features_core as _core
from tqdm import tqdm

# ---------------------------------------------------------------------------
def compute_alphas_cpp(
    df: pd.DataFrame,
    nums: Optional[List[int]] = None,
    *,
    verbose: bool = True,
    open_col:   str = "open",
    high_col:   str = "high",
    low_col:    str = "low",
    close_col:  str = "close",
    volume_col: str = "volume",
    amount_col: str = "amount",
    return_col: str = "past_return",
    date_col:   str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Compute Alpha191 factors using the C++ backend.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format OHLCV DataFrame sorted by [ticker, date].
    nums : list[int] | None
        Alpha numbers to compute (1-191).  None → compute all.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns [date, ticker, alpha001, alpha002, ...].
    """
    tickers = df[ticker_col].unique()
    dates   = df[date_col].unique()

    # Build pivot matrices  (tickers × dates)
    def _pivot(col: str) -> np.ndarray:
        p = df.pivot(index=ticker_col, columns=date_col, values=col)
        p = p.reindex(index=tickers, columns=dates)
        return p.values.astype(np.float64)

    calc = _core.Alpha191(
        open   = _pivot(open_col),
        high   = _pivot(high_col),
        low    = _pivot(low_col),
        close  = _pivot(close_col),
        volume = _pivot(volume_col),
        amount = _pivot(amount_col),
        returns= _pivot(return_col),
    )

    # Compute requested alphas
    if nums is None:
        results: dict[str, np.ndarray] = calc.calculate_all()
    else:
        iterator = tqdm(nums, desc="C++ alphas", unit="alpha") if verbose else nums
        results = {f"alpha{n:03d}": calc.calculate(n) for n in iterator}

    if not results:
        return pd.DataFrame(columns=[ticker_col, date_col])

    # ---- Vectorized unstack to long format ----
    alphas_list = sorted(results.keys())
    n_tickers   = len(tickers)
    n_dates     = len(dates)
    n_alphas    = len(alphas_list)

    # 3D array: tickers × dates × alphas
    all_data = np.empty((n_tickers, n_dates, n_alphas), dtype=np.float64)
    for i, name in enumerate(alphas_list):
        all_data[:, :, i] = results[name]

    # Flatten to long
    flat_data = all_data.reshape(n_tickers * n_dates, n_alphas)
    result_df = pd.DataFrame(flat_data, columns=alphas_list)

    # Add ticker/date columns
    ticker_idx = np.repeat(tickers, n_dates)
    date_idx   = np.tile(dates, n_tickers)
    result_df[ticker_col] = ticker_idx
    result_df[date_col]   = date_idx

    # Reorder columns
    cols = [ticker_col, date_col] + alphas_list
    result_df = result_df[cols]

    return result_df