# alpha-features-core

Fast C++ implementation of the [Alpha191](https://www.joinquant.com/help/api/help#name:alpha191) quantitative finance factor library, with a pure-Python/pandas fallback.

## Installation

```bash
pip install alpha-features-core
```

Building from source requires **CMake ≥ 3.15** and a C++17 compiler (MSVC on Windows, GCC/Clang on Linux/macOS).

## Quick start

```python
import pandas as pd
from alpha_features_core.bridge import compute_alphas_cpp   # C++ backend
from alpha_features_core.alpha191 import Alphas191          # pure-Python backend

# df is a long-format OHLCV DataFrame with columns:
#   ticker, date, open, high, low, close, volume, amount, past_return

# --- C++ backend (fast) ---
cpp_result = compute_alphas_cpp(df)          # all 191 factors
cpp_result = compute_alphas_cpp(df, nums=[1, 5, 10])  # subset

# --- Pure-Python backend ---
py_result = Alphas191(df).calculate_all_alphas(return_long=True)
```

Both functions return a long-format `pd.DataFrame` with columns `[ticker, date, alpha001, alpha002, ...]`.

## Performance

The C++ backend compiles all factors in a single pass over the data and significantly outperforms the pandas/numba implementation at scale.

```python
from alpha_features_core.bridge import compute_alphas_cpp
from alpha_features_core.alpha191 import Alphas191
import numpy as np, pandas as pd, time, matplotlib.pyplot as plt

EXPERIMENTS = [
    (5,  40), (10, 50), (20, 50), (30, 70), (50, 80),
    (80, 100), (100, 120), (150, 150), (200, 180),
    (250, 200), (300, 250), (350, 300), (500, 500),
]
ALL_NUMS = list(range(1, 192))
times_cpp, times_py, n_rows = [], [], []

for T, D in EXPERIMENTS:
    rng = np.random.default_rng(42)
    tickers = [f"T{i:04d}" for i in range(T)]
    dates   = pd.date_range("2020-01-01", periods=D, freq="B")
    idx     = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    n       = len(idx)
    close   = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "close":       close,
        "open":        close + rng.normal(0, 0.2, n),
        "high":        close + np.abs(rng.normal(0, 0.5, n)),
        "low":         close - np.abs(rng.normal(0, 0.5, n)),
        "volume":      np.abs(rng.normal(1e6, 1e5, n)),
        "past_return": rng.normal(0, 0.01, n),
    }, index=idx).reset_index()
    df["amount"] = df["close"] * df["volume"]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    t0 = time.perf_counter()
    compute_alphas_cpp(df, nums=ALL_NUMS, verbose=False)
    times_cpp.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    Alphas191(df).calculate_all_alphas(return_long=True)
    times_py.append(time.perf_counter() - t0)

    n_rows.append(n)
    print(f"{T}×{D} = {n:>7,} rows  C++: {times_cpp[-1]:.2f}s  "
          f"Python: {times_py[-1]:.2f}s  speedup: {times_py[-1]/times_cpp[-1]:.1f}x")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_rows, times_cpp, "o-", color="steelblue",  lw=2, ms=6, label="C++ backend")
ax.plot(n_rows, times_py,  "s-", color="darkorange", lw=2, ms=6, label="Pure Python (numba)")
ax.set_xlabel("Number of rows (tickers × dates)")
ax.set_ylabel("Execution time (s)")
ax.set_title("Alpha191: C++ vs Pure Python — scaling benchmark")
ax.legend()
plt.tight_layout()
plt.savefig("benchmark.png", dpi=150)
```

## Numerical differences between backends

The C++ and Python backends produce identical results for the vast majority of factors. Small numerical differences exist for factors that combine `Corr` and `Rank` operations (e.g. alpha001, alpha016, alpha090).

The root cause is a floating-point edge case in pandas' `rolling().corr()`: when one input series is constant within the rolling window, pandas internally produces `±inf` or `NaN` depending on accumulated floating-point rounding errors that vary per window. The C++ backend applies a deterministic `kEps` threshold and cannot replicate this non-deterministic behaviour exactly.

In practice the affected factors differ by at most 1–2 rank positions out of 100 tickers per date, which has negligible impact on factor signal quality.

The following factors are affected:
`alpha001`, `alpha005`, `alpha016`, `alpha036`, `alpha054`, `alpha056`,
`alpha061`, `alpha064`, `alpha073`, `alpha077`, `alpha083`, `alpha090`,
`alpha091`, `alpha092`, `alpha099`, `alpha101`, `alpha113`, `alpha115`,
`alpha119`, `alpha121`, `alpha123`, `alpha130`, `alpha131`, `alpha138`,
`alpha141`, `alpha148`, `alpha170`, `alpha176`, `alpha179`, `alpha191`.

All remaining 160+ factors match the Python backend within floating-point tolerance (`atol=1e-5`).

## Input DataFrame format

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Asset identifier |
| `date` | datetime | Trading date |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |
| `volume` | float | Trading volume |
| `amount` | float | Traded amount (close × volume) |
| `past_return` | float | Previous period return |

The DataFrame must be sorted by `[ticker, date]`. Column names can be customised via keyword arguments to `compute_alphas_cpp()`.

## License

MIT