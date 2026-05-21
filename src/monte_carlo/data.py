"""Price history loading for the bootstrap model.

Only the bootstrap model needs real data. The GBM model is fully parametric and
needs no download. As with the rest of these projects, the loader hides the data
vendor behind a small interface and caches results to parquet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/cache")


def _cache_key(ticker: str, start: str, end: str) -> str:
    raw = f"{ticker}|{start}|{end}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"price_{ticker}_{digest}.parquet"


class YFinanceLoader:
    """Adjusted close prices for one ticker, with parquet caching."""

    def __init__(
        self, cache_dir: Path | str = DEFAULT_CACHE_DIR, use_cache: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

    def load(self, ticker: str, start: str, end: str) -> pd.Series:
        cache_path = self.cache_dir / _cache_key(ticker, start, end)
        if self.use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)[ticker]

        prices = self._download(ticker, start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        prices.to_frame(name=ticker).to_parquet(cache_path)
        return prices

    @staticmethod
    def _download(ticker: str, start: str, end: str) -> pd.Series:
        import yfinance as yf

        raw = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError(
                f"No data returned for {ticker}. Check the ticker and dates."
            )
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        close = close.sort_index().dropna()
        close.name = ticker
        return close


def monthly_returns(prices: pd.Series) -> np.ndarray:
    """Month-end simple returns as a plain array, ready for the bootstrap."""
    monthly = prices.resample("ME").last()
    rets = monthly.pct_change().dropna()
    return rets.to_numpy()
