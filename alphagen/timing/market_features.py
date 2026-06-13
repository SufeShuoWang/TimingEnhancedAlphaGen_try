from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from alphagen_qlib.stock_data import StockData


@dataclass
class MarketFeatures:
    values: np.ndarray
    names: List[str]


class MarketFeatureBuilder:
    """Build MASTER-style CSI300 market features from an index csv file."""

    def __init__(
        self,
        index_csv_path: str,
        lookbacks: Sequence[int] = (5, 10, 20, 30, 60),
        standardize: bool = True,
        include_current: bool = True,
    ) -> None:
        self.index_csv_path = Path(index_csv_path)
        self.lookbacks = tuple(int(x) for x in lookbacks)
        self.standardize = standardize
        self.include_current = include_current
        self.feature_names = self._make_feature_names()
        self._feature_frame: Optional[pd.DataFrame] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, stock_data: StockData) -> "MarketFeatureBuilder":
        raw = self._build_raw(stock_data)
        self._mean = np.nanmean(raw, axis=0)
        self._std = np.nanstd(raw, axis=0)
        self._mean[~np.isfinite(self._mean)] = 0.0
        self._std[~np.isfinite(self._std) | (self._std < 1e-12)] = 1.0
        return self

    def transform(self, stock_data: StockData) -> MarketFeatures:
        raw = self._build_raw(stock_data)
        if self.standardize:
            if self._mean is None or self._std is None:
                raise RuntimeError("MarketFeatureBuilder.fit must be called before transform")
            raw = (raw - self._mean[None, :]) / self._std[None, :]
        raw[~np.isfinite(raw)] = 0.0
        return MarketFeatures(values=raw, names=list(self.feature_names))

    def fit_transform(self, stock_data: StockData) -> MarketFeatures:
        return self.fit(stock_data).transform(stock_data)

    def _build_raw(self, stock_data: StockData) -> np.ndarray:
        feature_frame = self._load_feature_frame()
        signal_dates = self._stock_data_dates(stock_data)
        if not self.include_current:
            feature_frame = feature_frame.shift(1)

        aligned = feature_frame.reindex(feature_frame.index.union(signal_dates)).sort_index().ffill()
        aligned = aligned.reindex(signal_dates)
        if aligned.isna().any().any():
            missing = aligned.index[aligned.isna().any(axis=1)]
            first_missing = missing[0].strftime("%Y-%m-%d") if len(missing) else "unknown"
            raise ValueError(
                f"Market index csv cannot cover signal date {first_missing}. "
                f"Please provide a longer CSI300 index history."
            )
        return aligned.to_numpy(dtype=np.float64)

    def _load_feature_frame(self) -> pd.DataFrame:
        if self._feature_frame is not None:
            return self._feature_frame
        if not self.index_csv_path.exists():
            raise FileNotFoundError(f"Market index csv not found: {self.index_csv_path}")

        df = pd.read_csv(self.index_csv_path)
        lower_columns = {c.lower(): c for c in df.columns}
        for required in ("date", "close", "volume"):
            if required not in lower_columns:
                raise ValueError(f"Market index csv must contain a {required!r} column")

        data = pd.DataFrame({
            "date": pd.to_datetime(df[lower_columns["date"]]),
            "close": pd.to_numeric(df[lower_columns["close"]], errors="coerce"),
            "volume": pd.to_numeric(df[lower_columns["volume"]], errors="coerce"),
        }).dropna(subset=["date"]).sort_values("date")
        data = data.drop_duplicates(subset=["date"], keep="last").set_index("date")

        features = pd.DataFrame(index=data.index)
        features["close"] = data["close"]
        for window in self.lookbacks:
            rolling_close = data["close"].rolling(window=window, min_periods=1)
            rolling_volume = data["volume"].rolling(window=window, min_periods=1)
            features[f"close_mean_{window}"] = rolling_close.mean()
            features[f"close_std_{window}"] = rolling_close.std(ddof=0).fillna(0.0)
            features[f"volume_mean_{window}"] = rolling_volume.mean()
            features[f"volume_std_{window}"] = rolling_volume.std(ddof=0).fillna(0.0)

        self._feature_frame = features[self.feature_names]
        return self._feature_frame

    def _make_feature_names(self) -> List[str]:
        names = ["close"]
        for window in self.lookbacks:
            names.extend([
                f"close_mean_{window}",
                f"close_std_{window}",
                f"volume_mean_{window}",
                f"volume_std_{window}",
            ])
        return names

    @staticmethod
    def _stock_data_dates(stock_data: StockData) -> pd.DatetimeIndex:
        dates = stock_data._dates  # Qlib calendar already includes backtrack and future buffers.
        start = stock_data.max_backtrack_days
        stop = len(dates) - stock_data.max_future_days
        if stock_data.max_future_days == 0:
            stop = len(dates)
        return pd.DatetimeIndex(dates[start:stop])
