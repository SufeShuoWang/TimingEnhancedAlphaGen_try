import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen.data.expression import Feature, Ref
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import FeatureType, StockData, initialize_qlib


DEFAULT_QLIB_DATA_PATH = os.environ.get(
    "ALPHAGEN_QLIB_DATA_PATH",
    "/home/stu_9/projects/repro_alphagen/cn_data_baostock_alphagen",
)


def _feature_types() -> List[FeatureType]:
    return [
        FeatureType.OPEN,
        FeatureType.CLOSE,
        FeatureType.HIGH,
        FeatureType.LOW,
        FeatureType.VOLUME,
        FeatureType.VWAP,
    ]


def _date_index(data: StockData) -> pd.Index:
    start = data.max_backtrack_days
    end = start + data.n_days
    return pd.Index(data._dates[start:end], name="datetime")  # type: ignore[attr-defined]


def _evaluate_expr(expr, data: StockData, normalize: bool) -> np.ndarray:
    value = expr.evaluate(data)
    if normalize:
        value = normalize_by_day(value)
    return value.detach().cpu().numpy()


def _build_xy(
    data: StockData,
    horizon: int,
    lookback: int,
    normalize_x_by_day: bool,
    normalize_y_by_day: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index, List[str]]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    feature_arrays: List[np.ndarray] = []
    feature_names: List[str] = []
    for lag in range(lookback):
        for feature_type in _feature_types():
            base_expr = Feature(feature_type)
            expr = base_expr if lag == 0 else Ref(base_expr, lag)
            feature_arrays.append(_evaluate_expr(expr, data, normalize_x_by_day))
            feature_names.append(f"{feature_type.name.lower()}_lag{lag}")

    x = np.stack(feature_arrays, axis=2)
    close = Feature(FeatureType.CLOSE)
    target_expr = Ref(close, -horizon) / close - 1
    y = _evaluate_expr(target_expr, data, normalize_y_by_day)

    finite = np.isfinite(y) & np.isfinite(x).all(axis=2)
    return x, y, finite, _date_index(data), data.stock_ids, feature_names


def _flatten_valid(
    x: np.ndarray,
    y: np.ndarray,
    finite: np.ndarray,
    sample_limit: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1)
    finite_flat = finite.reshape(-1)
    valid_idx = np.flatnonzero(finite_flat)
    if sample_limit is not None and sample_limit > 0 and len(valid_idx) > sample_limit:
        rng = np.random.default_rng(seed)
        valid_idx = rng.choice(valid_idx, size=sample_limit, replace=False)
    return x_flat[valid_idx], y_flat[valid_idx]


def _daily_corr(pred: np.ndarray, target: np.ndarray, rank: bool = False) -> float:
    values = []
    for pred_row, target_row in zip(pred, target):
        mask = np.isfinite(pred_row) & np.isfinite(target_row)
        if mask.sum() < 2:
            continue
        x = pred_row[mask]
        y = target_row[mask]
        if rank:
            x = pd.Series(x).rank(method="average").to_numpy()
            y = pd.Series(y).rank(method="average").to_numpy()
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def _predict_3d(model, x: np.ndarray, finite: np.ndarray) -> np.ndarray:
    pred = np.full(finite.shape, np.nan, dtype=np.float64)
    x_flat = x.reshape(-1, x.shape[-1])
    finite_flat = finite.reshape(-1)
    pred.reshape(-1)[finite_flat] = model.predict(x_flat[finite_flat])
    return pred


def _build_model(model_name: str, seed: int, n_jobs: int):
    model_name = model_name.lower()
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it with: "
                "python -m pip install xgboost"
            ) from exc
        return XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name in {"gbm", "lightgbm", "lgbm"}:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if model_name == "ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge

        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=2048,
                learning_rate_init=1e-3,
                max_iter=30,
                early_stopping=True,
                random_state=seed,
                verbose=False,
            ),
        )
    raise ValueError("model must be one of: gbm, lightgbm, xgboost, ridge, mlp")


def _feature_importance(model, feature_names: List[str], top_k: int = 30) -> List[Dict]:
    raw_model = model
    if hasattr(model, "steps"):
        raw_model = model.steps[-1][1]
    values = getattr(raw_model, "feature_importances_", None)
    if values is None:
        coef = getattr(raw_model, "coef_", None)
        values = np.abs(coef) if coef is not None else None
    if values is None:
        return []
    values = np.asarray(values).reshape(-1)
    order = np.argsort(values)[::-1][:top_k]
    return [
        {"feature": feature_names[int(i)], "importance": float(values[int(i)])}
        for i in order
    ]


def _evaluate_segment(
    model,
    data: StockData,
    horizon: int,
    lookback: int,
    normalize_x_by_day: bool,
    normalize_y_by_day: bool,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, pd.Index]]:
    x, y, finite, dates, stocks, _ = _build_xy(
        data=data,
        horizon=horizon,
        lookback=lookback,
        normalize_x_by_day=normalize_x_by_day,
        normalize_y_by_day=normalize_y_by_day,
    )
    pred = _predict_3d(model, x, finite)
    metrics = {
        "ic": _daily_corr(pred, y, rank=False),
        "rank_ic": _daily_corr(pred, y, rank=True),
        "n_days": int(data.n_days),
        "n_stocks": int(data.n_stocks),
        "n_valid_samples": int(finite.sum()),
    }
    return metrics, (pred, y, finite, dates, stocks)


def _save_predictions(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    finite: np.ndarray,
    dates: pd.Index,
    stocks: pd.Index,
) -> None:
    date_values = np.repeat(dates.astype(str).to_numpy(), len(stocks))
    stock_values = np.tile(stocks.astype(str).to_numpy(), len(dates))
    flat_mask = finite.reshape(-1)
    df = pd.DataFrame(
        {
            "datetime": date_values[flat_mask],
            "instrument": stock_values[flat_mask],
            "prediction": pred.reshape(-1)[flat_mask],
            "target": target.reshape(-1)[flat_mask],
        }
    )
    df.to_csv(path, index=False)


def main(
    model: str = "gbm",
    qlib_data_path: str = DEFAULT_QLIB_DATA_PATH,
    instruments: str = "csi300",
    train_start: str = "2012-01-01",
    train_end: str = "2021-12-31",
    test1_start: str = "2022-01-01",
    test1_end: str = "2022-06-30",
    test2_start: str = "2022-07-01",
    test2_end: str = "2022-12-31",
    test3_start: str = "2023-01-01",
    test3_end: str = "2023-06-30",
    horizon: int = 20,
    lookback: int = 20,
    train_sample_limit: Optional[int] = 300000,
    normalize_x_by_day: bool = True,
    normalize_y_by_day: bool = True,
    seed: int = 0,
    n_jobs: Optional[int] = None,
    output_dir: str = "out/ml_baselines",
    save_predictions: bool = False,
) -> Dict:
    """
    Train tabular ML baselines on the same Qlib data and IC metrics.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    initialize_qlib(qlib_data_path)

    device = torch.device("cpu")
    train_data = StockData(instruments, train_start, train_end, device=device)
    x_train, y_train, finite_train, _, _, feature_names = _build_xy(
        data=train_data,
        horizon=horizon,
        lookback=lookback,
        normalize_x_by_day=normalize_x_by_day,
        normalize_y_by_day=normalize_y_by_day,
    )
    train_x, train_y = _flatten_valid(
        x_train,
        y_train,
        finite_train,
        sample_limit=train_sample_limit,
        seed=seed,
    )

    estimator = _build_model(model, seed=seed, n_jobs=n_jobs)
    estimator.fit(train_x, train_y)

    test_ranges = [
        ("test_1", test1_start, test1_end),
        ("test_2", test2_start, test2_end),
        ("test_3", test3_start, test3_end),
    ]
    test_results = {}
    prediction_payloads = {}
    total_days = 0
    weighted_ic = 0.0
    weighted_rank_ic = 0.0

    for name, start, end in test_ranges:
        segment_data = StockData(instruments, start, end, device=device)
        metrics, payload = _evaluate_segment(
            model=estimator,
            data=segment_data,
            horizon=horizon,
            lookback=lookback,
            normalize_x_by_day=normalize_x_by_day,
            normalize_y_by_day=normalize_y_by_day,
        )
        test_results[name] = metrics
        total_days += metrics["n_days"]
        weighted_ic += metrics["ic"] * metrics["n_days"]
        weighted_rank_ic += metrics["rank_ic"] * metrics["n_days"]
        prediction_payloads[name] = payload

    test_results["mean"] = {
        "ic": weighted_ic / total_days,
        "rank_ic": weighted_rank_ic / total_days,
        "n_days": total_days,
    }

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = f"{model.lower()}_{instruments}_lb{lookback}_h{horizon}_{timestamp}"
    out_dir = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "summary.json"

    result = {
        "config": {
            "model": model,
            "qlib_data_path": qlib_data_path,
            "instruments": instruments,
            "train": [train_start, train_end],
            "tests": [[s, e] for _, s, e in test_ranges],
            "horizon": horizon,
            "lookback": lookback,
            "train_sample_limit": train_sample_limit,
            "normalize_x_by_day": normalize_x_by_day,
            "normalize_y_by_day": normalize_y_by_day,
            "seed": seed,
            "n_jobs": n_jobs,
        },
        "train": {
            "n_days": int(train_data.n_days),
            "n_stocks": int(train_data.n_stocks),
            "n_valid_samples": int(finite_train.sum()),
            "n_fit_samples": int(len(train_y)),
            "n_features": int(train_x.shape[1]),
        },
        "test": test_results,
        "feature_importance": _feature_importance(estimator, feature_names),
        "output": {
            "summary_path": str(result_path),
        },
    }

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if save_predictions:
        for name, payload in prediction_payloads.items():
            pred, target, finite, dates, stocks = payload
            _save_predictions(out_dir / f"{name}_predictions.csv", pred, target, finite, dates, stocks)

    print(json.dumps(result, indent=2))
    return None


if __name__ == "__main__":
    fire.Fire(main)
