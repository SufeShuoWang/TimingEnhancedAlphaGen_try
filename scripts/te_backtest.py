import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest import QlibBacktest, dump_pickle
from alphagen.data.expression import Expression
from alphagen.timing import Gate, MarketEncoder, MarketFeatureBuilder
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from alphagen_qlib.utils import load_alpha_pool_by_path


DEFAULT_QLIB_DATA_PATH = os.environ.get(
    "ALPHAGEN_QLIB_DATA_PATH",
    "/home/stu_9/projects/repro_alphagen/cn_data_baostock_alphagen",
)
DEFAULT_MARKET_INDEX_CSV = os.environ.get(
    "ALPHAGEN_MARKET_INDEX_CSV",
    str(PROJECT_ROOT / "market_data" / "SH000300.csv"),
)


def _load_pool_json(pool_path: str) -> Dict:
    with open(pool_path, encoding="utf-8") as f:
        return json.load(f)


def _load_state_dict(module: torch.nn.Module, raw_state: Dict, device: torch.device) -> None:
    if raw_state is None:
        raise ValueError("Pool json does not contain the required timing state dict")
    state = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in raw_state.items()
    }
    module.load_state_dict(state)


def _stack_alpha_values(
    calculator: QLibStockDataCalculator,
    exprs: List[Expression],
    device: torch.device,
) -> torch.Tensor:
    values = [calculator.evaluate_alpha(expr) for expr in exprs]
    return torch.stack(values, dim=-1).to(device)


def _make_dynamic_prediction(
    data: StockData,
    calculator: QLibStockDataCalculator,
    exprs: List[Expression],
    weights: List[float],
    market_features: np.ndarray,
    encoder: MarketEncoder,
    gate: Gate,
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    factors = _stack_alpha_values(calculator, exprs, device)
    finite = torch.isfinite(factors).all(dim=-1)
    factors_clean = torch.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    market_tensor = torch.tensor(market_features, dtype=torch.float32, device=device)
    base_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    with torch.no_grad():
        gate_values = gate(encoder(market_tensor))
        dynamic_weights = base_weights[None, :] * gate_values
        score = torch.einsum("dsk,dk->ds", factors_clean, dynamic_weights)
        score[~finite] = torch.nan

    prediction = data.make_dataframe(score)
    return (
        prediction,
        gate_values.detach().cpu().numpy(),
        dynamic_weights.detach().cpu().numpy(),
    )


def _write_timing_weights(
    output_path: str,
    data: StockData,
    exprs: List[Expression],
    weights: List[float],
    gate_values: np.ndarray,
    dynamic_weights: np.ndarray,
) -> None:
    dates = pd.DatetimeIndex(
        data._dates[data.max_backtrack_days:len(data._dates) - data.max_future_days]
        if data.max_future_days > 0 else
        data._dates[data.max_backtrack_days:]
    )
    fieldnames = (
        ["date"]
        + [f"gate_alpha_{i}" for i in range(len(exprs))]
        + [f"weight_alpha_{i}" for i in range(len(exprs))]
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, date in enumerate(dates):
            row = {"date": pd.Timestamp(date).strftime("%Y-%m-%d")}
            for i in range(len(exprs)):
                row[f"gate_alpha_{i}"] = float(gate_values[t, i])
                row[f"weight_alpha_{i}"] = float(dynamic_weights[t, i])
            writer.writerow(row)


def _write_timing_summary(
    output_path: str,
    exprs: List[Expression],
    weights: List[float],
    gate_values: np.ndarray,
    dynamic_weights: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha_idx",
                "expr",
                "base_weight_w0",
                "mean_gate",
                "std_gate",
                "min_gate",
                "max_gate",
                "mean_dynamic_weight",
                "std_dynamic_weight",
                "mean_abs_dynamic_weight",
            ],
        )
        writer.writeheader()
        for i, expr in enumerate(exprs):
            writer.writerow({
                "alpha_idx": i,
                "expr": str(expr),
                "base_weight_w0": float(weights[i]),
                "mean_gate": float(np.nanmean(gate_values[:, i])),
                "std_gate": float(np.nanstd(gate_values[:, i])),
                "min_gate": float(np.nanmin(gate_values[:, i])),
                "max_gate": float(np.nanmax(gate_values[:, i])),
                "mean_dynamic_weight": float(np.nanmean(dynamic_weights[:, i])),
                "std_dynamic_weight": float(np.nanstd(dynamic_weights[:, i])),
                "mean_abs_dynamic_weight": float(np.nanmean(np.abs(dynamic_weights[:, i]))),
            })


def run(
    pool_path: str,
    qlib_data_path: str = DEFAULT_QLIB_DATA_PATH,
    market_index_csv: str = DEFAULT_MARKET_INDEX_CSV,
    instruments: str = "csi300",
    train_start: str = "2012-01-01",
    train_end: str = "2021-12-31",
    test_start: str = "2022-01-01",
    test_end: str = "2023-06-30",
    top_k: int = 50,
    n_drop: int = 5,
    benchmark: str = "SH000300",
    deal: str = "close",
    open_cost: float = 0.0015,
    close_cost: float = 0.0015,
    min_cost: float = 5,
    device: str = "cuda:0",
    state_include_current: bool = True,
    output_prefix: Optional[str] = None,
    run_static_baseline: bool = True,
):
    """Run a full v1.5 market-gated Qlib backtest."""

    initialize_qlib(qlib_data_path)
    device_obj = torch.device(device)
    raw_pool = _load_pool_json(pool_path)
    timing = raw_pool.get("timing")
    if timing is None:
        raise ValueError("pool_path does not look like a v1.5 pool json: missing 'timing'")

    exprs, weights = load_alpha_pool_by_path(pool_path)
    if len(exprs) == 0:
        raise ValueError("pool contains no expressions")

    embedding_dim = int(timing.get("market_embedding_dim", 16))
    gate_temperature = float(timing.get("gate_temperature", 1.0))
    encoder = MarketEncoder(d_input=21, d_model=embedding_dim).to(device_obj)
    gate = Gate(d_input=embedding_dim, d_output=len(exprs), beta=gate_temperature).to(device_obj)
    _load_state_dict(encoder, timing.get("market_encoder_state"), device_obj)
    _load_state_dict(gate, timing.get("alpha_gate_state"), device_obj)
    encoder.eval()
    gate.eval()

    data_train = StockData(instruments, train_start, train_end, device=device_obj)
    data_test = StockData(instruments, test_start, test_end, device=device_obj)
    calc_test = QLibStockDataCalculator(data_test, None)

    market_builder = MarketFeatureBuilder(
        index_csv_path=market_index_csv,
        standardize=True,
        include_current=state_include_current,
    )
    market_builder.fit(data_train)
    market_test = market_builder.transform(data_test)
    if len(market_test.names) != 21:
        raise ValueError(f"Expected 21 market features, got {len(market_test.names)}")

    prediction, gate_values, dynamic_weights = _make_dynamic_prediction(
        data=data_test,
        calculator=calc_test,
        exprs=exprs,
        weights=weights,
        market_features=market_test.values,
        encoder=encoder,
        gate=gate,
        device=device_obj,
    )

    if output_prefix is None:
        pool_name = Path(pool_path).stem.replace("_pool", "")
        output_prefix = f"out/backtests/te15_dynamic/{pool_name}"

    dump_pickle(output_prefix + "-prediction.pkl", lambda: prediction, True)
    _write_timing_weights(
        output_prefix + "-timing-daily-weights.csv",
        data_test,
        exprs,
        weights,
        gate_values,
        dynamic_weights,
    )
    _write_timing_summary(
        output_prefix + "-timing-summary.csv",
        exprs,
        weights,
        gate_values,
        dynamic_weights,
    )

    qlib_backtest = QlibBacktest(
        benchmark=benchmark,
        top_k=top_k,
        n_drop=n_drop,
        deal=deal,
        open_cost=open_cost,
        close_cost=close_cost,
        min_cost=min_cost,
    )
    _, dynamic_result = qlib_backtest.run(prediction, output_prefix=output_prefix)

    static_result = None
    if run_static_baseline:
        static_prediction = data_test.make_dataframe(calc_test.make_ensemble_alpha(exprs, weights))
        static_prefix = output_prefix + "_static_w0"
        dump_pickle(static_prefix + "-prediction.pkl", lambda: static_prediction, True)
        _, static_result = qlib_backtest.run(static_prediction, output_prefix=static_prefix)

    summary = {
        "pool_path": pool_path,
        "qlib_data_path": qlib_data_path,
        "market_index_csv": market_index_csv,
        "instruments": instruments,
        "train": [train_start, train_end],
        "test": [test_start, test_end],
        "top_k": top_k,
        "n_drop": n_drop,
        "benchmark": benchmark,
        "dynamic_result": dynamic_result.to_dict(),
        "static_w0_result": None if static_result is None else static_result.to_dict(),
        "output_prefix": output_prefix,
    }
    with open(output_prefix + "-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    fire.Fire(run)
