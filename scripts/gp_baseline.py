import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen.data.expression import *  # noqa: F401,F403
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.features import *  # noqa: F401,F403
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, initialize_qlib
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor


DEFAULT_QLIB_DATA_PATH = os.environ.get(
    "ALPHAGEN_QLIB_DATA_PATH",
    "/home/stu_9/projects/repro_alphagen/cn_data_baostock_alphagen",
)


def main(
    qlib_data_path: str = DEFAULT_QLIB_DATA_PATH,
    instruments: str = "csi300",
    seed: int = 0,
    train_start: str = "2012-01-01",
    train_end: str = "2021-12-31",
    test_start: str = "2022-01-01",
    test_end: str = "2023-06-30",
    capacity: int = 20,
    population_size: int = 1000,
    generations: int = 40,
    tournament_size: int = 600,
    mutual_ic_thres: Optional[float] = 0.7,
    max_token_pairs: int = 20,
    device: str = "cpu",
    output_dir: str = "out/gp_baselines",
):
    reseed_everything(seed)
    initialize_qlib(qlib_data_path)

    torch_device = torch.device(device)
    data_train = StockData(instruments, train_start, train_end, device=torch_device)
    data_test = StockData(instruments, test_start, test_end, device=torch_device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    cache = {}

    def to_expr(key: str):
        return eval(key, globals())

    def metric(_, y, __):
        key = y[0]
        if key in cache:
            return cache[key]
        if key.count("(") + key.count(")") > max_token_pairs:
            cache[key] = -1.0
            return -1.0
        try:
            value = calculator_train.calc_single_IC_ret(to_expr(key))
        except Exception:
            value = -1.0
        cache[key] = -1.0 if np.isnan(value) else float(value)
        return cache[key]

    def build_pool():
        pool = MseAlphaPool(capacity=capacity, calculator=calculator_train, ic_lower_bound=None)
        exprs = []
        for key, score in Counter(cache).most_common():
            if score <= -0.999:
                continue
            expr = to_expr(key)
            if mutual_ic_thres is not None:
                try:
                    too_similar = any(
                        abs(pool.calculator.calc_mutual_IC(old, expr)) > mutual_ic_thres
                        for old in exprs
                    )
                except Exception:
                    continue
                if too_similar:
                    continue
            exprs.append(expr)
            if len(exprs) >= capacity:
                break
        if not exprs:
            raise RuntimeError("GP did not find any valid expression; increase population_size or generations.")
        pool.force_load_exprs(exprs)
        return pool

    funcs = [make_function(**func._asdict()) for func in generic_funcs]
    terminals = ["open_", "close", "high", "low", "volume", "vwap"] + [
        f"Constant({v})" for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
    ]
    x_train = np.array([terminals])
    y_train = np.array([[1]])

    model = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        init_depth=(2, 6),
        tournament_size=tournament_size,
        stopping_criteria=1.0,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=funcs,
        metric=make_fitness(function=metric, greater_is_better=True),  # type: ignore
        const_range=None,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    pool = build_pool()
    ic_train, ric_train = pool.test_ensemble(calculator_train)
    ic_test, ric_test = pool.test_ensemble(calculator_test)

    run_dir = Path(output_dir) / f"{instruments}_{capacity}_{seed}_{datetime.now():%Y%m%d%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "qlib_data_path": qlib_data_path,
            "instruments": instruments,
            "seed": seed,
            "train": [train_start, train_end],
            "test": [test_start, test_end],
            "capacity": capacity,
            "population_size": population_size,
            "generations": generations,
            "tournament_size": tournament_size,
            "mutual_ic_thres": mutual_ic_thres,
            "max_token_pairs": max_token_pairs,
            "device": device,
        },
        "metrics": {
            "ic_train": float(ic_train),
            "rank_ic_train": float(ric_train),
            "ic_test": float(ic_test),
            "rank_ic_test": float(ric_test),
        },
        "pool_state": pool.to_json_dict(),
        "cache_size": len(cache),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
