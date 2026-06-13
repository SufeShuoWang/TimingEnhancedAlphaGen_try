import copy
import csv
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .linear_alpha_pool import LinearAlphaPool
from ..data.calculator import AlphaCalculator, TensorAlphaCalculator
from ..data.expression import Expression, OutOfDataRangeError
from ..data.pool_update import AddRemoveAlphas
from ..timing.market_encoder import Gate, MarketEncoder
from ..timing.market_features import MarketFeatureBuilder
from ..utils.correlation import batch_pearsonr, batch_spearmanr


def _state_dict_to_json(state: Optional[Dict[str, Tensor]]) -> Optional[Dict[str, Any]]:
    if state is None:
        return None
    return {key: value.detach().cpu().tolist() for key, value in state.items()}


class StateAwareMseAlphaPool(LinearAlphaPool):
    """Alpha pool whose reward is conditioned on a MASTER-style market gate."""

    def __init__(
        self,
        capacity: int,
        calculator: TensorAlphaCalculator,
        market_features: Union[np.ndarray, Tensor],
        market_feature_builder: Optional[MarketFeatureBuilder] = None,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        gate_l2: float = 1e-3,
        encoder_l2: float = 1e-4,
        embedding_dim: int = 16,
        gate_temperature: float = 1.0,
        train_lr: float = 1e-3,
        train_max_steps: int = 600,
        train_tolerance: int = 80,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(capacity, calculator, ic_lower_bound, device)
        self.tensor_calculator = calculator
        self.market_features = self._as_market_tensor(market_features)
        if self.market_features.shape[0] != calculator.n_days:
            raise ValueError(
                f"market_features days ({self.market_features.shape[0]}) must match "
                f"calculator.n_days ({calculator.n_days})"
            )
        self.market_feature_builder = market_feature_builder
        self._l1_alpha = float(l1_alpha)
        self._gate_l2 = float(gate_l2)
        self._encoder_l2 = float(encoder_l2)
        self._embedding_dim = int(embedding_dim)
        self._gate_temperature = float(gate_temperature)
        self._train_lr = float(train_lr)
        self._train_max_steps = int(train_max_steps)
        self._train_tolerance = int(train_tolerance)

        self._encoder_state: Optional[Dict[str, Tensor]] = None
        self._gate_state: Optional[Dict[str, Tensor]] = None
        self._state_size = 0
        self._last_dynamic_importance: Optional[np.ndarray] = None
        self._last_objective: Optional[float] = None
        self._last_loss: Optional[float] = None

    @property
    def state(self) -> Dict[str, Any]:
        state = super().state
        state["dynamic_importance"] = (
            None if self._last_dynamic_importance is None
            else list(self._last_dynamic_importance[:self.size])
        )
        state["objective"] = self.best_obj
        state["gate"] = {
            "embedding_dim": self._embedding_dim,
            "temperature": self._gate_temperature,
            "last_loss": self._last_loss,
        }
        return state

    def to_json_dict(self) -> Dict[str, Any]:
        data = super().to_json_dict()
        data["weights"] = [float(x) for x in self.weights]
        data["timing"] = {
            "version": "v1.5",
            "market_embedding_dim": self._embedding_dim,
            "gate": "MASTER softmax Gate",
            "gate_temperature": self._gate_temperature,
            "feature_names": (
                [] if self.market_feature_builder is None
                else list(self.market_feature_builder.feature_names)
            ),
            "dynamic_importance": (
                None if self._last_dynamic_importance is None
                else [float(x) for x in self._last_dynamic_importance[:self.size]]
            ),
            "best_objective": float(self.best_obj),
            "market_encoder_state": _state_dict_to_json(self._encoder_state),
            "alpha_gate_state": _state_dict_to_json(self._gate_state),
        }
        return data

    def export_timing_diagnostics(
        self,
        path_prefix: str,
        extra_calculators: Optional[Sequence[TensorAlphaCalculator]] = None,
    ) -> None:
        """Write readable alpha-gate diagnostics for the current checkpoint."""

        if self.size == 0:
            return
        if self._encoder_state is None or self._gate_state is None or self._state_size != self.size:
            self.weights = self.optimize()

        prefix = Path(path_prefix)
        self._write_timing_summary(prefix.with_name(prefix.name + "_timing_train_summary.csv"))
        self._write_daily_timing_weights(
            prefix.with_name(prefix.name + "_timing_train_daily_weights.csv"),
            self.tensor_calculator,
            self.market_features,
        )
        if extra_calculators is not None:
            for i, calculator in enumerate(extra_calculators, start=1):
                market_features = self._market_features_for_calculator(calculator)
                if market_features is None:
                    continue
                self._write_daily_timing_weights(
                    prefix.with_name(prefix.name + f"_timing_test_{i}_daily_weights.csv"),
                    calculator,
                    market_features,
                )

    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
            return 0.
        if str(expr) in self._failure_cache:
            return self.best_obj

        self.eval_cnt += 1
        old_pool: List[Expression] = self.exprs[:self.size]     # type: ignore
        old_pool_ic = self.best_ic_ret
        self._add_factor(expr, ic_ret, ic_mut)

        if self.size > 1:
            new_weights = self.optimize()
            worst_idx = None
            if self.size > self.capacity:
                worst_idx = self._get_worst_index(new_weights)
                if worst_idx == self.capacity:
                    self.weights = new_weights
                    self._pop(worst_idx)
                    if self.size > 0:
                        self.weights = self.optimize()
                    self._failure_cache.add(str(expr))
                    return self.best_obj

            self.weights = new_weights
            removed_idx = [worst_idx] if worst_idx is not None else []
            if worst_idx is not None:
                self._pop(worst_idx)
                if self.size > 0:
                    self.weights = self.optimize()

            new_ic, new_obj = self.calculate_ic_and_objective()
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=removed_idx,
                old_pool=old_pool,
                old_pool_ic=old_pool_ic,
                new_pool_ic=new_ic
            ))
        else:
            self.weights = self.optimize()
            new_ic, new_obj = self.calculate_ic_and_objective()
            self.update_history.append(AddRemoveAlphas(
                added_exprs=[expr],
                removed_idx=[],
                old_pool=[],
                old_pool_ic=0.,
                new_pool_ic=new_ic
            ))

        self._failure_cache = set()
        self._maybe_update_best(new_ic, new_obj)
        return new_obj

    def force_load_exprs(self, exprs: List[Expression], weights: Optional[List[float]] = None) -> None:
        self._failure_cache = set()
        old_ic = self.evaluate_ensemble()
        old_pool: List[Expression] = self.exprs[:self.size]  # type: ignore
        added = []
        for expr in exprs:
            if self.size >= self.capacity:
                break
            try:
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            except (OutOfDataRangeError, TypeError):
                continue
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret, ic_mut)
            added.append(expr)
            assert self.size <= self.capacity
        if weights is not None:
            if len(weights) != self.size:
                raise ValueError(f"Invalid weights length: got {len(weights)}, expected {self.size}")
            self.weights = np.array(weights)
        if self.size > 0:
            self.weights = self.optimize()
        new_ic, new_obj = self.calculate_ic_and_objective()
        self._maybe_update_best(new_ic, new_obj)
        self.update_history.append(AddRemoveAlphas(
            added_exprs=added,
            removed_idx=[],
            old_pool=old_pool,
            old_pool_ic=old_ic,
            new_pool_ic=new_ic
        ))

    def calculate_ic_and_objective(self) -> Tuple[float, float]:
        ic = self.evaluate_ensemble()
        return ic, ic

    def optimize(
        self,
        lr: Optional[float] = None,
        max_steps: Optional[int] = None,
        tolerance: Optional[int] = None
    ) -> np.ndarray:
        if self.size == 0:
            return np.zeros(0, dtype=np.float64)

        lr = self._train_lr if lr is None else float(lr)
        max_steps = self._train_max_steps if max_steps is None else int(max_steps)
        tolerance = self._train_tolerance if tolerance is None else int(tolerance)

        factors = self._train_alpha_tensor()
        target = self.tensor_calculator.target.to(self.device)
        market_features = self.market_features.to(self.device)
        finite = torch.isfinite(target) & torch.isfinite(factors).all(dim=-1)
        if not finite.any():
            return self.weights

        factors_clean = torch.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
        target_clean = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        encoder, gate = self._new_modules(self.size)
        w0 = torch.tensor(self.weights, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam(
            [w0, *encoder.parameters(), *gate.parameters()],
            lr=lr,
        )

        best_loss = float("inf")
        best_w0 = w0.detach().clone()
        best_encoder_state = copy.deepcopy(encoder.state_dict())
        best_gate_state = copy.deepcopy(gate.state_dict())
        tolerance_count = 0

        for step in count():
            embedding = encoder(market_features)
            alpha_gate = gate(embedding)
            dynamic_weights = w0[None, :] * alpha_gate
            score = torch.einsum("dsk,dk->ds", factors_clean, dynamic_weights)
            diff = score - target_clean
            loss_mse = diff[finite].pow(2).mean()
            loss_l1 = torch.norm(w0, p=1)
            loss_gate = gate.trans.weight.pow(2).sum()
            loss_encoder = sum(param.pow(2).sum() for param in encoder.parameters())
            loss = (
                loss_mse
                + self._l1_alpha * loss_l1
                + self._gate_l2 * loss_gate
                + self._encoder_l2 * loss_encoder
            )

            if not torch.isfinite(loss):
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = float(loss_mse.detach().item())
            if best_loss - loss_value > 1e-6:
                tolerance_count = 0
            else:
                tolerance_count += 1
            if loss_value < best_loss:
                best_loss = loss_value
                best_w0 = w0.detach().clone()
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_gate_state = copy.deepcopy(gate.state_dict())
            if tolerance_count >= tolerance or step >= max_steps:
                break

        self._encoder_state = {k: v.detach().cpu().clone() for k, v in best_encoder_state.items()}
        self._gate_state = {k: v.detach().cpu().clone() for k, v in best_gate_state.items()}
        self._state_size = self.size
        self._last_loss = best_loss
        with torch.no_grad():
            encoder, gate = self._new_modules(self.size, load_state=True)
            dynamic_weights = best_w0[None, :] * gate(encoder(market_features))
            self._last_dynamic_importance = dynamic_weights.abs().mean(dim=0).detach().cpu().numpy()
            score = torch.einsum("dsk,dk->ds", factors_clean, dynamic_weights)
            self._last_objective = batch_pearsonr(score, target).mean().item()

        return best_w0.detach().cpu().numpy()

    def evaluate_ensemble(self) -> float:
        if self.size == 0:
            return 0.
        score = self._predict_score_for_calculator(self.tensor_calculator, self.market_features)
        return batch_pearsonr(score, self.tensor_calculator.target.to(self.device)).mean().item()

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float]:
        if not isinstance(calculator, TensorAlphaCalculator):
            return calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights)  # type: ignore
        market_features = self._market_features_for_calculator(calculator)
        if market_features is None:
            return calculator.calc_pool_all_ret(self.exprs[:self.size], self.weights)  # type: ignore
        score = self._predict_score_for_calculator(calculator, market_features)
        target = calculator.target.to(self.device)
        return (
            batch_pearsonr(score, target).mean().item(),
            batch_spearmanr(score, target).mean().item(),
        )

    def most_significant_indices(self, k: int) -> List[int]:
        if self.size == 0:
            return []
        if self._last_dynamic_importance is None or len(self._last_dynamic_importance) < self.size:
            return super().most_significant_indices(k)
        ranks = (-self._last_dynamic_importance[:self.size]).argsort().argsort()
        return [i for i in range(self.size) if ranks[i] < k]

    def leave_only(self, indices: Iterable[int]) -> None:
        super().leave_only(indices)
        if self.size > 0:
            self.weights = self.optimize()

    def _calc_main_objective(self) -> Optional[float]:
        return self._last_objective

    def _get_extra_info(self, expr: Expression) -> Any:
        return self.tensor_calculator.evaluate_alpha(expr).detach().to(self.device)

    def _get_worst_index(self, weights: Sequence[float]) -> int:
        if self._last_dynamic_importance is not None and len(self._last_dynamic_importance) == self.size:
            importance = self._last_dynamic_importance
            if np.isfinite(importance).all():
                return int(np.argmin(importance))
        return int(np.argmin(np.abs(weights)))

    def _write_timing_summary(self, path: Path) -> None:
        gate_values, dynamic_weights = self._gate_and_dynamic_weights(self.market_features)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "alpha_idx",
                    "expr",
                    "base_weight_w0",
                    "single_ic_ret",
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
            for i in range(self.size):
                writer.writerow({
                    "alpha_idx": i,
                    "expr": str(self.exprs[i]),
                    "base_weight_w0": float(self.weights[i]),
                    "single_ic_ret": float(self.single_ics[i]),
                    "mean_gate": float(np.nanmean(gate_values[:, i])),
                    "std_gate": float(np.nanstd(gate_values[:, i])),
                    "min_gate": float(np.nanmin(gate_values[:, i])),
                    "max_gate": float(np.nanmax(gate_values[:, i])),
                    "mean_dynamic_weight": float(np.nanmean(dynamic_weights[:, i])),
                    "std_dynamic_weight": float(np.nanstd(dynamic_weights[:, i])),
                    "mean_abs_dynamic_weight": float(np.nanmean(np.abs(dynamic_weights[:, i]))),
                })

    def _write_daily_timing_weights(
        self,
        path: Path,
        calculator: TensorAlphaCalculator,
        market_features: Tensor,
    ) -> None:
        gate_values, dynamic_weights = self._gate_and_dynamic_weights(market_features)
        dates = self._date_strings(calculator, gate_values.shape[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["date"]
            + [f"gate_alpha_{i}" for i in range(self.size)]
            + [f"weight_alpha_{i}" for i in range(self.size)]
        )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t, date in enumerate(dates):
                row = {"date": date}
                for i in range(self.size):
                    row[f"gate_alpha_{i}"] = float(gate_values[t, i])
                    row[f"weight_alpha_{i}"] = float(dynamic_weights[t, i])
                writer.writerow(row)

    def _gate_and_dynamic_weights(self, market_features: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        encoder, gate = self._new_modules(self.size, load_state=True)
        weights = torch.tensor(self.weights, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            gate_values = gate(encoder(market_features.to(self.device)))
            dynamic_weights = weights[None, :] * gate_values
        return (
            gate_values.detach().cpu().numpy(),
            dynamic_weights.detach().cpu().numpy(),
        )

    @staticmethod
    def _date_strings(calculator: TensorAlphaCalculator, fallback_days: int) -> List[str]:
        stock_data = getattr(calculator, "data", None)
        if stock_data is None or not hasattr(stock_data, "_dates"):
            return [str(i) for i in range(fallback_days)]
        dates = stock_data._dates
        start = stock_data.max_backtrack_days
        stop = len(dates) - stock_data.max_future_days
        if stock_data.max_future_days == 0:
            stop = len(dates)
        selected = list(dates[start:stop])
        if len(selected) != fallback_days:
            return [str(i) for i in range(fallback_days)]
        return [pd.Timestamp(date).strftime("%Y-%m-%d") for date in selected]

    def _train_alpha_tensor(self) -> Tensor:
        values = []
        for i in range(self.size):
            value = self._extra_info[i]
            if value is None:
                expr = self.exprs[i]
                assert expr is not None
                value = self._get_extra_info(expr)
                self._extra_info[i] = value
            values.append(value)
        return torch.stack(values, dim=-1).to(self.device)

    def _predict_score_for_calculator(
        self,
        calculator: TensorAlphaCalculator,
        market_features: Tensor,
    ) -> Tensor:
        if self._encoder_state is None or self._gate_state is None or self._state_size != self.size:
            self.weights = self.optimize()
        factors = torch.stack(
            [calculator.evaluate_alpha(expr) for expr in self.exprs[:self.size]],  # type: ignore
            dim=-1,
        ).to(self.device)
        factors_clean = torch.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
        market_features = market_features.to(self.device)
        encoder, gate = self._new_modules(self.size, load_state=True)
        weights = torch.tensor(self.weights, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dynamic_weights = weights[None, :] * gate(encoder(market_features))
            return torch.einsum("dsk,dk->ds", factors_clean, dynamic_weights)

    def _market_features_for_calculator(self, calculator: TensorAlphaCalculator) -> Optional[Tensor]:
        if calculator is self.tensor_calculator:
            return self.market_features
        if self.market_feature_builder is None:
            return None
        stock_data = getattr(calculator, "data", None)
        if stock_data is None:
            return None
        features = self.market_feature_builder.transform(stock_data).values
        return self._as_market_tensor(features)

    def _new_modules(self, n_alphas: int, load_state: bool = False) -> Tuple[MarketEncoder, Gate]:
        encoder = MarketEncoder(
            d_input=self.market_features.shape[1],
            d_model=self._embedding_dim,
        ).to(self.device)
        gate = Gate(
            d_input=self._embedding_dim,
            d_output=n_alphas,
            beta=self._gate_temperature,
        ).to(self.device)
        if load_state:
            if self._encoder_state is None or self._gate_state is None:
                raise RuntimeError("State-aware gate has not been optimized yet")
            encoder.load_state_dict({k: v.to(self.device) for k, v in self._encoder_state.items()})
            gate.load_state_dict({k: v.to(self.device) for k, v in self._gate_state.items()})
        return encoder, gate

    def _as_market_tensor(self, market_features: Union[np.ndarray, Tensor]) -> Tensor:
        if isinstance(market_features, Tensor):
            tensor = market_features.detach().clone().to(self.device, dtype=torch.float32)
        else:
            tensor = torch.tensor(market_features, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2:
            raise ValueError("market_features must have shape [days, features]")
        return tensor
