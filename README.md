# TE-AlphaGen: Market-Timing Enhanced AlphaGen

This repository is a lightweight extension of AlphaGen for formulaic alpha mining. The main change is a market-state-aware alpha pool: a CSI300 market representation is encoded and passed through a MASTER-style softmax gate to produce daily dynamic weights for alphas in the pool.

The original AlphaGen expression generator and stock feature tensor are kept unchanged. Market features are stored in a separate tensor and only interact with alpha outputs in the pool combination layer.

## Main Changes

- `alphagen/timing/market_features.py`: builds 21 CSI300 market features from index close and volume.
- `alphagen/timing/market_encoder.py`: maps market features to an embedding and applies a MASTER-style `Gate`.
- `alphagen/models/state_aware_alpha_pool.py`: replaces the static linear pool with a market-gated pool.
- `scripts/rl.py`: training entry for TE-AlphaGen v1.5.
- `scripts/te_backtest.py`: Qlib backtest using saved market-gated pools.
- `scripts/plot_te_backtest.py`: plots return curves, metrics, gate heatmap, and dynamic alpha weights.

## Repository Layout

```text
alphagen/                 Core AlphaGen modules and the timing-aware pool
alphagen_qlib/            Qlib data adapter
alphagen_generic/         Shared baseline utilities
alphagen_llm/             Optional LLM-related utilities
data_collection/          Baostock/Qlib data preparation scripts
dso/, gplearn/            Baseline implementations
scripts/                  Training, backtest, plotting, and baselines
backtest.py               Qlib backtest wrapper
trade_decision.py         Trading decision helpers
requirements.txt          Python dependencies
market_data/              CSI300 market index CSV used by the timing module
```

## Environment

Python 3.8 is recommended because this project follows the original AlphaGen/Qlib environment.

```bash
conda create -n alphagen_te python=3.8
conda activate alphagen_te
pip install -r requirements.txt
```

On CUDA machines, make sure the installed PyTorch version matches the server CUDA runtime.

## Data Preparation

You need two data inputs.

First, a Qlib-format stock dataset:

```text
calendars/
features/
instruments/
```

The dataset must contain the stock fields used by AlphaGen, especially `$open`, `$close`, `$high`, `$low`, `$volume`, and `$vwap`.

Second, a CSI300 market index CSV. This repository includes the timing input used in our experiments:

```text
market_data/SH000300.csv
```

The timing module only requires these columns:

```text
date, close, volume
```

The included file may contain more baostock columns, which are ignored by the timing feature builder. You can also pass another path through `--market_index_csv`.

## Train

Run from the repository root:

```bash
python scripts/rl.py \
  --random_seeds=0 \
  --pool_capacity=20 \
  --steps=100000 \
  --state_pool_train_steps=200 \
  --market_embedding_dim=64 \
  --instruments=csi300 \
  --qlib_data_path=/path/to/cn_data_baostock_alphagen \
  --market_index_csv=market_data/SH000300.csv
```

Important arguments:

- `--pool_capacity`: alpha pool size, e.g. `10` or `20`.
- `--steps`: PPO training timesteps.
- `--state_pool_train_steps`: optimization steps for `w0`, market encoder, and alpha gate each time the pool is updated.
- `--market_embedding_dim`: market embedding dimension before the gate, e.g. `16`, `64`, or `128`.
- `--state_include_current`: whether to use signal-day market features. Default is `True`.

Training outputs are saved under:

```text
out/results/<run_name>/
out/tensorboard/<run_name>/
```

Useful files:

```text
<steps>_steps_pool.json
<steps>_steps_timing_train_summary.csv
<steps>_steps_timing_train_daily_weights.csv
<steps>_steps_timing_test_1_daily_weights.csv
<steps>_steps_timing_test_2_daily_weights.csv
<steps>_steps_timing_test_3_daily_weights.csv
```

## Backtest

Use a saved pool JSON:

```bash
python scripts/te_backtest.py \
  --pool_path=out/results/<run_name>/<steps>_steps_pool.json \
  --qlib_data_path=/path/to/cn_data_baostock_alphagen \
  --market_index_csv=market_data/SH000300.csv \
  --instruments=csi300 \
  --test_start=2022-01-01 \
  --test_end=2023-06-30 \
  --top_k=50 \
  --n_drop=5 \
  --output_prefix=out/backtests/te15_dynamic/<run_name>_<steps>
```

Backtest outputs:

```text
<output_prefix>-prediction.pkl
<output_prefix>-report.pkl
<output_prefix>-result.json
<output_prefix>-summary.json
<output_prefix>-timing-daily-weights.csv
<output_prefix>-timing-summary.csv
```

During backtest, the alpha pool, `w0`, market encoder, and gate parameters are fixed. Daily market weights still change because the market input changes each day.

## Plot

```bash
python scripts/plot_te_backtest.py \
  --output_prefix=out/backtests/te15_dynamic/<run_name>_<steps>
```

Figures are saved to:

```text
out/backtests/te15_dynamic/<run_name>_<steps>_figures/
```

Main figures:

- `cumulative_returns.png`: dynamic TE-AlphaGen vs static `w0` and benchmark.
- `excess_returns.png`: excess return over benchmark.
- `metrics.png`: annual return, information ratio, max drawdown, and related metrics.
- `gate_heatmap.png`: daily market gate values over the backtest period.
- `top_dynamic_weights.png`: daily dynamic weights for the most important alphas.
- `alpha_importance.png`: base `w0` vs mean absolute dynamic weights.

## Model Formula

For alpha \(j\), stock \(i\), and day \(t\):

\[
e_t = MarketEncoder(m_t)
\]

\[
g_t = K \cdot softmax(W_g e_t / \tau)
\]

\[
w_{t,j} = w_{0,j} \cdot g_{t,j}
\]

\[
score_{i,t} = \sum_{j=1}^{K} w_{t,j}\alpha_j(i,t)
\]

Here `market_features` has shape `[D, 21]`, alpha outputs have shape `[D, S, K]`, and the final score has shape `[D, S]`.

## What To Upload To GitHub

Upload source code and lightweight metadata:

```text
README.md
requirements.txt
.gitignore
alphagen/
alphagen_qlib/
alphagen_generic/
alphagen_llm/
data_collection/
dso/
gplearn/
scripts/
images/
backtest.py
trade_decision.py
gp.py
dso.py
market_data/SH000300.csv
```

Do not upload local data, logs, checkpoints, or experiment outputs:

```text
out/
data/
logs/
tb_logs/
__pycache__/
*.pyc
SERVER_INFO.md
local Qlib data directories
large zip/pkl/model checkpoint files
```

If you want to share reproducible results, upload small selected `pool.json`, `summary.json`, or figure PNG files separately under a clearly named `examples/` or `docs/` directory.

## Citation

This project is based on AlphaGen:

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```
