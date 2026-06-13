# TE-AlphaGen

TE-AlphaGen 是在 AlphaGen 基础上加入市场择时信息的轻量扩展版本。原始 AlphaGen 的表达式生成器、RPN 动作空间和个股特征输入保持不变；新增部分只作用在 alpha pool 组合层，用 CSI300 市场状态为 pool 中不同 alpha 生成每日动态权重。

核心思想是：

```text
个股 alpha 表达式负责横截面选股；
CSI300 市场特征经过 MarketEncoder 和 MASTER-style Gate 后，负责调节不同 alpha 在不同市场状态下的权重。
```

## 数据来源与处理

本项目需要两类数据。

第一类是 Qlib 格式的个股数据，用于 AlphaGen 原始表达式计算。数据来自 baostock，并整理为 Qlib 本地数据目录，目录结构应包含：

```text
calendars/
features/
instruments/
```

个股数据至少需要包含 AlphaGen 默认使用的字段：

```text
$open
$close
$high
$low
$volume
$vwap
```

第二类是择时模块使用的 CSI300 指数数据，文件已放在：

```text
market_data/SH000300.csv
```

该数据同样来自 baostock。虽然 CSV 中包含多列指数行情字段，但当前择时模块实际只使用：

```text
date
close
volume
```

市场特征构造方式参考 MASTER 的市场表示思想。当前基于单个 CSI300 指数构造 21 维市场特征：

```text
close
close_mean_5, close_std_5, volume_mean_5, volume_std_5
close_mean_10, close_std_10, volume_mean_10, volume_std_10
close_mean_20, close_std_20, volume_mean_20, volume_std_20
close_mean_30, close_std_30, volume_mean_30, volume_std_30
close_mean_60, close_std_60, volume_mean_60, volume_std_60
```

标准化方式为训练期标准化：先在训练区间计算每一维市场特征的均值和标准差，再用同一组均值和标准差处理训练集、验证集和测试集，避免使用测试集统计量。

市场特征不会拼接到个股输入 tensor 中，而是作为单独的 `market_features` 输入 `StateAwareMseAlphaPool`。因此原始 AlphaGen 的表达式系统仍然只使用个股特征，择时信息只影响 alpha 组合权重。

## 运行代码

建议使用 Python 3.8 环境：

```bash
conda create -n alphagen_te python=3.8
conda activate alphagen_te
pip install -r requirements.txt
```

训练示例：

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

回测示例：

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

画图示例：

```bash
python scripts/plot_te_backtest.py \
  --output_prefix=out/backtests/te15_dynamic/<run_name>_<steps>
```

其中 `<run_name>` 和 `<steps>` 需要替换为实际训练输出目录和 checkpoint 步数，例如：

```text
out/results/csi300_20_0_20260606134731_te15/100352_steps_pool.json
```

## Citation

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}

@inproceedings{li2024master,
    title = {Master: Market-guided stock transformer for stock price forecasting},
    author = {Li, Tong and Liu, Zhaoyang and Shen, Yanyan and Wang, Xue and Chen, Haokun and Huang, Sen},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume = {38},
    number = {1},
    pages = {162--170},
    year = {2024}
}
```
