from typing import Type
from alphagen.data.expression import *


MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256  # 未被实际使用

OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs,  # Sign,
    Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var,  # Skew, Kurt,
    Max, Min,
    Med, Mad,  # Rank,
    Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]

DELTA_TIMES = [1, 5, 10, 20, 40]  # 代表不同的时间跨度，单位为交易日

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]  # 代表不同的数值常量，覆盖了常见的比例和幅度

REWARD_PER_STEP = 0.  # 每一步的奖励，实际奖励主要来自于最终的IC值
