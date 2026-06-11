"""Implementation of the Ranger optimizer.

Code taken from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/optimizer/ranger.py
"""

from .lookahead import Lookahead
from .radam import RAdam


def Ranger(
    params,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-5,
    weight_decay: float = 0.0,
    alpha: float = 0.5,
    k: int = 6,
):
    radam = RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return Lookahead(radam, alpha, k)
