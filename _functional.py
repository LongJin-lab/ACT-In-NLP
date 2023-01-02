r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Optional

# TODO: use foreach API in optim._functional to do all the computation

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)

'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''        
def sgd_act(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool,
        gamma: float,
        delta: float):
'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''
        d_p =gamma *torch.atan(d_p*delta)
'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr

        param.add_(d_p, alpha=alpha)


