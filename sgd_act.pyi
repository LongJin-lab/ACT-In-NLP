from .optimizer import _params_t, Optimizer
'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''
class SGD_ACT(Optimizer):
    def __init__(self, params: _params_t, lr: float, momentum: float=..., dampening: float=..., weight_decay:float=..., nesterov:bool=..., gamam:float=..., delta:float=...) -> None: ...
'''------------------------------------------------------------------------ACT-----------------------------------------------------------------------'''