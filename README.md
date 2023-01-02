Take SGD optimizer as an example.
Open the optim folder in your environment (anaconda3/envs/env_name/lib/python3.X/site-packages/torch/optim/) and modify the __init__.py, __init__.pyi, _functional.py, sgd_gaf .py and sgd_gaf.pyi, see the comments in the file for details of the changes.

The most core code is as follows.

```
d_p =gamma *torch.atan(d_p*delta)
```


Change the optimizer in train.py.

```
optimizer = optim.SGD_ACT(model.parameters(), lr=args.lr, momentum=args.momentum, gamma=0.3, delta=4.5)
```