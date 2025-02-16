Take SGD optimizer as an example.
Open the optim folder in your environment (anaconda3/envs/env_name/lib/python3.X/site-packages/torch/optim/) and modify the __init__.py, __init__.pyi and _functional.py, and add sgd_act .py and sgd_act.pyi, see the comments in the file for details of the changes.

The core code is as follows.

```
d_p =gamma *torch.atan(d_p*delta)
```


Change the optimizer in train.py.

```
optimizer = optim.SGD_ACT(model.parameters(), lr=args.lr, momentum=args.momentum, gamma=0.3, delta=4.5)
```

BertGCN folder from https://github.com/ZeroRin/BertGCN, we changed its optimizer to optimizer with gradient activation . Please run its sh file to see the effect of gradient activation.

BERT-LSTM-CRF folder from https://github.com/hemingkx/CLUENER2020/tree/main, we changed its optimizer to optimizer with gradient activation . Please run its sh file to see the effect of gradient activation.

