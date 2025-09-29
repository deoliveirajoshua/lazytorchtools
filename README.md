# lazytorchtools

Tiny utilities for PyTorch 2.8+.

Quick usage:

```py
from lazytorchtools import FFNN, count_parameters

model = FFNN(10, 1, hidden_dims=[32, 16])
print('params', count_parameters(model))
```

Notes:
- Requires torch and numpy. functorch is used via torch.func on PyTorch 2.x.
