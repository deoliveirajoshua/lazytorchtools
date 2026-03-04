![LazyTorchTools Logo](imgs/logo.png)

# LazyTorchTools

Small, focused utilities for PyTorch for those who want to be *lazy* with some boilerplate things.

This package is intentionally tiny and dependency-light: it expects PyTorch >= 2.0 (with `torch.func`) and NumPy.

Highlights
- Lightweight model builders: `FFNN`, `ConvNN`, and a minimal `ModularNN` wrapper.
- Small analysis helpers: parameter counting, seed setting, and more.
- `FeatureExtractor` context manager to easily snapshot intermediate layer activations.

Installation

Install PyTorch first (see https://pytorch.org). Then install this package directly from GitHub with pip:

```bash
pip install git+https://github.com/deoliveirajoshua/lazytorchtools.git
```

Quick example (imported as `lazy`)

This example shows a short script that:
- imports the package as `lazy`,
- sets a random seed for reproducibility,
- queries whether a GPU is available,
- builds a tiny FFNN model.

```py
import torch
import lazytorchtools as lazy

# deterministic-ish behavior for tests and demos
lazy.set_seed(42)

# get device info
got_gpu, device = lazy.get_gpu(verbose=True)

# build a tiny model with
# input dimension of 4
# output dimension of 2
# 3 hidden layers of dimensions 8, 10, then 12
model = lazy.FFNN(4, 2, hidden_dims=[8, 10, 12]).to(device)

# count trainable parameters
print('trainable params:', lazy.count_parameters(model))

# show how much VRAM is currently in use by PyTorch on the device
lazy.gpu_memory_report(device)
```

## Empirical NTK & Tangents

For Neural Tangent Kernel (NTK) computations and other tangent-space utilities, please see the separate [torchtangents](https://github.com/deoliveirajoshua/torchtangents) library!


## FFNN usage variants

You can specify hidden layers in two ways:

- Original (explicit list of hidden dims):

```py
# explicit hidden sizes per layer
model = lazy.FFNN(4, 2, hidden_dims=[32, 16])
```

- Fast-track (L layers of size h):

```py
# 3 hidden layers each with hidden size 64
model = lazy.FFNN(4, 2, hidden_layers=3, hidden_size=64)
```

## More
- See the `lazytorchtools.py` source for other helpers (weight init, conv builders, inference surface Hessians, etc.).
