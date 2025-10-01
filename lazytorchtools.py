"""lazytorchtools

Tiny utilities for building small PyTorch models and a few analysis helpers.

Designed for PyTorch 2.8+.
"""
from __future__ import annotations
import copy
import os
import random
from typing import Any, List, Optional, Sequence, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.func import hessian, vmap, jacrev, functional_call


def _ensure_module(m: Any) -> Optional[nn.Module]:
    if m is None:
        return None
    if isinstance(m, nn.Module):
        return m
    if isinstance(m, type) and issubclass(m, nn.Module):
        return m()
    if callable(m):
        try:
            res = m()
        except Exception:
            return m
        if isinstance(res, nn.Module):
            return res
        return m
    return m


class ModularNN(nn.Module):
    """Lightweight container that builds an ``nn.Sequential`` from a list of operations."""

    def __init__(self, operation_order: Sequence[Any]):
        super().__init__()
        self.operation_order = [copy.deepcopy(op) for op in operation_order]
        self.model = nn.Sequential(*self.operation_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def stitch_architecture(self, other_model: "ModularNN") -> "ModularNN":
        first = copy.deepcopy(self.operation_order)
        second = copy.deepcopy(other_model.operation_order)
        return ModularNN(first + second)

    def duplicate_architecture(self) -> "ModularNN":
        return ModularNN(copy.deepcopy(self.operation_order))

    def view_architecture(self) -> List[Any]:
        return copy.deepcopy(self.operation_order)

    def view_model(self) -> nn.Sequential:
        return copy.deepcopy(self.model)


class FFNN(ModularNN):
    """Fully-connected network builder."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        # fast-track: specify number of hidden layers and a single hidden size
        hidden_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        hidden_activation: Union[nn.Module, Sequence[Any], None] = nn.LeakyReLU(0.2, inplace=True),
        final_activation: Optional[Any] = None,
        dropout: float = 0.0,
        batchnorm1d: bool = False,
        hidden_block: Optional[Union[nn.Module, Sequence[Any]]] = None,
    ):
        ops: List[Any] = []
        # validate fast-track options vs explicit hidden_dims
        if hidden_dims is not None and (hidden_layers is not None or hidden_size is not None):
            raise ValueError("Provide either `hidden_dims` OR the fast-track `hidden_layers`/`hidden_size`, not both.")

        if hidden_dims is None:
            if hidden_layers is None:
                hidden_dims = []
            else:
                if hidden_size is None:
                    raise ValueError("`hidden_size` must be provided when `hidden_layers` is set")
                if hidden_layers < 0:
                    raise ValueError("`hidden_layers` must be non-negative")
                hidden_dims = [hidden_size] * hidden_layers
        layers = [in_dim, *hidden_dims, out_dim]

        for i in range(len(layers) - 1):
            ops.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                if hidden_block is not None:
                    if isinstance(hidden_block, list):
                        ops.extend(hidden_block)
                    else:
                        ops.append(hidden_block)
                if batchnorm1d:
                    ops.append(nn.BatchNorm1d(layers[i + 1]))
                if hidden_activation is not None:
                    act = hidden_activation[i] if isinstance(hidden_activation, list) else hidden_activation
                    act_mod = _ensure_module(act)
                    if isinstance(act_mod, nn.Module):
                        ops.append(act_mod)
                if dropout != 0.0:
                    if dropout <= 0 or dropout >= 1:
                        raise ValueError("Dropout parameter not legal, must be 0<d<1")
                    ops.append(nn.Dropout(dropout))

        if final_activation is not None:
            fa = _ensure_module(final_activation)
            if isinstance(fa, nn.Module):
                ops.append(fa)

        super().__init__(ops)


class ConvNN(ModularNN):
    """Convolutional builder for 1d/2d stacks."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        hidden_channels: Optional[Sequence[int]] = None,
        conv_kernel: int = 4,
        stride: int = 2,
        padding: int = 1,
        conv_bias: bool = False,
        pool_kernel: int = 2,
        convtype: str = "2d",
        flatten: bool = False,
        dropout: float = 0.0,
        hidden_activations: Union[nn.Module, Sequence[Any], None] = nn.LeakyReLU(0.2, inplace=True),
        pooling: Optional[str] = None,
        batchnorm: bool = True,
        final_activation: Optional[Any] = None,
        hidden_block: Optional[Union[nn.Module, Sequence[Any]]] = None,
    ):
        ops: List[Any] = []
        if hidden_channels is None:
            hidden_channels = []
        layers = [in_channel, *hidden_channels, out_channel]

        for i in range(len(layers) - 1):
            if i == len(layers) - 2:
                if flatten:
                    ops.append(nn.Flatten())
                if final_activation is not None:
                    fa = _ensure_module(final_activation)
                    if isinstance(fa, nn.Module):
                        ops.append(fa)
                break

            if convtype == "2d":
                if layers[i + 1] >= layers[i]:
                    ops.append(nn.Conv2d(layers[i], layers[i + 1], conv_kernel, stride, padding, bias=conv_bias))
                else:
                    ops.append(nn.ConvTranspose2d(layers[i], layers[i + 1], conv_kernel, stride, padding, bias=conv_bias))
                if pooling:
                    ops.append(nn.MaxPool2d(pool_kernel))
                if batchnorm:
                    ops.append(nn.BatchNorm2d(layers[i + 1]))
            elif convtype == "1d":
                if layers[i + 1] >= layers[i]:
                    ops.append(nn.Conv1d(layers[i], layers[i + 1], conv_kernel, stride, padding, bias=conv_bias))
                else:
                    ops.append(nn.ConvTranspose1d(layers[i], layers[i + 1], conv_kernel, stride, padding, bias=conv_bias))
                if pooling:
                    ops.append(nn.MaxPool1d(pool_kernel))
                if batchnorm:
                    ops.append(nn.BatchNorm1d(layers[i + 1]))
            else:
                raise ValueError("convtype expected one of {'1d','2d'}")

            if hidden_block is not None:
                if isinstance(hidden_block, list):
                    ops.extend(hidden_block)
                else:
                    ops.append(hidden_block)

            if hidden_activations is not None:
                act = hidden_activations[i] if isinstance(hidden_activations, list) else hidden_activations
                act_mod = _ensure_module(act)
                if isinstance(act_mod, nn.Module):
                    ops.append(act_mod)

            if dropout != 0.0:
                if dropout <= 0 or dropout >= 1:
                    raise ValueError("Dropout parameter not legal, must be 0<d<1")
                ops.append(nn.Dropout(dropout))

        super().__init__(ops)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "Linear" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.normal_(m.bias.data, 0.0, 1.0)
    else:
        raise Exception(f"Unrecognized or unsupported layer type: {classname}")


def get_gpu(verbose: bool = True) -> Tuple[bool, torch.device]:
    got_gpu = torch.cuda.is_available()
    if got_gpu:
        device = torch.device("cuda:0")
        if verbose:
            try:
                print(f"Found GPU: ({torch.cuda.get_device_name(device=device)})")
            except Exception:
                print("Found GPU")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Found only CPU")
    return got_gpu, device


def bytes_to_gb(x): 
    return x / (1024 ** 3)

def gpu_memory_report(device: int | torch.device = None):
    """Print current allocated GB and unused GB on a CUDA device."""
    if device is None:
        device = torch.device('cuda', torch.cuda.current_device())
    else:
        device = torch.device(device)

    # Synchronize to make stats more accurate for just-finished kernels
    torch.cuda.synchronize(device)

    # PyTorch allocator stats (per device, for this process)
    allocated = torch.cuda.memory_allocated(device)
    reserved  = torch.cuda.memory_reserved(device)
    unused_in_reserved = max(0, reserved - allocated)

    # Driver-level stats (whole device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes

    print(f"Device: {device}  (name: {torch.cuda.get_device_name(device)})")
    print(f"PyTorch allocated: {bytes_to_gb(allocated):.2f} GB")
    print(f"PyTorch reserved (total): {bytes_to_gb(reserved):.2f} GB")
    print(f"PyTorch reserved but unused: {bytes_to_gb(unused_in_reserved):.2f} GB")
    print(f"CUDA free (driver): {bytes_to_gb(free_bytes):.2f} GB")
    print(f"CUDA total: {bytes_to_gb(total_bytes):.2f} GB")
    print(f"CUDA used (driver): {bytes_to_gb(used_bytes):.2f} GB")



def set_seed(seed: int) -> None:
    """Set random seeds to attempt deterministic behavior across torch / numpy / random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def toggle_grads(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class SurfaceHessian(nn.Module):
    def __init__(self, model: Callable[..., torch.Tensor], out_dim: int, model_kwargs: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.out_dim = out_dim
        self.kwargs = model_kwargs or {}
        self.hess_func = vmap(hessian(model, argnums=0))
        self.eigh = vmap(torch.linalg.eigh)

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_dims = x_in[0].flatten().numel()
        hess = self.hess_func(x_in, **self.kwargs)
        if self.out_dim > 1:
            hess_mat = hess.view(len(x_in), self.out_dim, x_dims, x_dims)
            eigvals, eigvecs = self.eigh(hess_mat.double())
        else:
            hess_mat = hess.squeeze()
            eigvals, eigvecs = torch.linalg.eigh(hess_mat.double())
            eigvals = eigvals.unsqueeze(1)
            eigvecs = eigvecs.unsqueeze(1)
        return eigvals, eigvecs


class NTK:
    """Neural tangent kernel utilities implemented with torch.func only.

    This mirrors the previous API but avoids `functorch.make_functional` by
    using `torch.func.functional_call` with an explicit mapping of parameter
    names to tensors.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        # Record parameter names and buffers so we can build the mapping
        self.param_names = [name for name, _ in model.named_parameters()]
        self.buffer_dict = {name: buf for name, buf in model.named_buffers()}
        # Represent parameters as a tuple of tensors (detached, requires_grad=True)
        self.params = tuple(p.detach().clone().requires_grad_(True) for _, p in model.named_parameters())

    def _params_to_dict(self, params_seq: Sequence[torch.Tensor]) -> dict:
        d = {name: tensor for name, tensor in zip(self.param_names, params_seq)}
        # include buffers (non-trainable) unchanged from the model
        d.update(self.buffer_dict)
        return d

    def fnet_single(self, params: Sequence[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Run the model functionally for a single input `x` using `params`.

        `params` is a sequence of tensors in the same order as
        `model.named_parameters()`.
        """
        params_dict = self._params_to_dict(params)
        out = functional_call(self.model, params_dict, (x.unsqueeze(0),))
        return out.squeeze(0)

    def empirical_ntk_jacobian_contraction(self, x1: torch.Tensor, x2: torch.Tensor, compute: str = "full", chunk_size: Optional[int] = None) -> torch.Tensor:
        # jacrev over parameter PyTree; vmap over the batch dimension of x
        jac1 = vmap(jacrev(self.fnet_single), (None, 0), chunk_size=chunk_size)(self.params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        jac2 = vmap(jacrev(self.fnet_single), (None, 0), chunk_size=chunk_size)(self.params, x2)
        jac2 = [j.flatten(2) for j in jac2]

        if compute == "full":
            einsum_expr = "Naf,Mbf->NMab"
        elif compute == "trace":
            einsum_expr = "Naf,Maf->NM"
        elif compute == "diagonal":
            einsum_expr = "Naf,Maf->NMa"
        else:
            raise ValueError("compute must be one of {'full','trace','diagonal'}")

        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        result = result.sum(0).squeeze()
        return result

    def empirical_ntk_jacobian_reflexive_contraction(self, x1: torch.Tensor, compute: str = "full", chunk_size: Optional[int] = None) -> torch.Tensor:
        jac1 = vmap(jacrev(self.fnet_single), (None, 0), chunk_size=chunk_size)(self.params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        if compute == "full":
            einsum_expr = "Naf,Mbf->NMab"
        elif compute == "trace":
            einsum_expr = "Naf,Maf->NM"
        elif compute == "diagonal":
            einsum_expr = "Naf,Maf->NMa"
        else:
            raise ValueError("compute must be one of {'full','trace','diagonal'}")

        result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac1)])
        result = result.sum(0).squeeze()
        return result


# -----------------------
# Fancy-indexable, lazy dataset adapters
# -----------------------

def fast_labels(ds) -> np.ndarray:
    # Preferred: .targets
    if hasattr(ds, "targets") and ds.targets is not None:
        t = ds.targets
        if isinstance(t, torch.Tensor):
            return t.cpu().numpy().astype(np.int64)
        return np.asarray(t, dtype=np.int64)

    # Food101 (and some others) may provide .labels
    if hasattr(ds, "labels") and ds.labels is not None:
        lab = ds.labels
        # If labels are strings, map via class_to_idx
        if len(lab) > 0 and isinstance(lab[0], str) and hasattr(ds, "class_to_idx"):
            m = ds.class_to_idx
            return np.asarray([m[s] for s in lab], dtype=np.int64)
        return np.asarray(lab, dtype=np.int64)

    # Generic ImageFolder-style fallback: .samples or .imgs
    if hasattr(ds, "samples") and ds.samples is not None:
        return np.asarray([cls for _, cls in ds.samples], dtype=np.int64)
    if hasattr(ds, "imgs") and ds.imgs is not None:
        return np.asarray([cls for _, cls in ds.imgs], dtype=np.int64)

    # Last resort (slow): iterates items (avoid if at all possible)
    return np.asarray([int(ds[i][1] if not torch.is_tensor(ds[i][1]) else ds[i][1].item())
                       for i in range(len(ds))], dtype=np.int64)


IdxLike = Union[int, Sequence[int], np.ndarray, torch.Tensor]


class FancyIndexableDataset(Dataset):
    def __init__(self, base_ds, transform=None, show_progress=False, tqdm_desc="loading from disc"):
        self.base_ds = base_ds
        self.transform = transform
        self.show_progress = show_progress
        self.tqdm_desc = tqdm_desc

    # ... keep __len__ and _get_one unchanged ...

    def __getitem__(self, idx):
        # Single index path (no progress bar)
        if isinstance(idx, (int, np.integer)) or (torch.is_tensor(idx) and idx.dim() == 0):
            x, y = self._get_one(int(idx))
            return x, y

        # Normalize to a 1D numpy array for fancy indexing
        if isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)
        if torch.is_tensor(idx):
            idx = idx.to(torch.long).cpu().numpy()
        if isinstance(idx, np.ndarray):
            idx = idx.astype(np.int64)

        # Use tqdm only for "batch" fancy indexing
        it = idx
        if self.show_progress:
            it = tqdm(it, total=len(idx), desc=self.tqdm_desc, leave=False)

        xs, ys = [], []
        for i in it:
            x, y = self._get_one(int(i))
            xs.append(x)
            ys.append(y)

        X = torch.stack(xs, dim=0)
        Y = torch.tensor(ys, dtype=torch.long)
        return X, Y


class IndexedView(Dataset):
    """
    Restrict a FancyIndexableDataset to a fixed list of absolute indices.
    Still supports fancy indexing relative to this view.
    """
    def __init__(self, base: FancyIndexableDataset, idxs: np.ndarray):
        self.base = base
        self.idxs = np.asarray(idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: IdxLike):
        if isinstance(idx, (int, np.integer)) or (torch.is_tensor(idx) and idx.dim() == 0):
            return self.base[int(self.idxs[int(idx)])]
        if isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)
        if torch.is_tensor(idx):
            idx = idx.to(torch.long).cpu().numpy()
        if isinstance(idx, np.ndarray):
            idx = idx.astype(np.int64)
        abs_idx = self.idxs[idx]
        return self.base[abs_idx]


class PairView(Dataset):
    """
    Pair a (lazy, fancy-indexable) image dataset with a provided label tensor (possibly noisy).
    Returns (X, Y) for __getitem__, where idx can be scalar or a batch of indices.
    """
    def __init__(self, img_ds: Dataset, labels: torch.Tensor):
        assert len(img_ds) == len(labels)
        self.img_ds = img_ds
        self.labels = labels

    def __len__(self):
        return len(self.img_ds)

    def __getitem__(self, idx: IdxLike):
        X, _ = self.img_ds[idx]  # discard base label; use provided labels
        if isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)
        if torch.is_tensor(idx):
            idx = idx.to(torch.long).cpu().numpy()
        if isinstance(idx, np.ndarray):
            idx = idx.astype(np.int64)
            Y = self.labels[idx]
        elif isinstance(idx, (int, np.integer)) or (torch.is_tensor(idx) and idx.dim() == 0):
            Y = self.labels[int(idx)]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
        return X, Y




__all__ = [
    "ModularNN",
    "FFNN",
    "ConvNN",
    "weights_init",
    "get_gpu",
    "bytes_to_gb",
    "gpu_memory_report",
    "set_seed",
    "toggle_grads",
    "count_parameters",
    "SurfaceHessian",
    "NTK",
    "fast_labels",
    "FancyIndexableDataset",
    "IndexedView",
    "PairView"
]
