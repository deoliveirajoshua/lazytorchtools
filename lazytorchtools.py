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
        hidden_activation: Union[nn.Module, Sequence[Any], None] = nn.LeakyReLU(0.2, inplace=True),
        final_activation: Optional[Any] = None,
        dropout: float = 0.0,
        batchnorm1d: bool = False,
        hidden_block: Optional[Union[nn.Module, Sequence[Any]]] = None,
    ):
        ops: List[Any] = []
        if hidden_dims is None:
            hidden_dims = []
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


__all__ = [
    "ModularNN",
    "FFNN",
    "ConvNN",
    "weights_init",
    "get_gpu",
    "set_seed",
    "toggle_grads",
    "count_parameters",
    "SurfaceHessian",
    "NTK",
]
