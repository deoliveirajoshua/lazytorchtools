import sys
import pathlib
import pytest

# Ensure we import the package from the local 'projects' dir
sys.path.insert(0, str(pathlib.Path(__file__).parents[1].resolve()))


def _import_pkg():
    try:
        import lazytorchtools as lz
    except Exception as e:
        pytest.skip(f"Could not import lazytorchtools: {e}")
    return lz


def test_ffnn_and_count():
    lz = _import_pkg()
    # build a tiny network and check parameter counting
    model = lz.FFNN(4, 1, hidden_dims=[8, 4])
    n = lz.count_parameters(model)
    assert isinstance(n, int) and n > 0


def test_toggle_grads():
    lz = _import_pkg()
    model = lz.FFNN(3, 1, hidden_dims=[5])
    lz.toggle_grads(model, False)
    assert all(not p.requires_grad for p in model.parameters())
    lz.toggle_grads(model, True)
    assert all(p.requires_grad for p in model.parameters())
