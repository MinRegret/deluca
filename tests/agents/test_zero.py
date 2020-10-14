"""tests.agents.test_zero.py"""
import pytest

from deluca.agents import Zero


@pytest.mark.parametrize("shape", [None, 1, {}, (1, None), [2, ""]])
def test_bad_shapes(shape):
    """Test bad shapes"""
    with pytest.raises(ValueError):
        Zero(shape)


@pytest.mark.parametrize("shape", [[1, 2, 3], (4, 3, 2), (), []])
def test_zero(shape):
    """Test normal behavior"""
    agent = Zero(shape)

    assert agent.shape == tuple(shape)
