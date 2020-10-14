"""tests.agents.test_pid.py"""
import pytest

from deluca.agents import PID


@pytest.mark.parametrize("K", ["", None, [None], [{}], [1, 2]])
def test_bad_inputs(K):
    """Test bad inputs"""
    with pytest.raises(ValueError):
        PID(K=K)
