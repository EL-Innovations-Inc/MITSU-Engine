import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mitsu_engine as me  # noqa: E402


@pytest.fixture(autouse=True)
def reset_engine():
    """Each test starts from float32 and a fixed global seed."""
    me.set_dtype(np.float32)
    me.manual_seed(0)
    yield
    me.set_dtype(np.float32)
