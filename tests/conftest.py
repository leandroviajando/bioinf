import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_numpy():
    np.set_printoptions(precision=4, suppress=True)
