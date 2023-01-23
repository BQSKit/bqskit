"""This module implements helper methods for random generators."""
from __future__ import annotations

import ctypes

import numpy as np


def seed_random_sources(seed: int) -> None:
    """Seed the various sources of randomness BQSKit uses."""
    # set rand() seed, used by Ceres
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
    libc.srand(seed)
    # set numpy seed
    np.random.seed(seed)
