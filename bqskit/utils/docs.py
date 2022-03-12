"""This module implements helper methods for generating docs."""
from __future__ import annotations

import os


def building_docs() -> bool:
    """Return true if currently building documentations."""

    if 'READTHEDOCS' in os.environ:
        return True
    if '__SPHINX_BUILD__' in os.environ:
        return True
    return False
