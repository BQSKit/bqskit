"""This module implements helper methods for generating docs."""
from __future__ import annotations

import os


def building_docs() -> bool:
    """Return true if currently building documentations."""

    if 'BQSKIT_DOC_CHECK_OVERRIDE' in os.environ:
        return False
    if 'READTHEDOCS' in os.environ:
        return True
    if '__SPHINX_BUILD__' in os.environ:
        return True
    return False
