"""This module implements the Rebase2QuditGatePass (Deprecated Location)."""
from __future__ import annotations

import warnings

warnings.warn(
    'The Rebase2QuditGatePass has moved to bqskit.passes.retarget.two.'
    ' The old location will be removed and this warning will become an'
    ' error in the future.',
    DeprecationWarning,
)

from bqskit.passes.retarget.two import Rebase2QuditGatePass  # noqa
