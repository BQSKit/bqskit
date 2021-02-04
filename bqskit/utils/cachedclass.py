"""This module implements the CachedClass base class."""
from __future__ import annotations

import logging
from typing import Any


_logger = logging.getLogger(__name__)


class CachedClass:
    """
    CachedClass base class.

    Any class that inherits from CachedClass will be instantiated once per
    parameter set. Any subsequent attempts to instantiate a CachedClass with
    the same parameters will return the same object.

    Examples:
        >>> x = CachedClass(1)
        >>> y = CachedClass(1)
        >>> z = CachedClass(2)
        >>> x is y
        True
        >>> x is z
        False
    """
    _instances: dict[Any, CachedClass] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> CachedClass:
        if cls._instances.get((cls, args, tuple(kwargs.items())), None) is None:
            _logger.debug(
                'Creating singleton instance for class: %s'
                % cls.__name__,
            )
            cls._instances[(cls, args, tuple(kwargs.items()))] = super().__new__(cls)
        return cls._instances[(cls, args, tuple(kwargs.items()))]