"""This module implements the CachedClass base class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Hashable


_logger = logging.getLogger(__name__)


class CachedClass:
    """
    CachedClass base class.

    Any class that inherits from CachedClass will be instantiated once per
    parameter set. Any subsequent attempts to instantiate a CachedClass with
    the same parameters will return the same object. CachedClass is not
    thread-safe.

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
        hash_a = all(isinstance(arg, Hashable) for arg in args)
        hash_kw = all(isinstance(arg, Hashable) for arg in kwargs.values())

        if not hash_a or not hash_kw:
            return super().__new__(cls)

        if cls._instances.get(
                (cls, args, tuple(kwargs.items())), None,
        ) is None:
            _logger.debug(
                (
                    'Creating cached instance for class: %s,'
                    ' with args %s, and kwargs %s'
                )
                % (cls.__name__, args, kwargs),
            )
            cls._instances[
                (cls, args, tuple(kwargs.items()))
            ] = super().__new__(cls)
        return cls._instances[(cls, args, tuple(kwargs.items()))]
