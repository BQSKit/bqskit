"""This module implements the CachedClass base classes."""
from __future__ import annotations

import logging
from typing import Any
from typing import Hashable
from typing import TypeVar

from bqskit.utils.docs import building_docs

if not building_docs():
    from numpy.lib.mixins import NDArrayOperatorsMixin
else:
    class NDArrayOperatorsMixin:  # type: ignore
        pass


_logger = logging.getLogger(__name__)


T = TypeVar('T')


class CachedClass:
    """
    A class that caches its instances.

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

    def __new__(cls: type[T], *args: Any, **kwargs: Any) -> T:
        hash_a = all(
            isinstance(arg, Hashable)
            and not isinstance(arg, NDArrayOperatorsMixin)
            for arg in args
        )
        hash_kw = all(
            isinstance(arg, Hashable)
            and not isinstance(arg, NDArrayOperatorsMixin)
            for arg in kwargs.values()
        )

        if not hash_a or not hash_kw:
            obj = object.__new__(cls)
            obj.__cache_key__ = (cls, args, kwargs)  # type: ignore
            return obj

        key = (cls, args, tuple(kwargs.items()))

        _instances = cls._instances  # type: ignore

        if _instances.get(key, None) is None:
            _logger.debug(
                (
                    'Creating cached instance for class: %s,'
                    ' with args %s, and kwargs %s'
                )
                % (cls.__name__, args, kwargs),
            )
            _instances[key] = object.__new__(cls)
            _instances[key].__cache_key__ = (cls, args, kwargs)

        return _instances[key]

    def __copy__(self) -> CachedClass:
        return self

    def __deepcopy__(self, memo: Any) -> CachedClass:
        return self.__copy__()

    def __getnewargs_ex__(self) -> tuple[Any, Any]:
        return self.__cache_key__[1], self.__cache_key__[2]  # type: ignore
