"""This module implements the objects necessary for internal properties."""
from __future__ import annotations
import abc
import logging
from typing import Callable, Iterator, MutableMapping, Sequence
from typing import Generic, TypeVar, Any, Callable
_logger = logging.getLogger(__name__)

T = TypeVar('T')


class internal_property(Generic[T]):
    def __init__(self, fget: Callable[[Any], T]) -> None:
        self.__doc__ = fget.__doc__
        self.name = fget.__name__
        self.attr_name = f'__internal_{self.name}__'

    def __get__(self, instance: Any, cls: Any) -> T:
        if instance is None:
            raise AttributeError(f"Can only access {self.name} from instance.")

        if not hasattr(instance, self.attr_name):
            raise AttributeError(f"Property, {self.name}, never assigned.")

        return getattr(instance, self.attr_name)

    def __set__(self, instance: Any, value: T) -> None:
        if hasattr(instance, self.attr_name):
            raise AttributeError(f"Cannot reset {self.name} property.")
        
        setattr(instance, self.attr_name, value)

class final_property(Generic[T]):
    def __init__(self, name: str, value: T) -> None:
        self.__doc__ = value.__doc__
        self.name = name
        self.value = value

        if isinstance(self.value, Callable):  # type: ignore
            raise TypeError(
                "Cannot override an internal property with a function."
            )

    def __get__(self, instance: Any, cls: Any) -> T:
        if instance is None:
            raise AttributeError(f"Can only access {self.name} from instance.")

        if isinstance(self.value, property):
            return self.value.__get__(instance, cls)

        return self.value

    def __set__(self, instance: Any, value: T) -> None:
        raise AttributeError(f"Cannot reset {self.name} property.")
    
class internal_impl_dict(MutableMapping[str, Any]):
    def __init__(self, internals: list[str]) -> None:
        self._dict = {'__internalproperties__': internals}
        self._internals = internals

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._internals:
            self._dict[key] = final_property(key, value)
        else:
            self._dict[key] = value

    def __delitem__(self, key: str) -> None:
        del self._dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

class InternalABCMeta(abc.ABCMeta):
    @classmethod
    def __prepare__(metacls, name: str, bases: Sequence[type], **kwargs) -> internal_impl_dict:
        __internalproperties__ = []
        for base in bases:
            for cls in base.mro():
                for k, v in cls.__dict__.items():
                    if isinstance(v, internal_property):
                        __internalproperties__.append(k)
        
        return internal_impl_dict(__internalproperties__)

    def __new__(metacls, name: str, bases: Sequence[type], data: internal_impl_dict) -> InternalABCMeta:
        return super().__new__(metacls, name, bases, dict(data))
    
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = super().__call__(*args, **kwargs)
        for prop in instance.__internalproperties__:
            if not hasattr(instance, prop):
                print(instance.__class__.__name__, f"Warning internal property never set {prop}.")
        return instance
    
    def __setattr__(cls, name: str, value: Any) -> None:
        if name in cls.__internalproperties__:
            raise AttributeError(f"Cannot reset {name} property.")
        return super().__setattr__(name, value)
