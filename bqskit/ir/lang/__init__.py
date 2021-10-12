"""This package implements language reading and writing features."""
from __future__ import annotations

from bqskit.ir.lang.language import Language

__all__ = ['Language']

_language_dict: dict[str, Language] = {}


def register_language(extension: str, language: Language) -> None:
    _language_dict[extension] = language


def get_language(extension: str) -> Language:
    if extension not in _language_dict:
        raise ValueError(f'Unsupported extension: {extension}.')

    return _language_dict[extension]
