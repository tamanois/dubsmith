"""Dubsmith local web API."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args: object, **kwargs: object) -> object:
    """Lazy import so importing the package never builds an application."""
    from .app import create_app as factory

    return factory(*args, **kwargs)
