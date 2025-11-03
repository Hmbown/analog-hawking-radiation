"""Compatibility alias for legacy ``physics_engine`` imports."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable

from analog_hawking import physics_engine as _physics_engine

_alias_module = sys.modules[__name__]


def _export_public_attributes(source: ModuleType, alias: ModuleType) -> None:
    for attribute in dir(source):
        if attribute.startswith("_"):
            continue
        setattr(alias, attribute, getattr(source, attribute))


def _register_submodules(names: Iterable[str]) -> None:
    for name in names:
        try:
            module = importlib.import_module(f"{_physics_engine.__name__}.{name}")
        except ModuleNotFoundError:
            continue
        sys.modules[f"{__name__}.{name}"] = module
        setattr(_alias_module, name.split(".")[0], module)


_export_public_attributes(_physics_engine, _alias_module)
_register_submodules(
    [
        "plasma_models",
        "optimization",
        "simulation",
        "plasma_mirror",
        "horizon",
        "horizon_hybrid",
        "multi_beam_superposition",
        "anabhel_attribution",
    ]
)

__all__ = getattr(
    _physics_engine, "__all__", [name for name in dir(_alias_module) if not name.startswith("_")]
)
