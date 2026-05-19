from __future__ import annotations

from copy import deepcopy
from typing import Any


class SerializeField:
    def __init__(
        self,
        default: Any = None,
        field_type: str | type | None = None,
        ref: str | None = None,
        choices: list[Any] | tuple[Any, ...] | None = None,
        **metadata: Any,
    ):
        self.default = default
        self.field_type = _normalize_type(field_type) if field_type is not None else _infer_type(default, ref)
        self.ref = ref
        self.choices = list(choices) if choices is not None else None
        self.metadata = metadata
        self.name: str | None = None
        self.storage_name: str | None = None

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self.storage_name = f"_{name}"

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.storage_name in instance.__dict__:
            return instance.__dict__[self.storage_name]
        return deepcopy(self.default)

    def __set__(self, instance, value) -> None:
        instance.__dict__[self.storage_name] = value

    def to_metadata(self) -> dict[str, Any]:
        data = {
            "type": self.field_type,
            "default": self.default,
        }
        if self.ref:
            data["ref"] = self.ref
        if self.choices:
            data["choices"] = self.choices
        data.update(self.metadata)
        return data


def get_serialized_fields(component_cls: type) -> dict[str, SerializeField]:
    fields: dict[str, SerializeField] = {}
    for cls in reversed(component_cls.__mro__):
        for name, value in cls.__dict__.items():
            if isinstance(value, SerializeField):
                fields[name] = value
    return fields


def get_serialized_field_metadata(component_cls: type) -> dict[str, dict[str, Any]]:
    return {name: field.to_metadata() for name, field in get_serialized_fields(component_cls).items()}


def _normalize_type(field_type: str | type) -> str:
    if isinstance(field_type, str):
        return field_type
    return getattr(field_type, "__name__", str(field_type))


def _infer_type(default: Any, ref: str | None) -> str:
    if ref:
        return ref
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    if isinstance(default, (list, tuple)):
        return "vector"
    return "Any"
