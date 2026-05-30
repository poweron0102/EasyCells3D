from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


def export(method: Callable) -> Callable:
    method.__easycells_export__ = True
    return method


class Asset:
    pass


@dataclass
class ExportedMethodMetadata:
    name: str
    required_args: list["AssetParameterMetadata"] = field(default_factory=list)
    optional_args: list["AssetParameterMetadata"] = field(default_factory=list)
    return_type: str = "Any"


@dataclass
class AssetParameterMetadata:
    name: str
    type: str = "Any"
    default: Any = None
    required: bool = True


@dataclass
class AssetMetadata:
    name: str
    class_path: str
    module: str
    required_args: list[AssetParameterMetadata] = field(default_factory=list)
    optional_args: list[AssetParameterMetadata] = field(default_factory=list)
    methods: dict[str, ExportedMethodMetadata] = field(default_factory=dict)
    asset_cls: type | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("asset_cls", None)
        return data


def get_exported_methods(asset_cls: type) -> dict[str, Callable]:
    methods: dict[str, Callable] = {}
    for name, value in inspect.getmembers(asset_cls, inspect.isfunction):
        if getattr(value, "__easycells_export__", False):
            methods[name] = value
    return methods


def get_exported_method_metadata(asset_cls: type) -> dict[str, ExportedMethodMetadata]:
    metadata: dict[str, ExportedMethodMetadata] = {}
    for name, method in get_exported_methods(asset_cls).items():
        required, optional = method_parameters(method)
        return_type = "Any"
        try:
            annotation = inspect.signature(method).return_annotation
            if annotation is not inspect._empty:
                return_type = _annotation_to_string(annotation)
        except (TypeError, ValueError):
            pass
        metadata[name] = ExportedMethodMetadata(
            name=name,
            required_args=required,
            optional_args=optional,
            return_type=return_type,
        )
    return metadata


def init_parameters(cls: type) -> tuple[list[AssetParameterMetadata], list[AssetParameterMetadata]]:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return [], []
    return _parameters_from_signature(signature)


def method_parameters(method: Callable) -> tuple[list[AssetParameterMetadata], list[AssetParameterMetadata]]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return [], []
    return _parameters_from_signature(signature)


def _parameters_from_signature(signature: inspect.Signature) -> tuple[list[AssetParameterMetadata], list[AssetParameterMetadata]]:
    required: list[AssetParameterMetadata] = []
    optional: list[AssetParameterMetadata] = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = "Any"
        if parameter.annotation is not inspect._empty:
            annotation = _annotation_to_string(parameter.annotation)
        has_default = parameter.default is not inspect._empty
        info = AssetParameterMetadata(
            name=name,
            type=annotation,
            default=None if not has_default else parameter.default,
            required=not has_default,
        )
        (optional if has_default else required).append(info)
    return required, optional


def _annotation_to_string(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))
