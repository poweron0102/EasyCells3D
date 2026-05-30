from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_COMPONENT_PACKAGES = (
    "EasyCells3D.Components",
    "EasyCells3D.PhysicsComponents",
    "EasyCells3D.NetworkComponents",
    "UserComponents",
)

DEFAULT_ASSET_PACKAGES = (
    "EasyCells3D.AssetTypes",
    "UserAssets",
)


@dataclass
class ParameterMetadata:
    name: str
    type: str = "Any"
    default: Any = None
    required: bool = True


@dataclass
class SerializedFieldMetadata:
    name: str
    type: str = "Any"
    default: Any = None
    ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentMetadata:
    name: str
    class_path: str
    module: str
    required_args: list[ParameterMetadata] = field(default_factory=list)
    optional_args: list[ParameterMetadata] = field(default_factory=list)
    fields: dict[str, SerializedFieldMetadata] = field(default_factory=dict)
    component_cls: type | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("component_cls", None)
        return data


@dataclass
class ExportedMethodMetadata:
    name: str
    required_args: list[ParameterMetadata] = field(default_factory=list)
    optional_args: list[ParameterMetadata] = field(default_factory=list)
    return_type: str = "Any"


@dataclass
class AssetMetadata:
    name: str
    class_path: str
    module: str
    required_args: list[ParameterMetadata] = field(default_factory=list)
    optional_args: list[ParameterMetadata] = field(default_factory=list)
    methods: dict[str, ExportedMethodMetadata] = field(default_factory=dict)
    asset_cls: type | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("asset_cls", None)
        return data


def discover_components(
    project_root: str | Path | None = None,
    mode: str = "runtime",
    packages: tuple[str, ...] = DEFAULT_COMPONENT_PACKAGES,
) -> list[ComponentMetadata]:
    if mode == "ast":
        return discover_components_ast(project_root, packages)
    if mode == "runtime":
        return discover_components_runtime(project_root, packages)
    raise ValueError(f"modo de descoberta desconhecido: {mode}")


def discover_project_metadata(
    project_root: str | Path | None = None,
    mode: str = "runtime",
    component_packages: tuple[str, ...] = DEFAULT_COMPONENT_PACKAGES,
    asset_packages: tuple[str, ...] = DEFAULT_ASSET_PACKAGES,
) -> dict[str, list[ComponentMetadata] | list[AssetMetadata]]:
    return {
        "components": discover_components(project_root, mode, component_packages),
        "assets": discover_assets(project_root, mode, asset_packages),
    }


def discover_assets(
    project_root: str | Path | None = None,
    mode: str = "runtime",
    packages: tuple[str, ...] = DEFAULT_ASSET_PACKAGES,
) -> list[AssetMetadata]:
    if mode == "ast":
        return discover_assets_ast(project_root, packages)
    if mode == "runtime":
        return discover_assets_runtime(project_root, packages)
    raise ValueError(f"modo de descoberta desconhecido: {mode}")


def discover_components_runtime(
    project_root: str | Path | None = None,
    packages: tuple[str, ...] = DEFAULT_COMPONENT_PACKAGES,
) -> list[ComponentMetadata]:
    _ensure_project_root(project_root)

    from EasyCells3D.Components import Component
    from EasyCells3D.Serialization import get_serialized_fields

    discovered: dict[str, ComponentMetadata] = {}
    for module_name in _iter_runtime_modules(packages):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            warnings.warn(f"ComponentDiscovery: falha ao importar {module_name}: {exc}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Component or obj.__module__ != module.__name__:
                continue
            try:
                if not issubclass(obj, Component):
                    continue
            except TypeError:
                continue

            metadata = _metadata_from_runtime_class(obj, get_serialized_fields)
            discovered[metadata.class_path] = metadata

    return list(discovered.values())


def discover_components_ast(
    project_root: str | Path | None = None,
    packages: tuple[str, ...] = DEFAULT_COMPONENT_PACKAGES,
) -> list[ComponentMetadata]:
    root = Path(project_root or Path.cwd()).resolve()
    discovered: dict[str, ComponentMetadata] = {}

    for package in packages:
        package_dir = root.joinpath(*package.split("."))
        if not package_dir.exists():
            continue

        for filepath in package_dir.rglob("*.py"):
            if filepath.name == "__init__.py":
                continue
            module_name = _module_name_from_file(root, filepath)
            try:
                tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
            except Exception as exc:
                warnings.warn(f"ComponentDiscovery: falha ao analisar {filepath}: {exc}")
                continue

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and _inherits_component(node):
                    metadata = _metadata_from_ast_class(node, module_name)
                    discovered[metadata.class_path] = metadata

    return list(discovered.values())


def discover_assets_runtime(
    project_root: str | Path | None = None,
    packages: tuple[str, ...] = DEFAULT_ASSET_PACKAGES,
) -> list[AssetMetadata]:
    _ensure_project_root(project_root)

    from EasyCells3D.Assets import Asset, get_exported_method_metadata, init_parameters

    discovered: dict[str, AssetMetadata] = {}
    for module_name in _iter_runtime_modules(packages):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            warnings.warn(f"ComponentDiscovery: falha ao importar {module_name}: {exc}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Asset or obj.__module__ != module.__name__:
                continue
            try:
                if not issubclass(obj, Asset):
                    continue
            except TypeError:
                continue

            required_args, optional_args = init_parameters(obj)
            metadata = AssetMetadata(
                name=obj.__name__,
                class_path=f"{obj.__module__}.{obj.__name__}",
                module=obj.__module__,
                required_args=[
                    ParameterMetadata(arg.name, arg.type, arg.default, arg.required)
                    for arg in required_args
                ],
                optional_args=[
                    ParameterMetadata(arg.name, arg.type, arg.default, arg.required)
                    for arg in optional_args
                ],
                methods={
                    name: ExportedMethodMetadata(
                        name=method.name,
                        required_args=[
                            ParameterMetadata(arg.name, arg.type, arg.default, arg.required)
                            for arg in method.required_args
                        ],
                        optional_args=[
                            ParameterMetadata(arg.name, arg.type, arg.default, arg.required)
                            for arg in method.optional_args
                        ],
                        return_type=method.return_type,
                    )
                    for name, method in get_exported_method_metadata(obj).items()
                },
                asset_cls=obj,
            )
            discovered[metadata.class_path] = metadata

    return list(discovered.values())


def discover_assets_ast(
    project_root: str | Path | None = None,
    packages: tuple[str, ...] = DEFAULT_ASSET_PACKAGES,
) -> list[AssetMetadata]:
    root = Path(project_root or Path.cwd()).resolve()
    discovered: dict[str, AssetMetadata] = {}

    for package in packages:
        package_dir = root.joinpath(*package.split("."))
        if not package_dir.exists():
            continue

        for filepath in package_dir.rglob("*.py"):
            if filepath.name == "__init__.py":
                continue
            module_name = _module_name_from_file(root, filepath)
            try:
                tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
            except Exception as exc:
                warnings.warn(f"ComponentDiscovery: falha ao analisar {filepath}: {exc}")
                continue

            for node in tree.body:
                if isinstance(node, ast.ClassDef) and _inherits_asset(node):
                    metadata = _asset_metadata_from_ast_class(node, module_name)
                    discovered[metadata.class_path] = metadata

    return list(discovered.values())


def components_to_dicts(components: list[ComponentMetadata]) -> list[dict[str, Any]]:
    return [component.to_dict() for component in components]


def assets_to_dicts(assets: list[AssetMetadata]) -> list[dict[str, Any]]:
    return [asset.to_dict() for asset in assets]


def _ensure_project_root(project_root: str | Path | None) -> None:
    if project_root is None:
        return
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _iter_runtime_modules(packages: tuple[str, ...]):
    for package_name in packages:
        yield package_name
        try:
            package = importlib.import_module(package_name)
        except Exception as exc:
            warnings.warn(f"ComponentDiscovery: falha ao importar pacote {package_name}: {exc}")
            continue
        if not hasattr(package, "__path__"):
            continue
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            yield module_name


def _metadata_from_runtime_class(component_cls: type, get_serialized_fields) -> ComponentMetadata:
    required_args, optional_args = _runtime_init_parameters(component_cls)
    fields = {
        name: SerializedFieldMetadata(
            name=name,
            type=field.field_type,
            default=field.default,
            ref=field.ref,
            metadata={key: value for key, value in field.to_metadata().items() if key not in {"type", "default", "ref"}},
        )
        for name, field in get_serialized_fields(component_cls).items()
    }
    return ComponentMetadata(
        name=component_cls.__name__,
        class_path=f"{component_cls.__module__}.{component_cls.__name__}",
        module=component_cls.__module__,
        required_args=required_args,
        optional_args=optional_args,
        fields=fields,
        component_cls=component_cls,
    )


def _runtime_init_parameters(component_cls: type) -> tuple[list[ParameterMetadata], list[ParameterMetadata]]:
    try:
        signature = inspect.signature(component_cls.__init__)
    except (TypeError, ValueError):
        return [], []

    required: list[ParameterMetadata] = []
    optional: list[ParameterMetadata] = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = "Any"
        if parameter.annotation is not inspect._empty:
            annotation = getattr(parameter.annotation, "__name__", str(parameter.annotation))
        has_default = parameter.default is not inspect._empty
        info = ParameterMetadata(
            name=name,
            type=annotation,
            default=None if not has_default else parameter.default,
            required=not has_default,
        )
        (optional if has_default else required).append(info)
    return required, optional


def _metadata_from_ast_class(node: ast.ClassDef, module_name: str) -> ComponentMetadata:
    required_args: list[ParameterMetadata] = []
    optional_args: list[ParameterMetadata] = []
    fields: dict[str, SerializedFieldMetadata] = {}

    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            required_args, optional_args = _ast_init_parameters(child)
        field_name, field_data = _ast_serialized_field(child)
        if field_name and field_data:
            fields[field_name] = field_data

    return ComponentMetadata(
        name=node.name,
        class_path=f"{module_name}.{node.name}",
        module=module_name,
        required_args=required_args,
        optional_args=optional_args,
        fields=fields,
    )


def _asset_metadata_from_ast_class(node: ast.ClassDef, module_name: str) -> AssetMetadata:
    required_args: list[ParameterMetadata] = []
    optional_args: list[ParameterMetadata] = []
    methods: dict[str, ExportedMethodMetadata] = {}

    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            required_args, optional_args = _ast_init_parameters(child)
        if isinstance(child, ast.FunctionDef) and _has_export_decorator(child):
            required, optional = _ast_init_parameters(child)
            methods[child.name] = ExportedMethodMetadata(
                name=child.name,
                required_args=required,
                optional_args=optional,
                return_type=_annotation_to_string(child.returns),
            )

    return AssetMetadata(
        name=node.name,
        class_path=f"{module_name}.{node.name}",
        module=module_name,
        required_args=required_args,
        optional_args=optional_args,
        methods=methods,
    )


def _ast_init_parameters(node: ast.FunctionDef) -> tuple[list[ParameterMetadata], list[ParameterMetadata]]:
    args = node.args.posonlyargs + node.args.args
    if args and args[0].arg == "self":
        args = args[1:]

    defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
    required: list[ParameterMetadata] = []
    optional: list[ParameterMetadata] = []
    for arg, default_node in zip(args, defaults):
        has_default = default_node is not None
        info = ParameterMetadata(
            name=arg.arg,
            type=_annotation_to_string(arg.annotation),
            default=_literal_or_source(default_node) if has_default else None,
            required=not has_default,
        )
        (optional if has_default else required).append(info)

    for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults):
        has_default = default_node is not None
        info = ParameterMetadata(
            name=arg.arg,
            type=_annotation_to_string(arg.annotation),
            default=_literal_or_source(default_node) if has_default else None,
            required=not has_default,
        )
        (optional if has_default else required).append(info)

    return required, optional


def _ast_serialized_field(node: ast.AST) -> tuple[str | None, SerializedFieldMetadata | None]:
    target: ast.AST | None = None
    value: ast.AST | None = None
    annotation: ast.AST | None = None

    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        value = node.value
        annotation = node.annotation

    if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
        return None, None
    if _call_name(value.func) != "SerializeField":
        return None, None

    data: dict[str, Any] = {}
    if value.args:
        data["default"] = _literal_or_source(value.args[0])
    for keyword in value.keywords:
        if keyword.arg:
            data[keyword.arg] = _literal_or_source(keyword.value)

    field_type = data.pop("field_type", None) or data.pop("type", None) or _annotation_to_string(annotation)
    ref = data.pop("ref", None)
    default = data.pop("default", None)
    if field_type == "Any":
        field_type = _infer_type(default, ref)
    return target.id, SerializedFieldMetadata(
        name=target.id,
        type=str(field_type),
        default=default,
        ref=ref,
        metadata=data,
    )


def _inherits_component(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _call_name(base) == "Component":
            return True
    return False


def _inherits_asset(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _call_name(base) == "Asset":
            return True
    return False


def _has_export_decorator(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        if _call_name(decorator) == "export":
            return True
    return False


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return ""


def _annotation_to_string(node: ast.AST | None) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _literal_or_source(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


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


def _module_name_from_file(root: Path, filepath: Path) -> str:
    rel = filepath.resolve().relative_to(root)
    return ".".join(rel.with_suffix("").parts)
