import inspect
from dataclasses import dataclass
from typing import Any, Type

from EasyCells3D.Assets import Asset
from EasyCells3D.ComponentDiscovery import AssetMetadata, ComponentMetadata, discover_assets, discover_components
from EasyCells3D.Components.Component import Component, Item


@dataclass
class ComponentCreationContext:
    game: Any
    item: Item
    node: dict
    scene_path: str
    objects_by_name: dict[str, Item]
    objects_by_node_index: dict[int, Item]
    objects_by_easycells_id: dict[str, Item] | None = None
    assets: dict[str, Any] | None = None
    components_by_id: dict[str, Component] | None = None
    component_registry: Any | None = None


class ComponentRegistry:
    _components: dict[str, Type[Component]] = {}
    _assets: dict[str, Type[Asset]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, name: str, component_cls: Type[Component]) -> Type[Component]:
        if not issubclass(component_cls, Component):
            raise TypeError(f"{component_cls!r} nao herda de Component")
        cls._components[name] = component_cls
        cls._components[f"{component_cls.__module__}.{component_cls.__name__}"] = component_cls
        return component_cls

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._components.pop(name, None)

    @classmethod
    def register_asset(cls, name: str, asset_cls: Type[Asset]) -> Type[Asset]:
        if not issubclass(asset_cls, Asset):
            raise TypeError(f"{asset_cls!r} nao herda de Asset")
        cls._assets[name] = asset_cls
        cls._assets[f"{asset_cls.__module__}.{asset_cls.__name__}"] = asset_cls
        return asset_cls

    @classmethod
    def unregister_asset(cls, name: str) -> None:
        cls._assets.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Type[Component] | None:
        return cls._components.get(name)

    @classmethod
    def all(cls) -> dict[str, Type[Component]]:
        return dict(cls._components)

    @classmethod
    def get_asset(cls, name: str) -> Type[Asset] | None:
        return cls._assets.get(name)

    @classmethod
    def all_assets(cls) -> dict[str, Type[Asset]]:
        return dict(cls._assets)

    @classmethod
    def discover(cls, project_root=None) -> dict[str, list]:
        components = discover_components(project_root=project_root, mode="runtime")
        assets = discover_assets(project_root=project_root, mode="runtime")
        cls.register_discovered(components)
        cls.register_discovered_assets(assets)
        cls._discovered = True
        return {"components": components, "assets": assets}

    @classmethod
    def register_discovered(cls, components: list[ComponentMetadata]) -> None:
        for metadata in components:
            if metadata.component_cls is None:
                continue
            cls.register(metadata.name, metadata.component_cls)
            cls.register(metadata.class_path, metadata.component_cls)

    @classmethod
    def register_discovered_assets(cls, assets: list[AssetMetadata]) -> None:
        for metadata in assets:
            if metadata.asset_cls is None:
                continue
            cls.register_asset(metadata.name, metadata.asset_cls)
            cls.register_asset(metadata.class_path, metadata.asset_cls)

    @classmethod
    def ensure_discovered(cls, project_root=None) -> None:
        if not cls._discovered:
            cls.discover(project_root=project_root)

    @classmethod
    def create(cls, type_name: str, args: dict | None, context: ComponentCreationContext | None = None) -> Component:
        component_cls = cls.get(type_name)
        if component_cls is None:
            raise KeyError(f"componente '{type_name}' nao registrado")

        args = args or {}

        try:
            return component_cls(**args)
        except TypeError as exc:
            signature = inspect.signature(component_cls)
            raise TypeError(
                f"nao foi possivel criar {type_name} com args {args}. "
                f"Construtor esperado: {type_name}{signature}"
            ) from exc

    @classmethod
    def create_asset(cls, type_name: str, args: dict | None) -> Asset:
        asset_cls = cls.get_asset(type_name)
        if asset_cls is None:
            raise KeyError(f"asset '{type_name}' nao registrado")

        args = args or {}

        try:
            return asset_cls(**args)
        except TypeError as exc:
            signature = inspect.signature(asset_cls)
            raise TypeError(
                f"nao foi possivel criar asset {type_name} com args {args}. "
                f"Construtor esperado: {type_name}{signature}"
            ) from exc
