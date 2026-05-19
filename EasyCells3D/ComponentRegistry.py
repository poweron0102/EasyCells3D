import inspect
from dataclasses import dataclass
from typing import Any, Type

from EasyCells3D.ComponentDiscovery import ComponentMetadata, discover_components
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


class ComponentRegistry:
    _components: dict[str, Type[Component]] = {}
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
    def get(cls, name: str) -> Type[Component] | None:
        return cls._components.get(name)

    @classmethod
    def all(cls) -> dict[str, Type[Component]]:
        return dict(cls._components)

    @classmethod
    def discover(cls, project_root=None) -> list[ComponentMetadata]:
        components = discover_components(project_root=project_root, mode="runtime")
        cls.register_discovered(components)
        cls._discovered = True
        return components

    @classmethod
    def register_discovered(cls, components: list[ComponentMetadata]) -> None:
        for metadata in components:
            if metadata.component_cls is None:
                continue
            cls.register(metadata.name, metadata.component_cls)
            cls.register(metadata.class_path, metadata.component_cls)

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
