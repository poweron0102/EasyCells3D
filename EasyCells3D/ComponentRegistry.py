import inspect
from dataclasses import dataclass
from typing import Any, Type

from EasyCells3D.Components.Component import Component, Item


@dataclass
class ComponentCreationContext:
    game: Any
    item: Item
    node: dict
    scene_path: str
    objects_by_name: dict[str, Item]
    objects_by_node_index: dict[int, Item]


class ComponentRegistry:
    _components: dict[str, Type[Component]] = {}

    @classmethod
    def register(cls, name: str, component_cls: Type[Component]) -> Type[Component]:
        if not issubclass(component_cls, Component):
            raise TypeError(f"{component_cls!r} nao herda de Component")
        cls._components[name] = component_cls
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
    def create(cls, type_name: str, config: dict | None, context: ComponentCreationContext) -> Component:
        component_cls = cls.get(type_name)
        if component_cls is None:
            raise KeyError(f"componente '{type_name}' nao registrado")

        config = config or {}
        factory = getattr(component_cls, "from_config", None)
        if callable(factory):
            return factory(config, context)

        try:
            return component_cls(**config)
        except TypeError as exc:
            signature = inspect.signature(component_cls)
            raise TypeError(
                f"nao foi possivel criar {type_name} com config {config}. "
                f"Construtor esperado: {type_name}{signature}"
            ) from exc
