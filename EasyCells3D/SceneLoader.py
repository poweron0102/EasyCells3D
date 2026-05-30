import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from EasyCells3D.Assets import Asset
from EasyCells3D.ComponentRegistry import ComponentCreationContext, ComponentRegistry
from EasyCells3D.Components import Item, Transform
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.Serialization import get_serialized_fields


UINT128_MAX = 340282366920938463463374607431768211455


@dataclass
class _PendingSerializedFields:
    item: Item
    component: Any
    component_def: dict
    context: ComponentCreationContext


@dataclass
class _SceneContext:
    scene_path: Path
    assets: dict[str, Any]
    objects_by_name: dict[str, Item]
    objects_by_node_index: dict[int, Item]
    objects_by_easycells_id: dict[str, Item]
    components_by_id: dict[str, Any]
    components_by_item_id: dict[str, list[Any]]


class SceneLoader:
    def __init__(self, game, component_registry: type[ComponentRegistry] = ComponentRegistry):
        self.game = game
        self.component_registry = component_registry

    def load(self, path: str | Path) -> list[Item]:
        scene_path = Path(path)
        document = _load_scene_document(scene_path)
        self.component_registry.ensure_discovered(_find_project_root(scene_path))

        _validate_scene_header(document, scene_path)
        item_defs = document["Item"]
        _validate_ids_and_parents(item_defs)

        context = _SceneContext(
            scene_path=scene_path,
            assets=self._create_global_assets(document.get("assets") or {}),
            objects_by_name={},
            objects_by_node_index={},
            objects_by_easycells_id={},
            components_by_id={},
            components_by_item_id={},
        )

        items_by_id = self._create_items(item_defs, context)
        self._apply_hierarchy(item_defs, items_by_id)
        pending_fields = self._create_components(item_defs, items_by_id, context)

        for pending in pending_fields:
            self._apply_serialized_fields(pending)

        return [items_by_id[item_def["id"]] for item_def in item_defs if item_def.get("parent") is None]

    def _create_global_assets(self, asset_defs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(asset_defs, dict):
            raise ValueError("SceneLoader: assets deve ser objeto")

        assets: dict[str, Any] = {}
        for name, asset_def in asset_defs.items():
            if not isinstance(asset_def, dict):
                raise ValueError(f"SceneLoader: asset '{name}' deve ser objeto")
            assets[str(name)] = self._create_asset_from_definition(asset_def)
        return assets

    def _create_asset_from_definition(self, asset_def: dict[str, Any]) -> Any:
        type_name = asset_def.get("type")
        args = asset_def.get("args") or {}
        if type_name and self.component_registry.get_asset(str(type_name)) is not None:
            if not isinstance(args, dict):
                raise ValueError(f"SceneLoader: args do asset '{type_name}' deve ser objeto")
            return self.component_registry.create_asset(str(type_name), args)
        return dict(asset_def)

    def _create_items(self, item_defs: list[dict[str, Any]], context: _SceneContext) -> dict[str, Item]:
        items_by_id: dict[str, Item] = {}
        for index, item_def in enumerate(item_defs):
            item_id = str(item_def["id"])
            item = self.game.CreateItem()
            item.name = str(item_def["name"])
            item.easycells_id = item_id
            item.enabled = bool(item_def.get("enabled", True))
            item.transform = _scene_transform(item_def.get("transform") or {})

            items_by_id[item_id] = item
            context.objects_by_node_index[index] = item
            context.objects_by_easycells_id[item_id] = item
            context.objects_by_name[_unique_name(item.name, context.objects_by_name)] = item
        return items_by_id

    def _apply_hierarchy(self, item_defs: list[dict[str, Any]], items_by_id: dict[str, Item]) -> None:
        for item_def in item_defs:
            parent_id = item_def.get("parent")
            if parent_id is None:
                continue
            item = items_by_id[str(item_def["id"])]
            parent = items_by_id[str(parent_id)]
            parent.AddChild(item)

    def _create_components(
        self,
        item_defs: list[dict[str, Any]],
        items_by_id: dict[str, Item],
        scene_context: _SceneContext,
    ) -> list[_PendingSerializedFields]:
        pending_fields: list[_PendingSerializedFields] = []
        for index, item_def in enumerate(item_defs):
            item = items_by_id[str(item_def["id"])]
            components = item_def.get("components") or []
            if not isinstance(components, list):
                raise ValueError(f"SceneLoader: components deve ser lista em '{item.name}'")

            creation_context = ComponentCreationContext(
                game=self.game,
                item=item,
                node=item_def,
                scene_path=str(scene_context.scene_path),
                objects_by_name=scene_context.objects_by_name,
                objects_by_node_index=scene_context.objects_by_node_index,
                objects_by_easycells_id=scene_context.objects_by_easycells_id,
                assets=scene_context.assets,
                components_by_id=scene_context.components_by_id,
                component_registry=self.component_registry,
            )

            for component_def in components:
                if not isinstance(component_def, dict):
                    raise ValueError(f"SceneLoader: componente invalido em '{item.name}'")
                component_id = str(component_def["id"])
                type_name = component_def.get("type")
                if not type_name:
                    raise ValueError(f"SceneLoader: componente sem 'type' em '{item.name}'")

                args = component_def.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError(f"SceneLoader: args deve ser objeto em '{item.name}' ({type_name})")
                resolved_args = _resolve_serialized_value(args, None, creation_context, allow_scene_refs=False)
                component = self.component_registry.create(str(type_name), resolved_args, creation_context)
                component.enable = bool(component_def.get("enabled", True))
                component.easycells_id = component_id
                item.AddComponent(component)

                scene_context.components_by_id[component_id] = component
                scene_context.components_by_item_id.setdefault(str(item_def["id"]), []).append(component)
                pending_fields.append(_PendingSerializedFields(item, component, component_def, creation_context))

        return pending_fields

    def _apply_serialized_fields(self, pending: _PendingSerializedFields) -> None:
        fields = pending.component_def.get("fields") or {}
        if not isinstance(fields, dict):
            raise ValueError(
                f"SceneLoader: fields deve ser objeto em '{pending.item.name}' "
                f"({pending.component.__class__.__name__})"
            )

        declared_fields = get_serialized_fields(pending.component.__class__)
        for name, raw_value in fields.items():
            field = declared_fields.get(name)
            value = _resolve_serialized_value(raw_value, field, pending.context, allow_scene_refs=True)
            value = _coerce_serialized_value(value, field)
            setattr(pending.component, name, value)


def _load_scene_document(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".ecscene":
        raise ValueError(f"SceneLoader suporta apenas .ecscene v2: {path}")
    with path.open("r", encoding="utf-8") as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError(f"SceneLoader: documento invalido: {path}")
    return document


def _validate_scene_header(document: dict[str, Any], scene_path: Path) -> None:
    if document.get("format") != "easycells3d.scene":
        raise ValueError(f"arquivo nao e uma cena EasyCells3D v2: {scene_path}")
    if document.get("version") != 2:
        raise ValueError(f"SceneLoader: versao de cena nao suportada: {document.get('version')}")
    if "Item" not in document or not isinstance(document["Item"], list):
        raise ValueError("SceneLoader: campo 'Item' deve existir e ser lista")
    for item_def in document["Item"]:
        if not isinstance(item_def, dict):
            raise ValueError("SceneLoader: cada Item deve ser objeto")
        if "id" not in item_def:
            raise ValueError("SceneLoader: Item sem id")
        if "name" not in item_def:
            raise ValueError(f"SceneLoader: Item '{item_def.get('id')}' sem name")
        if "parent" not in item_def:
            raise ValueError(f"SceneLoader: Item '{item_def.get('id')}' sem parent")


def _validate_ids_and_parents(item_defs: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    item_ids: set[str] = set()
    for item_def in item_defs:
        item_id = _validate_uint128_id(item_def["id"], "Item.id")
        if item_id in ids:
            raise ValueError(f"SceneLoader: id duplicado: {item_id}")
        ids.add(item_id)
        item_ids.add(item_id)

        components = item_def.get("components") or []
        if not isinstance(components, list):
            raise ValueError(f"SceneLoader: components deve ser lista em '{item_def.get('name')}'")
        for component_def in components:
            if not isinstance(component_def, dict):
                raise ValueError(f"SceneLoader: componente invalido em '{item_def.get('name')}'")
            if "id" not in component_def:
                raise ValueError(f"SceneLoader: componente sem id em '{item_def.get('name')}'")
            component_id = _validate_uint128_id(component_def["id"], "Component.id")
            if component_id in ids:
                raise ValueError(f"SceneLoader: id duplicado: {component_id}")
            ids.add(component_id)

    for item_def in item_defs:
        parent = item_def.get("parent")
        if parent is None:
            continue
        parent_id = _validate_uint128_id(parent, "Item.parent")
        if parent_id not in item_ids:
            raise KeyError(f"SceneLoader: parent nao encontrado: {parent_id}")


def _validate_uint128_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError(f"SceneLoader: {field_name} deve ser UInt128 decimal string: {value!r}")
    number = int(value)
    if number < 0 or number > UINT128_MAX:
        raise ValueError(f"SceneLoader: {field_name} fora do intervalo UInt128: {value!r}")
    return value


def _scene_transform(transform_def: dict[str, Any]) -> Transform:
    if not isinstance(transform_def, dict):
        raise ValueError("SceneLoader: transform deve ser objeto")
    position = _vec3(transform_def.get("position"), [0.0, 0.0, 0.0], "transform.position")
    rotation = _vec3(transform_def.get("rotation"), [0.0, 0.0, 0.0], "transform.rotation")
    scale = _vec3(transform_def.get("scale"), [1.0, 1.0, 1.0], "transform.scale")
    return Transform(
        Vec3(position[0], position[1], position[2]),
        Quaternion.from_euler_angles(Vec3(
            math.radians(rotation[0]),
            math.radians(rotation[1]),
            math.radians(rotation[2]),
        )),
        Vec3(scale[0], scale[1], scale[2]),
    )


def _vec3(value: Any, default: list[float], field_name: str) -> list[float]:
    if value is None:
        return list(default)
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"SceneLoader: {field_name} deve ter 3 numeros")
    return [float(value[0]), float(value[1]), float(value[2])]


def _resolve_serialized_value(
    value: Any,
    field,
    context: ComponentCreationContext,
    allow_scene_refs: bool = True,
) -> Any:
    if isinstance(value, list):
        return [_resolve_serialized_value(entry, field, context, allow_scene_refs) for entry in value]
    if not isinstance(value, dict):
        return value

    if "$assetRef" in value:
        asset = _resolve_asset_ref(value, context)
        return _apply_asset_selector(asset, value.get("selector"))

    registry = context.component_registry or ComponentRegistry
    if _is_inline_asset(value, registry):
        type_name = value.get("type")
        asset = (
            registry.create_asset(str(type_name), value.get("args") or {})
            if type_name and registry.get_asset(str(type_name))
            else dict(value)
        )
        return _apply_asset_selector(asset, value.get("selector"))

    if "$componentRef" in value:
        if not allow_scene_refs:
            raise ValueError("SceneLoader: args nao pode conter ComponentRef")
        component_id = str(value["$componentRef"])
        component = (context.components_by_id or {}).get(component_id)
        if component is None:
            raise KeyError(f"SceneLoader: componentRef nao encontrado: {component_id}")
        return component

    reference = value.get("$id") or value.get("$ref")
    if reference:
        if not allow_scene_refs:
            raise ValueError("SceneLoader: args nao pode conter ItemRef ou ComponentRef")
        item = None
        if context.objects_by_easycells_id:
            item = context.objects_by_easycells_id.get(str(reference))
        if item is None:
            item = context.objects_by_name.get(str(reference))
        if item is None:
            raise KeyError(f"SceneLoader: referencia nao encontrada: {reference}")

        component_type = value.get("$component")
        if component_type:
            return _find_component_on_item(item, str(component_type))
        if field is not None and field.ref == "component":
            component_type = value.get("type") or value.get("component")
            if component_type:
                return _find_component_on_item(item, str(component_type))
        return item

    return {
        key: _resolve_serialized_value(entry, field, context, allow_scene_refs)
        for key, entry in value.items()
    }


def _resolve_asset_ref(value: dict[str, Any], context: ComponentCreationContext) -> Any:
    asset_name = str(value["$assetRef"])
    if not context.assets or asset_name not in context.assets:
        raise KeyError(f"SceneLoader: assetRef nao encontrado: {asset_name}")
    return context.assets[asset_name]


def _is_inline_asset(value: dict[str, Any], registry: type[ComponentRegistry]) -> bool:
    if "$asset" in value:
        return True
    type_name = value.get("type")
    return bool(type_name and "args" in value and registry.get_asset(str(type_name)) is not None)


def _apply_asset_selector(asset: Any, selector: Any) -> Any:
    if selector is None:
        return asset
    if not isinstance(selector, dict):
        raise ValueError("SceneLoader: selector deve ser objeto")
    method_name = selector.get("method")
    if not method_name:
        raise ValueError("SceneLoader: selector deve conter 'method'")
    if not isinstance(asset, Asset):
        raise TypeError(f"SceneLoader: selector method requer Asset runtime, recebido {type(asset).__name__}")

    method = getattr(asset, str(method_name), None)
    raw_method = getattr(method, "__func__", method)
    if method is None or not getattr(raw_method, "__easycells_export__", False):
        raise KeyError(f"SceneLoader: metodo de asset nao exportado: {method_name}")

    args = selector.get("args") or {}
    if isinstance(args, dict):
        return method(**args)
    if isinstance(args, list):
        return method(*args)
    raise ValueError("SceneLoader: selector.args deve ser objeto ou lista")


def _find_component_on_item(item: Item, component_type: str):
    matches = []
    for component in set(item.components.values()):
        cls = component.__class__
        if component_type in {cls.__name__, f"{cls.__module__}.{cls.__name__}"}:
            matches.append(component)
    if len(matches) != 1:
        raise KeyError(
            f"componente '{component_type}' deve existir exatamente uma vez em '{item.name}', "
            f"encontrados {len(matches)}"
        )
    return matches[0]


def _coerce_serialized_value(value: Any, field) -> Any:
    if field is None:
        return value
    field_type = field.field_type
    if field_type == "bool" and not isinstance(value, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "sim"}
        return bool(value)
    if field_type == "int" and value is not None:
        return int(value)
    if field_type == "float" and value is not None:
        return float(value)
    if field_type in {"str", "string"} and value is not None:
        return str(value)
    return value


def _unique_name(name: str, existing: dict[str, Item]) -> str:
    if name not in existing:
        return name
    index = 2
    while f"{name}.{index}" in existing:
        index += 1
    return f"{name}.{index}"


def _find_project_root(scene_path: Path) -> Path:
    candidates = [Path.cwd()]
    try:
        candidates.extend(scene_path.resolve().parents)
    except OSError:
        candidates.extend(scene_path.absolute().parents)

    for candidate in candidates:
        if (candidate / "EasyCells3D").is_dir():
            return candidate
    return Path.cwd()
