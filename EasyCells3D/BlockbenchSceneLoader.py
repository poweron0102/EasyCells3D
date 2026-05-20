import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from EasyCells3D.ComponentRegistry import ComponentCreationContext, ComponentRegistry
from EasyCells3D.Components import Item, Transform
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.SceneLoader import (
    _PendingSerializedFields,
    _coerce_serialized_value,
    _find_project_root,
    _resolve_serialized_value,
    _unique_name,
)
from EasyCells3D.Serialization import get_serialized_fields


@dataclass
class _BlockbenchNode:
    index: int
    data: dict[str, Any]


class BlockbenchSceneLoader:
    def __init__(self, game, component_registry: type[ComponentRegistry] = ComponentRegistry):
        self.game = game
        self.component_registry = component_registry

    def load(self, path: str | Path, model_path: str | Path | None = None) -> list[Item]:
        scene_path = Path(path)
        with scene_path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        if document.get("format") != "easycells3d.blockbench.scene":
            raise ValueError(f"arquivo nao e uma cena Blockbench EasyCells3D: {scene_path}")

        self.component_registry.ensure_discovered(_find_project_root(scene_path))
        nodes = [_BlockbenchNode(index, node) for index, node in enumerate(document.get("nodes", []))]
        nodes_by_id = {str(node.data.get("id")): node for node in nodes if node.data.get("id")}
        children_by_parent: dict[str | None, list[_BlockbenchNode]] = {}
        for node in nodes:
            parent_id = node.data.get("parent")
            children_by_parent.setdefault(str(parent_id) if parent_id else None, []).append(node)

        objects_by_name: dict[str, Item] = {}
        objects_by_node_index: dict[int, Item] = {}
        objects_by_easycells_id: dict[str, Item] = {}
        visual_roots = self._load_visual_items(scene_path, document, model_path)

        for item in _walk_items(visual_roots):
            unique_name = _unique_name(item.name, objects_by_name)
            objects_by_name[unique_name] = item
            if getattr(item, "easycells_id", None):
                objects_by_easycells_id[str(item.easycells_id)] = item

        def create_node(node: _BlockbenchNode, parent: Item | None = None) -> Item:
            item = _match_visual_item(node.data, objects_by_name, objects_by_easycells_id)
            if item is None:
                item = parent.CreateChild() if parent else self.game.CreateItem()
                item.name = node.data.get("name") or f"BlockbenchNode_{node.index}"
                item.transform = _node_transform(node.data)
            elif parent is not None and item.parent is None:
                item.SetParent(parent)

            unique_name = _unique_name(item.name, objects_by_name)
            if unique_name not in objects_by_name:
                objects_by_name[unique_name] = item
            objects_by_node_index[node.index] = item

            easycells_id = node.data.get("id")
            if easycells_id:
                item.easycells_id = str(easycells_id)
                objects_by_easycells_id[str(easycells_id)] = item

            for child in children_by_parent.get(str(easycells_id), []):
                create_node(child, item)

            return item

        root_items = visual_roots or [create_node(node) for node in children_by_parent.get(None, [])]
        if visual_roots:
            for node in children_by_parent.get(None, []):
                create_node(node)

        pending_fields: list[_PendingSerializedFields] = []
        for node in nodes:
            item = objects_by_node_index.get(node.index)
            if item is None:
                continue
            context = ComponentCreationContext(
                game=self.game,
                item=item,
                node=node.data,
                scene_path=str(scene_path),
                objects_by_name=objects_by_name,
                objects_by_node_index=objects_by_node_index,
                objects_by_easycells_id=objects_by_easycells_id,
            )
            pending_fields.extend(self._add_components(item, node.data, context))

        for pending in pending_fields:
            self._apply_fields(pending)

        missing_parent_nodes = [
            node for node in nodes
            if node.data.get("parent") and str(node.data.get("parent")) not in nodes_by_id
        ]
        for node in missing_parent_nodes:
            warnings.warn(
                f"BlockbenchSceneLoader: parent ausente em '{node.data.get('name', node.index)}': "
                f"{node.data.get('parent')}"
            )

        return root_items

    def _load_visual_items(
        self,
        scene_path: Path,
        document: dict[str, Any],
        model_path: str | Path | None,
    ) -> list[Item]:
        visual_model = model_path or (document.get("visual") or {}).get("model")
        if not visual_model:
            return []

        visual_path = Path(visual_model)
        if not visual_path.is_absolute():
            root_candidate = _find_project_root(scene_path)
            project_relative = root_candidate / visual_path
            visual_path = project_relative if project_relative.exists() else scene_path.parent / visual_path

        from EasyCells3D.SceneLoader import SceneLoader

        try:
            return SceneLoader(self.game, self.component_registry).load(visual_path)
        except Exception as exc:
            warnings.warn(f"BlockbenchSceneLoader: falha ao carregar visual GLB '{visual_path}': {exc}")
            return []

    def _add_components(
        self,
        item: Item,
        node: dict[str, Any],
        context: ComponentCreationContext,
    ) -> list[_PendingSerializedFields]:
        pending: list[_PendingSerializedFields] = []
        components = node.get("components") or []
        if not isinstance(components, list):
            warnings.warn(f"BlockbenchSceneLoader: components deve ser lista em '{item.name}'")
            return pending

        for component_def in components:
            if not isinstance(component_def, dict):
                continue
            type_name = component_def.get("type")
            args = component_def.get("args") or {}
            if not type_name:
                warnings.warn(f"BlockbenchSceneLoader: componente sem 'type' em '{item.name}'")
                continue
            try:
                component = self.component_registry.create(type_name, args, context)
                item.AddComponent(component)
                pending.append(_PendingSerializedFields(item, component, component_def, context))
            except Exception as exc:
                warnings.warn(
                    f"BlockbenchSceneLoader: erro ao criar componente '{type_name}' "
                    f"em '{item.name}': {exc}"
                )
        return pending

    def _apply_fields(self, pending: _PendingSerializedFields) -> None:
        fields = pending.component_def.get("fields") or {}
        if not isinstance(fields, dict):
            return
        declared_fields = get_serialized_fields(pending.component.__class__)
        for name, raw_value in fields.items():
            field = declared_fields.get(name)
            try:
                value = _resolve_serialized_value(raw_value, field, pending.context)
                value = _coerce_serialized_value(value, field)
                setattr(pending.component, name, value)
            except Exception as exc:
                warnings.warn(
                    f"BlockbenchSceneLoader: erro ao aplicar field '{name}' em "
                    f"{pending.component.__class__.__name__} no objeto '{pending.item.name}': {exc}"
                )


def _node_transform(node: dict[str, Any]) -> Transform:
    translation = node.get("translation") or [0.0, 0.0, 0.0]
    rotation = node.get("rotation") or node.get("rotation_euler_degrees") or [0.0, 0.0, 0.0]
    scale = node.get("scale") or [1.0, 1.0, 1.0]

    if len(rotation) == 4:
        quat = Quaternion(float(rotation[3]), float(rotation[0]), float(rotation[1]), float(rotation[2])).normalize()
    else:
        quat = Quaternion.from_euler_angles(Vec3(
            math.radians(float(rotation[0])),
            math.radians(float(rotation[1])),
            math.radians(float(rotation[2])),
        ))

    return Transform(
        Vec3(float(translation[0]), float(translation[1]), float(translation[2])),
        quat,
        Vec3(float(scale[0]), float(scale[1]), float(scale[2])),
    )


def _walk_items(items: list[Item]) -> list[Item]:
    result: list[Item] = []
    for item in items:
        result.append(item)
        result.extend(_walk_items(list(item.children)))
    return result


def _match_visual_item(
    node: dict[str, Any],
    objects_by_name: dict[str, Item],
    objects_by_easycells_id: dict[str, Item],
) -> Item | None:
    easycells_id = node.get("id")
    if easycells_id and str(easycells_id) in objects_by_easycells_id:
        return objects_by_easycells_id[str(easycells_id)]

    name = node.get("name")
    if name:
        item = objects_by_name.get(str(name))
        if item is not None:
            return item
        for existing_name, existing_item in objects_by_name.items():
            if existing_name.startswith(f"{name}."):
                return existing_item
    return None
