import json
import math
import struct
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from EasyCells3D.ComponentRegistry import ComponentCreationContext, ComponentRegistry
from EasyCells3D.Components import Item, StaticModel, Transform
from EasyCells3D.Geometry import Quaternion, Vec3


class SceneLoader:
    def __init__(self, game, component_registry: type[ComponentRegistry] = ComponentRegistry):
        self.game = game
        self.component_registry = component_registry

    def load(self, path: str | Path) -> list[Item]:
        scene_path = Path(path)
        document = _load_gltf_document(scene_path)
        nodes = document.get("nodes", [])
        scene_index = document.get("scene", 0)
        scenes = document.get("scenes", [])
        root_node_indices = scenes[scene_index].get("nodes", []) if scenes else list(range(len(nodes)))

        objects_by_name: dict[str, Item] = {}
        objects_by_node_index: dict[int, Item] = {}

        mesh_draw_indices: dict[int, int] = {}
        next_draw_index = 0

        def create_node(node_index: int, parent: Item | None = None) -> Item:
            nonlocal next_draw_index
            node = nodes[node_index]
            item = parent.CreateChild() if parent else self.game.CreateItem()
            item.name = node.get("name") or f"Node_{node_index}"
            item.transform = _node_transform(node)

            unique_name = _unique_name(item.name, objects_by_name)
            objects_by_name[unique_name] = item
            objects_by_node_index[node_index] = item

            if "mesh" in node:
                mesh_index = int(node["mesh"])
                if mesh_index not in mesh_draw_indices:
                    mesh_draw_indices[mesh_index] = next_draw_index
                    next_draw_index += _primitive_count(document, mesh_index)
                primitive_count = _primitive_count(document, mesh_index)
                mesh_indices = list(range(mesh_draw_indices[mesh_index], mesh_draw_indices[mesh_index] + primitive_count))
                item.AddComponent(StaticModel(
                    str(scene_path),
                    base_path=".",
                    mesh_index=mesh_indices,
                    shared=True,
                ))

            for child_index in node.get("children", []):
                create_node(child_index, item)

            return item

        root_items = [create_node(node_index) for node_index in root_node_indices]
        if root_items and mesh_draw_indices:
            root_items[0].AddComponent(StaticModel(
                str(scene_path),
                base_path=".",
                shared=True,
                fallback_only=True,
            ))

        for node_index, item in list(objects_by_node_index.items()):
            context = ComponentCreationContext(
                game=self.game,
                item=item,
                node=nodes[node_index],
                scene_path=str(scene_path),
                objects_by_name=objects_by_name,
                objects_by_node_index=objects_by_node_index,
            )
            self._add_configured_components(item, nodes[node_index], context)

        return root_items

    def _add_configured_components(self, item: Item, node: dict, context: ComponentCreationContext) -> None:
        for component_def in _extract_component_defs(node):
            type_name = component_def.get("type")
            config = component_def.get("config", {})
            if not type_name:
                warnings.warn(f"SceneLoader: componente sem 'type' no objeto '{item.name}'")
                continue

            try:
                component = self.component_registry.create(type_name, config, context)
                item.AddComponent(component)
            except Exception as exc:
                warnings.warn(
                    f"SceneLoader: erro ao criar componente '{type_name}' no objeto "
                    f"'{item.name}': {exc}"
                )


def _load_gltf_document(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".gltf":
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    if suffix == ".glb":
        return _load_glb_json(path)
    raise ValueError(f"SceneLoader suporta apenas .glb e .gltf: {path}")


def _load_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"arquivo GLB invalido: {path}")
    if version != 2:
        raise ValueError(f"SceneLoader suporta GLB versao 2, recebido {version}: {path}")
    if length != len(data):
        raise ValueError(f"tamanho GLB inconsistente: {path}")

    offset = 12
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            return json.loads(chunk.decode("utf-8").rstrip("\x00 "))

    raise ValueError(f"GLB sem chunk JSON: {path}")


def _node_transform(node: dict) -> Transform:
    if "matrix" in node:
        return _transform_from_matrix(node["matrix"])

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])

    return Transform(
        Vec3(float(translation[0]), float(translation[1]), float(translation[2])),
        Quaternion(float(rotation[3]), float(rotation[0]), float(rotation[1]), float(rotation[2])).normalize(),
        Vec3(float(scale[0]), float(scale[1]), float(scale[2])),
    )


def _transform_from_matrix(values: list[float]) -> Transform:
    matrix = np.array(values, dtype=float).reshape((4, 4), order="F")
    translation = matrix[:3, 3]

    basis = matrix[:3, :3].copy()
    scale = np.linalg.norm(basis, axis=0)
    for index in range(3):
        if scale[index] != 0:
            basis[:, index] /= scale[index]

    return Transform(
        Vec3(float(translation[0]), float(translation[1]), float(translation[2])),
        _quaternion_from_rotation_matrix(basis),
        Vec3(float(scale[0]), float(scale[1]), float(scale[2])),
    )


def _quaternion_from_rotation_matrix(matrix: np.ndarray) -> Quaternion:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ).normalize()

    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return Quaternion((matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s).normalize()
    if axis == 1:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return Quaternion((matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s).normalize()

    s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return Quaternion((matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s).normalize()


def _extract_component_defs(node: dict) -> list[dict]:
    extras = node.get("extras") or {}
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except json.JSONDecodeError:
            return []

    raw_components = extras.get("components") or extras.get("easycells_components")
    if raw_components is None and isinstance(extras.get("EasyCells3D"), dict):
        raw_components = extras["EasyCells3D"].get("components")

    if isinstance(raw_components, str):
        try:
            raw_components = json.loads(raw_components)
        except json.JSONDecodeError:
            warnings.warn(f"SceneLoader: components invalido em '{node.get('name', '<sem nome>')}'")
            return []

    if not raw_components:
        return []
    if not isinstance(raw_components, list):
        warnings.warn(f"SceneLoader: components deve ser lista em '{node.get('name', '<sem nome>')}'")
        return []
    return [entry for entry in raw_components if isinstance(entry, dict)]


def _unique_name(name: str, existing: dict[str, Item]) -> str:
    if name not in existing:
        return name
    index = 2
    while f"{name}.{index}" in existing:
        index += 1
    return f"{name}.{index}"


def _primitive_count(document: dict[str, Any], mesh_index: int) -> int:
    meshes = document.get("meshes", [])
    if mesh_index < 0 or mesh_index >= len(meshes):
        return 1
    primitives = meshes[mesh_index].get("primitives") or []
    return max(1, len(primitives))
