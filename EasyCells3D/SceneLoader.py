import json
import math
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyray as rl

from EasyCells3D.ComponentRegistry import ComponentCreationContext, ComponentRegistry
from EasyCells3D.Components import AnimatedModel, Animator3D, Camera3D, Item, Light3D, StaticModel, Transform
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.Serialization import get_serialized_fields


@dataclass
class _PendingSerializedFields:
    item: Item
    component: Any
    component_def: dict
    context: ComponentCreationContext


class SceneLoader:
    def __init__(self, game, component_registry: type[ComponentRegistry] = ComponentRegistry):
        self.game = game
        self.component_registry = component_registry

    def load(self, path: str | Path) -> list[Item]:
        scene_path = Path(path)
        document = _load_gltf_document(scene_path)
        self.component_registry.ensure_discovered(_find_project_root(scene_path))
        nodes = document.get("nodes", [])
        scene_index = document.get("scene", 0)
        scenes = document.get("scenes", [])
        root_node_indices = scenes[scene_index].get("nodes", []) if scenes else list(range(len(nodes)))

        objects_by_name: dict[str, Item] = {}
        objects_by_node_index: dict[int, Item] = {}
        objects_by_easycells_id: dict[str, Item] = {}

        mesh_draw_indices = _mesh_draw_indices(document)

        def create_node(node_index: int, parent: Item | None = None, suppress_static_model: bool = False) -> Item:
            node = nodes[node_index]
            item = parent.CreateChild() if parent else self.game.CreateItem()
            item.name = node.get("name") or f"Node_{node_index}"
            item.transform = _node_transform(node)
            animated_model_def = _extract_animated_model_def(node)

            unique_name = _unique_name(item.name, objects_by_name)
            objects_by_name[unique_name] = item
            objects_by_node_index[node_index] = item
            easycells_id = _extract_easycells_id(node)
            if easycells_id:
                item.easycells_id = easycells_id
                objects_by_easycells_id[easycells_id] = item

            if animated_model_def:
                item.AddComponent(_animated_model_component_from_gltf(animated_model_def, scene_path))
                if not _node_has_component_type(node, {"Animator3D", "EasyCells3D.Components.Animator3D.Animator3D"}):
                    item.AddComponent(Animator3D(
                        current_animation=animated_model_def.get("current_animation"),
                        autoplay=bool(animated_model_def.get("autoplay", True)),
                    ))

            if "mesh" in node and not suppress_static_model and not animated_model_def:
                mesh_index = int(node["mesh"])
                primitive_count = _primitive_count(document, mesh_index)
                mesh_indices = list(range(mesh_draw_indices[mesh_index], mesh_draw_indices[mesh_index] + primitive_count))
                item.AddComponent(StaticModel(
                    str(scene_path),
                    base_path=".",
                    mesh_index=mesh_indices,
                    shared=True,
                    baked_transform=item.global_transform_get().clone(),
                ))

            _add_native_gltf_components(item, node, document)

            for child_index in node.get("children", []):
                create_node(child_index, item, suppress_static_model or bool(animated_model_def))

            return item

        root_items = [create_node(node_index) for node_index in root_node_indices]
        if root_items and mesh_draw_indices and not _scene_has_configured_components(nodes) and not _scene_has_animated_models(nodes):
            root_items[0].AddComponent(StaticModel(
                str(scene_path),
                base_path=".",
                shared=True,
                fallback_only=True,
            ))

        pending_fields: list[_PendingSerializedFields] = []
        for node_index, item in list(objects_by_node_index.items()):
            context = ComponentCreationContext(
                game=self.game,
                item=item,
                node=nodes[node_index],
                scene_path=str(scene_path),
                objects_by_name=objects_by_name,
                objects_by_node_index=objects_by_node_index,
                objects_by_easycells_id=objects_by_easycells_id,
            )
            pending_fields.extend(self._add_configured_components(item, nodes[node_index], context))

        for pending in pending_fields:
            self._apply_serialized_fields(pending)

        return root_items

    def _add_configured_components(
        self,
        item: Item,
        node: dict,
        context: ComponentCreationContext,
    ) -> list[_PendingSerializedFields]:
        pending_fields: list[_PendingSerializedFields] = []
        for component_def in _extract_component_defs(node):
            type_name = component_def.get("type")
            args = component_def.get("args", {})
            if not type_name:
                warnings.warn(f"SceneLoader: componente sem 'type' no objeto '{item.name}'")
                continue

            try:
                component = self.component_registry.create(type_name, args, context)
                item.AddComponent(component)
                pending_fields.append(_PendingSerializedFields(item, component, component_def, context))
            except Exception as exc:
                warnings.warn(
                    f"SceneLoader: erro ao criar componente '{type_name}' no objeto "
                    f"'{item.name}': {exc}"
                )
        return pending_fields

    def _apply_serialized_fields(self, pending: _PendingSerializedFields) -> None:
        fields = pending.component_def.get("fields") or {}
        if not isinstance(fields, dict):
            warnings.warn(
                f"SceneLoader: fields deve ser objeto em '{pending.item.name}' "
                f"({pending.component.__class__.__name__})"
            )
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
                    f"SceneLoader: erro ao aplicar field '{name}' em "
                    f"{pending.component.__class__.__name__} no objeto '{pending.item.name}': {exc}"
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


def _extract_animated_model_def(node: dict) -> dict | None:
    extras = _node_extras(node)
    if not extras:
        return None
    value = extras.get("easycells_animated_model") or extras.get("animated_model")
    if value is None and isinstance(extras.get("EasyCells3D"), dict):
        value = extras["EasyCells3D"].get("animated_model")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"path": value}
    if isinstance(value, dict) and value.get("path"):
        return value
    return None


def _animated_model_component_from_gltf(animated_model_def: dict, scene_path: Path) -> AnimatedModel:
    return AnimatedModel(
        _resolve_animated_model_path(str(animated_model_def["path"]), scene_path),
        clip_names=_string_list(animated_model_def.get("clip_names")),
        clip_fps=float(animated_model_def.get("clip_fps", 24.0)),
        base_path=".",
    )


def _resolve_animated_model_path(path: str, scene_path: Path) -> str:
    model_path = Path(path)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    scene_relative = scene_path.parent / model_path
    if scene_relative.exists():
        return str(scene_relative)
    return path


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(entry) for entry in value if entry is not None]


def _add_native_gltf_components(item: Item, node: dict, document: dict[str, Any]) -> None:
    if "camera" in node and not _node_has_component_type(node, {"Camera3D", "EasyCells3D.Components.Camera3D.Camera3D"}):
        component = _camera_component_from_gltf(node["camera"], document)
        if component:
            item.AddComponent(component)

    if not _node_has_component_type(node, {"Light3D", "EasyCells3D.Components.Light3D.Light3D"}):
        component = _light_component_from_gltf(node, document)
        if component:
            item.AddComponent(component)


def _camera_component_from_gltf(camera_index: Any, document: dict[str, Any]) -> Camera3D | None:
    cameras = document.get("cameras") or []
    try:
        camera = cameras[int(camera_index)]
    except (IndexError, TypeError, ValueError):
        warnings.warn(f"SceneLoader: camera glTF invalida: {camera_index}")
        return None

    camera_type = camera.get("type", "perspective")
    if camera_type == "orthographic":
        orthographic = camera.get("orthographic") or {}
        ymag = float(orthographic.get("ymag", orthographic.get("xmag", 5.0)))
        return Camera3D(
            vfov=ymag * 2.0,
            projection=rl.CameraProjection.CAMERA_ORTHOGRAPHIC,
        )

    perspective = camera.get("perspective") or {}
    yfov = float(perspective.get("yfov", math.radians(60.0)))
    return Camera3D(vfov=math.degrees(yfov))


def _light_component_from_gltf(node: dict, document: dict[str, Any]) -> Light3D | None:
    extension = (node.get("extensions") or {}).get("KHR_lights_punctual")
    if not isinstance(extension, dict) or "light" not in extension:
        return None

    lights = ((document.get("extensions") or {}).get("KHR_lights_punctual") or {}).get("lights") or []
    try:
        light = lights[int(extension["light"])]
    except (IndexError, TypeError, ValueError):
        warnings.warn(f"SceneLoader: luz glTF invalida: {extension.get('light')}")
        return None

    spot = light.get("spot") or {}
    return Light3D(
        light_type=light.get("type", "point"),
        color=light.get("color", [1.0, 1.0, 1.0]),
        intensity=light.get("intensity", 1.0),
        range=light.get("range"),
        inner_cone_angle=spot.get("innerConeAngle", 0.0),
        outer_cone_angle=spot.get("outerConeAngle", math.pi / 4.0),
        name=light.get("name"),
    )


def _node_has_component_type(node: dict, type_names: set[str]) -> bool:
    for component_def in _extract_component_defs(node):
        type_name = component_def.get("type")
        if type_name in type_names:
            return True
    return False


def _scene_has_configured_components(nodes: list[dict]) -> bool:
    return any(_extract_component_defs(node) for node in nodes)


def _scene_has_animated_models(nodes: list[dict]) -> bool:
    return any(_extract_animated_model_def(node) for node in nodes)


def _extract_easycells_id(node: dict) -> str | None:
    extras = _node_extras(node)
    if not isinstance(extras, dict):
        return None
    value = extras.get("easycells_id")
    return str(value) if value else None


def _node_extras(node: dict) -> dict:
    extras = node.get("extras") or {}
    if isinstance(extras, str):
        try:
            return json.loads(extras)
        except json.JSONDecodeError:
            return {}
    if not isinstance(extras, dict):
        return {}
    return extras


def _resolve_serialized_value(value: Any, field, context: ComponentCreationContext) -> Any:
    if not isinstance(value, dict):
        return value

    reference = value.get("$id") or value.get("$ref")
    if reference:
        item = None
        if context.objects_by_easycells_id:
            item = context.objects_by_easycells_id.get(str(reference))
        if item is None:
            item = context.objects_by_name.get(str(reference))
        if item is None:
            raise KeyError(f"referencia nao encontrada: {reference}")

        component_type = value.get("$component")
        if component_type:
            return _find_component_on_item(item, str(component_type))
        if field is not None and field.ref == "component":
            component_type = value.get("type") or value.get("component")
            if component_type:
                return _find_component_on_item(item, str(component_type))
        return item

    return value


def _find_component_on_item(item: Item, component_type: str):
    for component in set(item.components.values()):
        cls = component.__class__
        if component_type in {cls.__name__, f"{cls.__module__}.{cls.__name__}"}:
            return component
    raise KeyError(f"componente '{component_type}' nao encontrado em '{item.name}'")


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


def _primitive_count(document: dict[str, Any], mesh_index: int) -> int:
    meshes = document.get("meshes", [])
    if mesh_index < 0 or mesh_index >= len(meshes):
        return 1
    primitives = meshes[mesh_index].get("primitives") or []
    return max(1, len(primitives))


def _mesh_draw_indices(document: dict[str, Any]) -> dict[int, int]:
    indices: dict[int, int] = {}
    next_draw_index = 0
    for mesh_index, _ in enumerate(document.get("meshes", [])):
        indices[mesh_index] = next_draw_index
        next_draw_index += _primitive_count(document, mesh_index)
    return indices


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
