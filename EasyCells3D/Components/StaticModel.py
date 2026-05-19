import math
from pathlib import Path
from typing import ClassVar

import pyray as rl

from EasyCells3D.Components.Component import Transform
from EasyCells3D.Components.Camera3D import Renderable3D


def resolve_asset_path(path: str, base_path: str = "Assets") -> str:
    model_path = Path(path)
    if model_path.is_absolute() or model_path.exists():
        return str(model_path)
    return str(Path(base_path) / model_path)


def quaternion_to_axis_angle(q) -> tuple[rl.Vector3, float]:
    q = q.normalize()
    angle = 2 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    if s < 0.001:
        return rl.Vector3(0, 1, 0), 0.0
    return rl.Vector3(q.x / s, q.y / s, q.z / s), math.degrees(angle)


class StaticModel(Renderable3D):
    """
    Carrega um modelo 3D
    """

    _shared_models: ClassVar[dict[str, dict]] = {}

    def __init__(
            self,
            model_path: str,
            color: rl.Color = rl.WHITE,
            base_path: str = "Assets",
            mesh_index: int | list[int] | tuple[int, ...] | None = None,
            material_index: int | None = None,
            shared: bool = False,
            fallback_only: bool = False,
            baked_transform: Transform | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.base_path = base_path
        self.color = color
        self.mesh_indices = self._normalize_mesh_indices(mesh_index)
        self.material_index = material_index
        self.shared = shared
        self.fallback_only = fallback_only
        self.baked_transform = baked_transform
        self.resolved_model_path: str | None = None
        self.model: rl.Model = None
        self._render_warning_printed = False

    def init(self):
        super().init()
        self.resolved_model_path = resolve_asset_path(self.model_path, self.base_path)
        self.model = self._load_model(self.resolved_model_path)
        if self.fallback_only:
            self._register_as_shared_fallback_owner()
        if hasattr(rl, "is_model_valid") and not rl.is_model_valid(self.model):
            print(f"StaticModel: modelo invalido ou vazio: {self.model_path}")

    def on_destroy(self):
        super().on_destroy()
        if self.shared and self.resolved_model_path:
            self._release_shared_model(self.resolved_model_path)
        elif self.model:
            rl.unload_model(self.model)
        self.model = None

    def render(self):
        if not self.model: return
        pos = self.global_transform.position.to_raylib()
        axis, angle = quaternion_to_axis_angle(self.global_transform.rotation)

        scale = rl.Vector3(self.global_transform.scale.x, self.global_transform.scale.y, self.global_transform.scale.z)

        if self.fallback_only:
            if self._should_draw_shared_full_model_fallback():
                rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)
            return

        if self.mesh_indices is None:
            rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)
            return

        if self._should_draw_shared_full_model_fallback():
            rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)
            return

        if self._shared_submesh_render_failed():
            return

        if self._draw_meshes_ex(pos, axis, angle, scale):
            return

        if not self.shared:
            rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)
            return

        self._mark_shared_submesh_render_failed()
        if self._should_draw_shared_full_model_fallback():
            rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)

    def _load_model(self, resolved_path: str) -> rl.Model:
        if not self.shared:
            return rl.load_model(resolved_path)

        entry = StaticModel._shared_models.get(resolved_path)
        if entry:
            entry["refs"] += 1
            return entry["model"]

        model = rl.load_model(resolved_path)
        StaticModel._shared_models[resolved_path] = {
            "model": model,
            "refs": 1,
            "submesh_failed": False,
            "fallback_owner": None,
        }
        return model

    @staticmethod
    def _release_shared_model(resolved_path: str) -> None:
        entry = StaticModel._shared_models.get(resolved_path)
        if not entry:
            return

        entry["refs"] -= 1
        if entry["refs"] <= 0:
            rl.unload_model(entry["model"])
            StaticModel._shared_models.pop(resolved_path, None)

    @staticmethod
    def _normalize_mesh_indices(mesh_index: int | list[int] | tuple[int, ...] | None) -> list[int] | None:
        if mesh_index is None:
            return None
        if isinstance(mesh_index, int):
            return [mesh_index]
        return [int(index) for index in mesh_index]

    def _draw_meshes_ex(self, pos: rl.Vector3, axis: rl.Vector3, angle: float, scale: rl.Vector3) -> bool:
        try:
            mesh_count = self._model_count("mesh")
            material_count = self._model_count("material")
            transform = rl.matrix_multiply(
                rl.matrix_scale(scale.x, scale.y, scale.z),
                rl.matrix_rotate(axis, math.radians(angle)),
            )
            transform = rl.matrix_multiply(transform, rl.matrix_translate(pos.x, pos.y, pos.z))
            if self.baked_transform is not None:
                transform = rl.matrix_multiply(self._inverse_transform_matrix(self.baked_transform), transform)

            for mesh_index in self.mesh_indices:
                if mesh_index < 0 or mesh_index >= mesh_count:
                    self._print_render_warning(
                        f"StaticModel: mesh_index fora do intervalo: {mesh_index} "
                        f"em {self.model_path} (mesh_count={mesh_count})"
                    )
                    return False

                material_index = self._material_index_for_mesh(mesh_index, material_count)
                mesh = self.model.meshes[mesh_index]
                material = self.model.materials[material_index]
                rl.draw_mesh(mesh, material, transform)

            return True
        except Exception as exc:
            self._print_render_warning(
                f"StaticModel: nao foi possivel desenhar meshes {self.mesh_indices} "
                f"de {self.model_path}: {exc}"
            )
            return False

    def _model_count(self, name: str) -> int:
        snake_name = f"{name}_count"
        camel_name = f"{name}Count"
        return int(getattr(self.model, snake_name, getattr(self.model, camel_name, 0)) or 0)

    def _inverse_transform_matrix(self, transform: Transform):
        inv_scale = rl.matrix_scale(
            1.0 / transform.scale.x if transform.scale.x else 0.0,
            1.0 / transform.scale.y if transform.scale.y else 0.0,
            1.0 / transform.scale.z if transform.scale.z else 0.0,
        )
        axis, angle = quaternion_to_axis_angle(transform.rotation.inverse())
        inv_rotation = rl.matrix_rotate(axis, math.radians(angle))
        inv_translation = rl.matrix_translate(
            -transform.position.x,
            -transform.position.y,
            -transform.position.z,
        )
        matrix = rl.matrix_multiply(inv_translation, inv_rotation)
        return rl.matrix_multiply(matrix, inv_scale)

    def _material_index_for_mesh(self, mesh_index: int, material_count: int) -> int:
        material_index = self.material_index
        if material_index is None:
            mesh_material = getattr(self.model, "mesh_material", None)
            if mesh_material is None:
                mesh_material = getattr(self.model, "meshMaterial", None)
            if mesh_material is not None:
                material_index = int(mesh_material[mesh_index])
        if material_index is None:
            material_index = 0
        if material_index < 0 or material_index >= material_count:
            return 0
        return material_index

    def _shared_entry(self) -> dict | None:
        if not self.shared or not self.resolved_model_path:
            return None
        return StaticModel._shared_models.get(self.resolved_model_path)

    def _shared_submesh_render_failed(self) -> bool:
        entry = self._shared_entry()
        return bool(entry and entry.get("submesh_failed"))

    def _mark_shared_submesh_render_failed(self) -> None:
        entry = self._shared_entry()
        if not entry:
            return
        entry["submesh_failed"] = True
        if entry.get("fallback_owner") is None:
            entry["fallback_owner"] = id(self)

    def _register_as_shared_fallback_owner(self) -> None:
        entry = self._shared_entry()
        if not entry:
            return
        entry["fallback_owner"] = id(self)

    def _should_draw_shared_full_model_fallback(self) -> bool:
        entry = self._shared_entry()
        if not entry or not entry.get("submesh_failed"):
            return False
        return entry.get("fallback_owner") == id(self)

    def _print_render_warning(self, message: str) -> None:
        if self._render_warning_printed:
            return
        print(message)
        self._render_warning_printed = True
