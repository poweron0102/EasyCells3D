"""Debug draw das shapes de colisão (M7).

``PhysicsDebugRenderer`` é um ``Renderable3D`` opt-in: adicione-o a qualquer
item (ex.: o da câmera) e ligue ``game.physics_world.debug_draw = True``.
Quando desligado o custo é zero (retorna logo no início do ``render``).

Os wireframes são desenhados com primitivas da Raylib na pose global de cada
corpo, usando o stack de matrizes do rlgl pra aplicar a rotação.
"""
from __future__ import annotations

import pyray as rl

from ..Components.Camera3D import Renderable3D
from ..Components.StaticModel import quaternion_to_axis_angle
from .body import PhysicsBody3D
from .shapes import (
    BoxShape, CapsuleShape, CompoundShape, ConvexHullShape,
    CylinderShape, SphereShape, TriangleMeshShape,
)
from .world import BulletPhysicsWorld

_DEFAULT_COLOR = rl.Color(0, 255, 0, 255)
_TRIGGER_COLOR = rl.Color(255, 220, 0, 255)


class PhysicsDebugRenderer(Renderable3D):
    def __init__(self, color: rl.Color = _DEFAULT_COLOR,
                 trigger_color: rl.Color = _TRIGGER_COLOR):
        super().__init__()
        self.color = color
        self.trigger_color = trigger_color

    def render(self):
        world = self.game.physics_world
        if not isinstance(world, BulletPhysicsWorld) or not world.debug_draw:
            return
        for body in world.bodies:
            if body.uid is None or body.shape is None:
                continue
            self._draw_body(body)

    def _draw_body(self, body: PhysicsBody3D):
        gt = body.global_transform
        pos = gt.position
        scale = gt.scale
        color = self.trigger_color if body.is_trigger else self.color
        axis, angle = quaternion_to_axis_angle(gt.rotation)

        rl.rl_push_matrix()
        rl.rl_translatef(pos.x, pos.y, pos.z)
        rl.rl_rotatef(angle, axis.x, axis.y, axis.z)
        self._draw_shape(body.shape, scale, color)
        rl.rl_pop_matrix()

    def _draw_shape(self, shape, scale, color):
        origin = rl.Vector3(0.0, 0.0, 0.0)
        if isinstance(shape, BoxShape):
            rl.draw_cube_wires(
                origin,
                shape.half_extents.x * 2 * abs(scale.x),
                shape.half_extents.y * 2 * abs(scale.y),
                shape.half_extents.z * 2 * abs(scale.z),
                color,
            )
        elif isinstance(shape, SphereShape):
            s = max(abs(scale.x), abs(scale.y), abs(scale.z))
            rl.draw_sphere_wires(origin, shape.radius * s, 8, 8, color)
        elif isinstance(shape, CapsuleShape):
            r = shape.radius * max(abs(scale.x), abs(scale.z))
            h = shape.height * abs(scale.y)
            top = rl.Vector3(0.0, h / 2.0, 0.0)
            bottom = rl.Vector3(0.0, -h / 2.0, 0.0)
            rl.draw_capsule_wires(top, bottom, r, 8, 4, color)
        elif isinstance(shape, CylinderShape):
            r = shape.radius * max(abs(scale.x), abs(scale.z))
            h = shape.height * abs(scale.y)
            bottom = rl.Vector3(0.0, -h / 2.0, 0.0)
            top = rl.Vector3(0.0, h / 2.0, 0.0)
            rl.draw_cylinder_wires_ex(bottom, top, r, r, 12, color)
        elif isinstance(shape, ConvexHullShape):
            self._draw_points(shape.vertices, scale, color)
        elif isinstance(shape, TriangleMeshShape):
            self._draw_triangles(shape.vertices, shape.indices, scale, color)
        elif isinstance(shape, CompoundShape):
            for child in shape.children:
                c_axis, c_angle = quaternion_to_axis_angle_xyzw(child.orientation)
                rl.rl_push_matrix()
                rl.rl_translatef(child.position.x * scale.x,
                                 child.position.y * scale.y,
                                 child.position.z * scale.z)
                rl.rl_rotatef(c_angle, c_axis.x, c_axis.y, c_axis.z)
                self._draw_shape(child.shape, scale, color)
                rl.rl_pop_matrix()

    @staticmethod
    def _draw_points(vertices, scale, color):
        for v in vertices[:: max(1, len(vertices) // 64)]:
            p = rl.Vector3(v[0] * scale.x, v[1] * scale.y, v[2] * scale.z)
            rl.draw_point_3d(p, color)

    @staticmethod
    def _draw_triangles(vertices, indices, scale, color):
        # Desenha as arestas dos triângulos. Custo proporcional ao nº de tris
        # (opt-in; use só pra depurar colliders pequenos/médios).
        def vp(i):
            v = vertices[i]
            return rl.Vector3(v[0] * scale.x, v[1] * scale.y, v[2] * scale.z)

        for t in range(0, len(indices) - 2, 3):
            a, b, c = indices[t], indices[t + 1], indices[t + 2]
            va, vb, vc = vp(a), vp(b), vp(c)
            rl.draw_line_3d(va, vb, color)
            rl.draw_line_3d(vb, vc, color)
            rl.draw_line_3d(vc, va, color)


def quaternion_to_axis_angle_xyzw(q_xyzw) -> tuple[rl.Vector3, float]:
    """Igual a quaternion_to_axis_angle, mas pra quat no formato Bullet [x,y,z,w]."""
    import math
    x, y, z, w = q_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 0.001:
        return rl.Vector3(0, 1, 0), 0.0
    return rl.Vector3(x / s, y / s, z / s), math.degrees(angle)
