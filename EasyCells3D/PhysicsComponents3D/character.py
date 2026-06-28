"""``CharacterController3D`` — helper de personagem, agnóstico de câmera.

Construído **só** sobre a API pública de :class:`PhysicsBody3D` e
:class:`PhysicsWorld` (serve de exemplo de referência). Espera um
``PhysicsBody3D`` no mesmo item, idealmente uma cápsula ``DYNAMIC`` com
``lock_rotation=True`` e ``allow_sleep=False``.

Quem controla decide a câmera/input: :meth:`move` recebe uma direção em
world-space e :meth:`jump` só pula se :attr:`is_grounded`.
"""
from __future__ import annotations

import math

from ..Components import Component
from ..Geometry import Vec3
from .body import PhysicsBody3D
from .shapes import CapsuleShape


class CharacterController3D(Component):
    def __init__(
            self,
            move_speed: float = 5.0,
            jump_height: float = 1.2,
            ground_check_distance: float = 0.15,
            max_slope: float = 50.0,
    ):
        self.move_speed = move_speed
        self.jump_height = jump_height
        self.ground_check_distance = ground_check_distance
        self.max_slope = max_slope  # graus

        self.body: PhysicsBody3D | None = None
        self._half_height = 1.0
        self._radius = 0.5
        self._last_ground_normal = Vec3(0.0, 1.0, 0.0)

    def init(self):
        self.body = self.GetComponent(PhysicsBody3D)
        assert self.body is not None, (
            "CharacterController3D requer um PhysicsBody3D (cápsula DYNAMIC) "
            "no mesmo item."
        )
        shape = self.body.shape
        if isinstance(shape, CapsuleShape):
            self._radius = shape.radius
            self._half_height = shape.height / 2.0 + shape.radius

    # -- API ------------------------------------------------------------------

    def move(self, direction: Vec3):
        """Move horizontalmente (world-space), preservando a velocidade vertical."""
        if self.body is None:
            return
        flat = Vec3(direction.x, 0.0, direction.z)
        mag = flat.magnitude()
        vel = self.body.velocity
        if mag > 1e-6:
            flat = flat / mag * self.move_speed
            self.body.velocity = Vec3(flat.x, vel.y, flat.z)
        else:
            self.body.velocity = Vec3(0.0, vel.y, 0.0)

    def jump(self):
        """Pula, se estiver no chão. Altura calculada a partir de ``jump_height``."""
        if self.body is None or not self.is_grounded:
            return
        g = abs(self.game.physics_world.gravity.y) or 9.81
        v = math.sqrt(2.0 * g * self.jump_height)
        vel = self.body.velocity
        self.body.velocity = Vec3(vel.x, v, vel.z)

    @property
    def is_grounded(self) -> bool:
        """Raycast curto pra baixo respeitando ``max_slope``."""
        world = self.game.physics_world
        if world is None:
            return False
        center = self.global_transform.position
        # Começa logo abaixo da base da cápsula pra não acertar a si mesmo.
        origin = Vec3(center.x, center.y - (self._half_height - self._radius * 0.5), center.z)
        hit = world.raycast(origin, Vec3(0.0, -1.0, 0.0),
                            self._radius * 0.5 + self.ground_check_distance)
        if hit is None or hit.body is self.body:
            return False
        self._last_ground_normal = hit.normal
        slope = math.degrees(math.acos(max(-1.0, min(1.0, hit.normal.y))))
        return slope <= self.max_slope

    @property
    def ground_normal(self) -> Vec3:
        return self._last_ground_normal
