from __future__ import annotations

from ..Components import Component
from ..Geometry import Vec2
from .Collider import Collider


class Rigidbody(Component):
    """2D rigidbody simulated by ``SATPhysicsWorld``."""

    def __init__(
            self,
            mass: float = 1.0,
            use_gravity: bool = True,
            is_kinematic: bool = False,
            drag: float = 0.05,
            angular_drag: float = 0.05,
            gravity_scale: float = 1.0,
            restitution: float = 0.5,
    ):
        self.velocity = Vec2(0, 0)
        self.angular_velocity = 0.0

        if mass <= 0:
            print("Warning: Rigidbody mass cannot be zero or less. Clamping to 0.001.")
            mass = 0.001
        self.mass = mass
        self._is_kinematic = bool(is_kinematic)
        self.inv_mass = 0.0 if self._is_kinematic else 1.0 / self.mass

        self.use_gravity = use_gravity
        self.drag = drag
        self.angular_drag = angular_drag
        self.gravity_scale = gravity_scale
        self.restitution = max(0.0, min(restitution, 1.0))

        self._force_accumulator = Vec2(0, 0)
        self._torque_accumulator = 0.0
        self.collider: Collider | None = None
        self._world = None

    @property
    def is_kinematic(self) -> bool:
        return self._is_kinematic

    @is_kinematic.setter
    def is_kinematic(self, value: bool) -> None:
        self._is_kinematic = bool(value)
        self.inv_mass = 0.0 if self._is_kinematic else 1.0 / self.mass

    def init(self):
        from .SATPhysicsWorld import SATPhysicsWorld

        world = self.game.physics_world
        if not isinstance(world, SATPhysicsWorld):
            raise RuntimeError(
                "Rigidbody 2D requer game.physics_world = SATPhysicsWorld() "
                "no init() do level."
            )

        self.collider = self.GetComponent(Collider)
        if not self.collider:
            print(
                f"Warning: Rigidbody on item '{self.item}' has no Collider component. "
                "It will not participate in collisions."
            )

        self._world = world
        world.add_body(self)

    def on_destroy(self):
        if self._world is not None:
            self._world.remove_body(self)
            self._world = None

    def add_force(self, force: Vec2):
        self._force_accumulator += force

    def add_impulse(self, impulse: Vec2):
        if self.is_kinematic:
            return
        self.velocity += impulse * self.inv_mass

    def _integrate(self, delta_time: float, gravity: Vec2):
        if self.is_kinematic or not self.enable:
            self._force_accumulator = Vec2(0, 0)
            self._torque_accumulator = 0.0
            return

        if self.use_gravity:
            self.add_force(gravity * self.mass * self.gravity_scale)

        acceleration = self._force_accumulator * self.inv_mass
        self.velocity += acceleration * delta_time
        self.velocity *= max(0, 1.0 - self.drag * delta_time)
        self.transform.positionVec2 += self.velocity * delta_time
        self.global_transform.positionVec2 += self.velocity * delta_time

        self.angular_velocity += self._torque_accumulator * delta_time
        self.angular_velocity *= max(0, 1.0 - self.angular_drag * delta_time)
        self.transform.angle += self.angular_velocity * delta_time
        self.global_transform.angle += self.angular_velocity * delta_time

        self._force_accumulator = Vec2(0, 0)
        self._torque_accumulator = 0.0
