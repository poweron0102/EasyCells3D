from __future__ import annotations
from typing import List

from .. import Game
from ..Components import Component
from ..Geometry import Vec2
from .Collider import Collider


class Rigidbody(Component):
    """
    A component that adds physics simulation to an Item.
    It controls an object's position and rotation using forces and torque.
    It requires a Collider component on the same Item to interact with other objects.

    IMPORTANT USAGE NOTE:
    The physics simulation runs in the `physics_step` static method.
    For stable and predictable physics, you should call this method from your
    main game loop at a fixed interval (Fixed Update).
    """

    # A static list containing all active rigidbodies in the scene.
    RigidBodies: List[Rigidbody] = []
    # Global gravity force, can be adjusted for your game's scale.
    Gravity = Vec2(0, 980)  # Using a value suitable for pixel-based coordinates

    def __init__(self, mass: float = 1.0, use_gravity: bool = True, is_kinematic: bool = False,
                 drag: float = 0.05, angular_drag: float = 0.05, gravity_scale: float = 1.0,
                 restitution: float = 0.5):
        """
        Initializes the Rigidbody component.
        :param mass: The mass of the object. Must be greater than 0.
        :param use_gravity: If true, the object is affected by gravity.
        :param is_kinematic: If true, the object is not affected by physics forces,
                             but can be moved by its transform and can affect other non-kinematic objects.
        :param drag: Linear drag to slow down the object's movement over time.
        :param angular_drag: Angular drag to slow down the object's rotation over time.
        :param gravity_scale: A multiplier for the effect of gravity on this specific object.
        :param restitution: The "bounciness" of the object. 0 means no bounce, 1 means a perfectly elastic bounce.
        """
        self.velocity = Vec2(0, 0)
        self.angular_velocity = 0.0

        if mass <= 0:
            print("Warning: Rigidbody mass cannot be zero or less. Clamping to 0.001.")
            mass = 0.001
        self.mass = mass
        self.inv_mass = 0.0 if is_kinematic else 1.0 / self.mass

        self.use_gravity = use_gravity
        self.is_kinematic = is_kinematic
        self.drag = drag
        self.angular_drag = angular_drag
        self.gravity_scale = gravity_scale
        self.restitution = max(0.0, min(restitution, 1.0))  # Clamp between 0 and 1

        self._force_accumulator = Vec2(0, 0)
        self._torque_accumulator = 0.0

        self.collider: Collider | None = None

    def init(self):
        self.collider = self.GetComponent(Collider)
        if not self.collider:
            print(
                f"Warning: Rigidbody on item '{self.item}' has no Collider component. It will not participate in collisions.")

        if self not in Rigidbody.RigidBodies:
            Rigidbody.RigidBodies.append(self)

    def on_destroy(self):
        if self in Rigidbody.RigidBodies:
            Rigidbody.RigidBodies.remove(self)
        self.on_destroy = lambda: None

    def add_force(self, force: Vec2):
        self._force_accumulator += force

    def add_impulse(self, impulse: Vec2):
        if self.is_kinematic:
            return
        self.velocity += impulse * self.inv_mass

    def loop(self):
        pass

    def _integrate(self, delta_time: float):
        if self.is_kinematic:
            return

        if self.use_gravity:
            self.add_force(Rigidbody.Gravity * self.mass * self.gravity_scale)

        acceleration = self._force_accumulator * self.inv_mass
        self.velocity += acceleration * delta_time
        self.velocity *= max(0, 1.0 - self.drag * delta_time)
        self.transform.position += self.velocity * delta_time

        self.angular_velocity += self._torque_accumulator * delta_time
        self.angular_velocity *= max(0, 1.0 - self.angular_drag * delta_time)
        self.transform.angle += self.angular_velocity * delta_time

        self._force_accumulator = Vec2(0, 0)
        self._torque_accumulator = 0.0

    @staticmethod
    def physics_step(delta_time: float):
        # 1. Integrate forces and update positions
        for rb in Rigidbody.RigidBodies:
            rb._integrate(delta_time)

        # 2. Collision detection and resolution
        for i in range(len(Rigidbody.RigidBodies)):
            for j in range(i + 1, len(Rigidbody.RigidBodies)):
                rb1 = Rigidbody.RigidBodies[i]
                rb2 = Rigidbody.RigidBodies[j]

                if (rb1.is_kinematic and rb2.is_kinematic) or not rb1.collider or not rb2.collider:
                    continue

                # The collision check now returns the MTV
                colliding, mtv_np = rb1.collider.check_collision_global(rb2.collider)

                if colliding:
                    mtv = Vec2(mtv_np[0], mtv_np[1])
                    Rigidbody._resolve_collision(rb1, rb2, mtv)

    @staticmethod
    def start_physics():
        """
        This method should be called at the start of your level to initialize the physics system.
        It will ensure that all rigidbodies are updated and collisions are resolved.
        """
        def physics_loop():
            last_time: float = Game.instance().time
            yield
            while True:
                current_time: float = Game.instance().time
                delta = (current_time - last_time) / 1000.0
                Rigidbody.physics_step(delta if delta < 0.02 else 0.02)  # Cap delta time to avoid large jumps
                last_time = current_time
                yield  1 / 60

        Game.instance().scheduler.add_generator(physics_loop(), 0)


    @staticmethod
    def _resolve_collision(rb1: Rigidbody, rb2: Rigidbody, mtv: Vec2):
        """
        Handles the physics response to a collision using the Minimum Translation Vector.
        """
        # print(f"Resolving collision between {rb1.item} and {rb2.item} with MTV: {mtv}")
        # print(f"start position rb1: {rb1.transform.position}, rb2: {rb2.transform.position}")
        # The collision normal is the normalized MTV.
        collision_normal = mtv.normalize()
        penetration_depth = mtv.magnitude()

        relative_velocity = rb2.velocity - rb1.velocity
        vel_along_normal = relative_velocity.dot(collision_normal)

        # Do not resolve if velocities are already separating
        if vel_along_normal > 0:
            return

        restitution = min(rb1.restitution, rb2.restitution)
        total_inv_mass = rb1.inv_mass + rb2.inv_mass

        if total_inv_mass <= 0:
            return

        # --- 1. Impulse Resolution (Velocity Change) ---
        j = -(1 + restitution) * vel_along_normal
        j /= total_inv_mass
        impulse = collision_normal * j

        if not rb1.is_kinematic:
            rb1.velocity -= impulse * rb1.inv_mass
        if not rb2.is_kinematic:
            rb2.velocity += impulse * rb2.inv_mass

        # --- 2. Positional Correction (Penetration Resolution) ---
        # This moves the objects apart to fix the overlap.
        percent = 0.4  # How much of the penetration to correct (usually 20-80%) to avoid jitter
        slop = 0.01  # A small buffer to prevent objects from getting stuck
        correction_amount = max(penetration_depth - slop, 0.0) / total_inv_mass * percent
        correction = collision_normal * correction_amount

        if not rb1.is_kinematic:
            rb1.transform.position -= correction * rb1.inv_mass
        if not rb2.is_kinematic:
            rb2.transform.position += correction * rb2.inv_mass

        #print(f"end   position rb1: {rb1.transform.position}, rb2: {rb2.transform.position}")
