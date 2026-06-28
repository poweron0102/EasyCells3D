from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..Geometry import Vec2, Vec3
from ..PhysicsComponents3D.world import PhysicsWorld, RaycastHit
from .Collider import Collider, Polygon, _polygon_sweep_numba, _ray_polygon_intersection_numba

if TYPE_CHECKING:
    from .Rigidbody import Rigidbody


class SATPhysicsWorld(PhysicsWorld):
    """2D physics world backed by the existing SAT polygon solver."""

    def __init__(
            self,
            gravity: Vec2 | Vec3 = Vec2(0, 980),
            fixed_timestep: float = 1.0 / 60.0,
            max_substeps: int = 5,
    ):
        self._gravity = self._to_vec2(gravity)
        self.fixed_timestep = fixed_timestep
        self.max_substeps = max_substeps
        self._accumulator = 0.0
        self._debug_draw = False
        self.rigidbodies: list[Rigidbody] = []
        self.colliders: list[Collider] = []

    @property
    def gravity(self) -> Vec2:
        return self._gravity

    def set_gravity(self, gravity: Vec3) -> None:
        self._gravity = Vec2(gravity.x, gravity.y)

    def add_body(self, body: Rigidbody) -> None:
        if body not in self.rigidbodies:
            self.rigidbodies.append(body)

    def remove_body(self, body: Rigidbody) -> None:
        if body in self.rigidbodies:
            self.rigidbodies.remove(body)

    def add_collider(self, collider: Collider) -> None:
        if collider not in self.colliders:
            self.colliders.append(collider)

    def remove_collider(self, collider: Collider) -> None:
        if collider in self.colliders:
            self.colliders.remove(collider)

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return

        self._accumulator += dt
        steps = 0
        while self._accumulator >= self.fixed_timestep and steps < self.max_substeps:
            self._step_fixed(self.fixed_timestep)
            self._accumulator -= self.fixed_timestep
            steps += 1

        if self._accumulator > self.fixed_timestep:
            self._accumulator = 0.0

    def raycast(self, origin: Vec3, direction: Vec3, max_dist: float,
                mask: int = -1) -> RaycastHit | None:
        hit = self.ray_cast(
            Vec2(origin.x, origin.y),
            Vec2(direction.x, direction.y),
            max_dist,
            mask,
        )
        if hit is None:
            return None
        collider, point, normal, distance = hit
        return RaycastHit(
            body=self.body_for_collider(collider) or collider,
            point=Vec3(point.x, point.y, origin.z),
            normal=Vec3(normal.x, normal.y, 0.0),
            distance=distance,
        )

    def ray_cast(
            self,
            origin: Vec2,
            direction: Vec2,
            max_distance: float,
            mask: int = -1,
    ) -> tuple[Collider, Vec2, Vec2, float] | None:
        direction = direction.normalize()
        if direction.magnitude() == 0 or max_distance <= 0:
            return None

        origin_array = np.array([origin.x, origin.y], dtype=np.float64)
        direction_array = np.array([direction.x, direction.y], dtype=np.float64)

        closest_collider = None
        closest_point = None
        closest_normal = None
        closest_distance = max_distance

        for collider in self.colliders:
            if not collider.enable:
                continue
            if mask != -1 and collider.mask & mask == 0:
                continue
            for polygon in collider.polygons:
                transformed = polygon.apply_transform(collider.global_transform)
                intersection, normal, distance = _ray_polygon_intersection_numba(
                    origin_array, direction_array, transformed.vertices, max_distance)
                if intersection is not None and distance < closest_distance:
                    closest_collider = collider
                    closest_point = Vec2(intersection[0], intersection[1])
                    closest_normal = Vec2(normal[0], normal[1])
                    closest_distance = distance

        if closest_collider is None:
            return None
        if np.dot(closest_normal.to_tuple, direction_array) > 0:
            closest_normal *= -1
        return closest_collider, closest_point, closest_normal, closest_distance

    def rect_cast(
            self,
            origin: Vec2,
            size: Vec2,
            angle: float,
            direction: Vec2,
            max_distance: float,
            mask: int = -1,
    ) -> tuple[Collider, Vec2, Vec2, float] | None:
        direction = direction.normalize()
        if direction.magnitude() == 0 or max_distance <= 0:
            return None

        w, h = size.x / 2, size.y / 2
        local_vertices = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=np.float64)
        rad = np.radians(angle)
        c, s = np.cos(rad), np.sin(rad)
        rotation_matrix_t = np.array([[c, s], [-s, c]])
        rect_vertices = (local_vertices @ rotation_matrix_t) + np.array([origin.x, origin.y])
        velocity = np.array([direction.x, direction.y], dtype=np.float64) * max_distance

        closest_collider = None
        closest_normal = None
        closest_distance = max_distance

        for collider in self.colliders:
            if not collider.enable:
                continue
            if mask != -1 and collider.mask & mask == 0:
                continue
            for polygon in collider.polygons:
                transformed = polygon.apply_transform(collider.global_transform)
                hit, t, normal = _polygon_sweep_numba(
                    rect_vertices, transformed.vertices, velocity)
                if hit:
                    distance = t * max_distance
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_collider = collider
                        closest_normal = Vec2(normal[0], normal[1])

        if closest_collider is None:
            return None
        return closest_collider, origin + direction * closest_distance, closest_normal, closest_distance

    def overlap_sphere(self, center: Vec3, radius: float,
                       mask: int = -1) -> list[object]:
        center2 = Vec2(center.x, center.y)
        bodies: list[object] = []
        for collider in self.overlap_circle(center2, radius, mask):
            bodies.append(self.body_for_collider(collider) or collider)
        return bodies

    def overlap_circle(self, center: Vec2, radius: float,
                       mask: int = -1) -> list[Collider]:
        result: list[Collider] = []
        point_array = np.array([center.x, center.y], dtype=np.float64)
        for collider in self.colliders:
            if not collider.enable:
                continue
            if mask != -1 and collider.mask & mask == 0:
                continue
            if self._circle_hits_collider(point_array, radius, collider):
                result.append(collider)
        return result

    def body_for_collider(self, collider: Collider) -> Rigidbody | None:
        for body in self.rigidbodies:
            if body.collider is collider:
                return body
        return None

    def destroy(self) -> None:
        for body in list(self.rigidbodies):
            body._world = None
        for collider in list(self.colliders):
            collider._world = None
        self.rigidbodies.clear()
        self.colliders.clear()
        self._accumulator = 0.0

    def _step_fixed(self, dt: float) -> None:
        for body in list(self.rigidbodies):
            body._integrate(dt, self._gravity)

        bodies = list(self.rigidbodies)
        for i in range(len(bodies)):
            rb1 = bodies[i]
            if not self._can_collide(rb1):
                continue
            for j in range(i + 1, len(bodies)):
                rb2 = bodies[j]
                if not self._can_collide(rb2):
                    continue
                if rb1.is_kinematic and rb2.is_kinematic:
                    continue

                colliding, mtv_np = rb1.collider.check_collision_global(rb2.collider)
                if colliding and mtv_np is not None:
                    self._resolve_collision(rb1, rb2, Vec2(mtv_np[0], mtv_np[1]))

    def _can_collide(self, body: Rigidbody) -> bool:
        return bool(body.enable and body.collider is not None and body.collider.enable)

    @staticmethod
    def _resolve_collision(rb1: Rigidbody, rb2: Rigidbody, mtv: Vec2) -> None:
        collision_normal = mtv.normalize()
        penetration_depth = mtv.magnitude()
        relative_velocity = rb2.velocity - rb1.velocity
        vel_along_normal = relative_velocity.dot(collision_normal)
        total_inv_mass = rb1.inv_mass + rb2.inv_mass

        if total_inv_mass <= 0:
            return

        if vel_along_normal <= 0:
            restitution = min(rb1.restitution, rb2.restitution)
            impulse_mag = -(1 + restitution) * vel_along_normal
            impulse_mag /= total_inv_mass
            impulse = collision_normal * impulse_mag
            if not rb1.is_kinematic:
                rb1.velocity -= impulse * rb1.inv_mass
            if not rb2.is_kinematic:
                rb2.velocity += impulse * rb2.inv_mass

        percent = 0.8
        slop = 0.01
        correction_amount = max(penetration_depth - slop, 0.0) / total_inv_mass * percent
        correction = collision_normal * correction_amount

        if not rb1.is_kinematic:
            rb1.transform.positionVec2 -= correction * rb1.inv_mass
            rb1.global_transform.positionVec2 -= correction * rb1.inv_mass
        if not rb2.is_kinematic:
            rb2.transform.positionVec2 += correction * rb2.inv_mass
            rb2.global_transform.positionVec2 += correction * rb2.inv_mass

    @staticmethod
    def _circle_hits_collider(point_array: np.ndarray, radius: float, collider: Collider) -> bool:
        for polygon in collider.polygons:
            transformed = polygon.apply_transform(collider.global_transform)
            if _point_in_polygon(point_array, transformed.vertices):
                return True
            if _polygon_distance_sq(point_array, transformed.vertices) <= radius * radius:
                return True
        return False

    @staticmethod
    def _to_vec2(value: Vec2 | Vec3) -> Vec2:
        if isinstance(value, Vec3):
            return Vec2(value.x, value.y)
        return value


def _point_in_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    x, y = point
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def _polygon_distance_sq(point: np.ndarray, vertices: np.ndarray) -> float:
    closest = float("inf")
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            candidate = float(np.dot(point - a, point - a))
        else:
            t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
            projection = a + ab * t
            candidate = float(np.dot(point - projection, point - projection))
        closest = min(closest, candidate)
    return closest
