from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pyray as rl
from numba import njit, prange

from ..Components import Camera2D
from ..Components.Component import Component, Transform
from ..Geometry import Vec2


class Polygon:
    def __init__(self, vertices: List[List[float]] | np.ndarray):
        if isinstance(vertices, list):
            self.vertices = np.array(vertices, dtype=np.float64)
        else:
            self.vertices = vertices

    def get_edges(self) -> np.ndarray:
        return np.roll(self.vertices, shift=-1, axis=0) - self.vertices

    def get_normals(self):
        edges = self.get_edges()
        normals = np.empty_like(edges)
        normals[:, 0] = -edges[:, 1]
        normals[:, 1] = edges[:, 0]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        np.divide(normals, norms, out=normals, where=norms > 1e-9)
        return normals

    def apply_transform(self, transform: Transform) -> "Polygon":
        angle_rad = np.radians(transform.angle)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix_t = np.array([[c, s], [-s, c]])
        vertices = (self.vertices @ rotation_matrix_t) * transform.scale.x
        vertices += np.array([transform.x, transform.y])
        return Polygon(vertices)


class Collider(Component):
    compiled: bool = False

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value
        if value:
            self.loop = self.loop_debug
        else:
            self.loop = self.loop_no_debug

    def __init__(self, polygons: List[Polygon], mask: int = 1, debug: bool = False):
        self.polygons: List[Polygon] = polygons
        self.compile_numba_functions()
        self.mask = mask
        self.debug = debug
        self._world = None

    def init(self):
        self._register_with_world()

    def _register_with_world(self):
        from .SATPhysicsWorld import SATPhysicsWorld

        world = self.game.physics_world
        if not isinstance(world, SATPhysicsWorld):
            raise RuntimeError(
                "Collider 2D requer game.physics_world = SATPhysicsWorld() "
                "no init() do level."
            )
        self._world = world
        world.add_collider(self)

    def on_destroy(self):
        if self._world is not None:
            self._world.remove_collider(self)
            self._world = None

    def loop_debug(self):
        if Camera2D.main:
            for polygon in self.polygons:
                poly_transformed = polygon.apply_transform(self.global_transform)
                points = [Vec2(p[0], p[1]) for p in poly_transformed.vertices]
                points.append(points[0])
                Camera2D.main.debug_polygon.append((points, rl.RED))

    def loop_no_debug(self):
        pass

    def check_collision_global(self, other: "Collider") -> Tuple[bool, np.ndarray | None]:
        has_collision = False
        best_mtv = None
        max_penetration_sq = -1.0

        for polygon in self.polygons:
            p1_transformed = polygon.apply_transform(self.global_transform)
            for other_polygon in other.polygons:
                p2_transformed = other_polygon.apply_transform(other.global_transform)
                colliding, mtv = _sat_collision(
                    p1_transformed.vertices, p2_transformed.vertices)
                if colliding:
                    has_collision = True
                    penetration_sq = mtv[0] ** 2 + mtv[1] ** 2
                    if penetration_sq > max_penetration_sq:
                        max_penetration_sq = penetration_sq
                        best_mtv = mtv

        return has_collision, best_mtv

    def is_point_inside(self, point: Vec2) -> bool:
        point_array = np.array([point.x, point.y], dtype=np.float64)
        for polygon in self.polygons:
            poly_transformed = polygon.apply_transform(self.global_transform)
            if _is_point_in_polygon_numba(point_array, poly_transformed.vertices):
                return True
        return False

    def compile_numba_functions(self):
        if Collider.compiled or not self.polygons:
            return

        _sat_collision(self.polygons[0].vertices, self.polygons[0].vertices)
        _ray_polygon_intersection_numba(
            np.array([0, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([[0, 0], [1, 1], [2, 2], [4, 4]], dtype=np.float64),
            10,
        )
        _is_point_in_polygon_numba(
            np.array([0, 0], dtype=np.float64), self.polygons[0].vertices)
        _polygon_sweep_numba(
            self.polygons[0].vertices,
            self.polygons[0].vertices,
            np.array([1.0, 0.0], dtype=np.float64),
        )

        print("Collider functions compiled")
        Collider.compiled = True

    def bounding_box(self) -> rl.Rectangle:
        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf

        for polygon in self.polygons:
            polygon = polygon.apply_transform(self.global_transform)
            for vertex in polygon.vertices:
                min_x = min(min_x, vertex[0])
                min_y = min(min_y, vertex[1])
                max_x = max(max_x, vertex[0])
                max_y = max(max_y, vertex[1])

        return rl.Rectangle(min_x, min_y, max_x - min_x, max_y - min_y)

    def ray_cast(
            self,
            origin: Vec2,
            direction: Vec2,
            max_distance: float,
    ) -> tuple[Vec2, Vec2] | None:
        origin_array = np.array([origin.x, origin.y], dtype=np.float64)
        direction_array = np.array([direction.x, direction.y], dtype=np.float64)

        closest_point = None
        closest_normal = None
        closest_distance = max_distance

        for polygon in self.polygons:
            polygon = polygon.apply_transform(self.global_transform)
            intersection, normal, distance = _ray_polygon_intersection_numba(
                origin_array, direction_array, polygon.vertices, max_distance)

            if intersection is not None and distance < closest_distance:
                closest_point = Vec2(intersection[0], intersection[1])
                closest_normal = Vec2(normal[0], normal[1])
                closest_distance = distance

        if closest_point:
            if np.dot(closest_normal.to_tuple, direction_array) > 0:
                closest_normal *= -1
            return closest_point, closest_normal

        return None


@njit
def _ray_polygon_intersection_numba(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray,
                                    max_distance: float):
    closest_intersection = None
    closest_normal = None
    closest_distance = max_distance
    num_vertices = len(vertices)

    for i in prange(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]
        edge = v2 - v1
        edge_normal = np.array([-edge[1], edge[0]])
        denom = np.dot(direction, edge_normal)
        if np.abs(denom) < 1e-6:
            continue

        t = np.dot(v1 - origin, edge_normal) / denom
        if t < 0 or t > max_distance:
            continue

        intersection_point = origin + direction * t
        edge_length = np.linalg.norm(edge)
        if edge_length <= 1e-9:
            continue
        edge_direction = edge / edge_length
        proj = np.dot(intersection_point - v1, edge_direction)
        if proj < 0 or proj > edge_length:
            continue

        if t < closest_distance:
            closest_distance = t
            closest_intersection = intersection_point
            closest_normal = edge_normal / np.linalg.norm(edge_normal)

    return closest_intersection, closest_normal, closest_distance


@njit
def _is_point_in_polygon_numba(point: np.ndarray, vertices: np.ndarray) -> bool:
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


@njit()
def project_polygon(vertices, axis):
    min_proj = np.inf
    max_proj = -np.inf
    for i in range(len(vertices)):
        projection = np.dot(vertices[i], axis)
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


@njit()
def _sat_collision(vertices_a, vertices_b):
    min_overlap = np.inf
    mtv_axis = None

    axes = []
    for i in range(len(vertices_a)):
        v1 = vertices_a[i]
        v2 = vertices_a[(i + 1) % len(vertices_a)]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm > 0:
            axes.append(axis / norm)

    for i in range(len(vertices_b)):
        v1 = vertices_b[i]
        v2 = vertices_b[(i + 1) % len(vertices_b)]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm > 0:
            axes.append(axis / norm)

    for i in range(len(axes)):
        axis = axes[i]
        min_a, max_a = project_polygon(vertices_a, axis)
        min_b, max_b = project_polygon(vertices_b, axis)
        if max_a < min_b or max_b < min_a:
            return False, None

        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = axis

    center_a = np.sum(vertices_a, axis=0) / vertices_a.shape[0]
    center_b = np.sum(vertices_b, axis=0) / vertices_b.shape[0]
    direction = center_b - center_a
    if np.dot(direction, mtv_axis) < 0:
        mtv_axis = -mtv_axis

    mtv = mtv_axis * min_overlap
    return True, mtv


@njit
def _polygon_sweep_numba(vertices_a, vertices_b, velocity):
    t_min = 0.0
    t_max = 1.0
    collision_normal = np.zeros(2)

    for i in range(len(vertices_a)):
        v1 = vertices_a[i]
        v2 = vertices_a[(i + 1) % len(vertices_a)]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis /= norm

        min_a, max_a = project_polygon(vertices_a, axis)
        min_b, max_b = project_polygon(vertices_b, axis)
        v_proj = np.dot(velocity, axis)

        if abs(v_proj) < 1e-9:
            if max_a < min_b or max_b < min_a:
                return False, 0.0, np.zeros(2)
        else:
            if v_proj > 0:
                t_enter = (min_b - max_a) / v_proj
                t_leave = (max_b - min_a) / v_proj
            else:
                t_enter = (max_b - min_a) / v_proj
                t_leave = (min_b - max_a) / v_proj

            if t_enter > t_min:
                t_min = t_enter
                collision_normal = -axis if v_proj > 0 else axis
            if t_leave < t_max:
                t_max = t_leave
            if t_min > t_max:
                return False, 0.0, np.zeros(2)

    for i in range(len(vertices_b)):
        v1 = vertices_b[i]
        v2 = vertices_b[(i + 1) % len(vertices_b)]
        edge = v2 - v1
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis /= norm

        min_a, max_a = project_polygon(vertices_a, axis)
        min_b, max_b = project_polygon(vertices_b, axis)
        v_proj = np.dot(velocity, axis)

        if abs(v_proj) < 1e-9:
            if max_a < min_b or max_b < min_a:
                return False, 0.0, np.zeros(2)
        else:
            if v_proj > 0:
                t_enter = (min_b - max_a) / v_proj
                t_leave = (max_b - min_a) / v_proj
            else:
                t_enter = (max_b - min_a) / v_proj
                t_leave = (min_b - max_a) / v_proj

            if t_enter > t_min:
                t_min = t_enter
                collision_normal = -axis if v_proj > 0 else axis
            if t_leave < t_max:
                t_max = t_leave
            if t_min > t_max:
                return False, 0.0, np.zeros(2)

    return True, t_min, collision_normal
