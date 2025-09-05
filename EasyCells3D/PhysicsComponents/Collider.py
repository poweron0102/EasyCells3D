from typing import List, Tuple

import numpy as np
import pygame as pg
from numba import njit, prange

from ..Components.Camera import Camera
from ..Components.Component import Component, Transform
from ..Geometry import Vec2


class Polygon:
    def __init__(self, vertices: List[List[float]] | np.ndarray):
        if isinstance(vertices, list):
            self.vertices = np.array(vertices, dtype=np.float64)
        else:
            self.vertices = vertices

    def get_edges(self) -> np.ndarray:
        """
        Returns the edges of the polygon as an array of vectors.
        """
        # Creates edges by subtracting each vertex from the next one.
        # np.roll shifts elements along an axis.
        return np.roll(self.vertices, shift=-1, axis=0) - self.vertices

    def get_normals(self):
        """
        Returns the normal vectors of the polygon's edges.
        """
        edges = self.get_edges()
        # For an edge (x, y), the perpendicular is (-y, x)
        normals = np.empty_like(edges)
        normals[:, 0] = -edges[:, 1]
        normals[:, 1] = edges[:, 0]
        # Normalize each normal vector
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Avoid division by zero for zero-length edges (which create zero-length normals)
        np.divide(normals, norms, out=normals, where=norms > 1e-9)
        return normals

    def apply_transform(self, transform: Transform) -> 'Polygon':
        """
        Applies a transformation to the polygon.
        """
        c, s = np.cos(transform.angle), np.sin(transform.angle)
        # Transposed rotation matrix for operating on row vectors [x, y]
        rotation_matrix_T = np.array([[c, s], [-s, c]])

        # Rotate, then scale, then translate.
        new_vertices = (self.vertices @ rotation_matrix_T) * transform.scale + np.array([transform.x, transform.y])
        return Polygon(new_vertices)


class Collider(Component):
    compiled: bool = False
    colliders: List['Collider'] = []

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
        """
        polygons: list of Polygon objects
        mask: collision mask (bitwise)
        """
        self.word_position = Transform()
        self.polygons: List[Polygon] = polygons
        self.compile_numba_functions()
        self.mask = mask
        self.debug = debug
        Collider.colliders.append(self)

    def init(self):
        self.word_position = self.CalculateGlobalTransform()

    def on_destroy(self):
        Collider.colliders.remove(self)
        self.on_destroy = lambda: None

    def loop_debug(self):
        self.word_position = Transform.Global
        Camera.instance().debug_draws.append(self.draw)

    def loop_no_debug(self):
        self.word_position = Transform.Global

    def draw(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        """
        For debug only
        """
        position = self.word_position * scale
        position.scale *= scale

        for polygon in self.polygons:
            vertices = polygon.apply_transform(position).vertices
            pg.draw.polygon(
                self.game.screen,
                (255, 0, 0),
                vertices - np.array([cam_x, cam_y]),
                3
            )

    def check_collision_global(self, other: 'Collider') -> Tuple[bool, np.ndarray | None]:
        """
        Checks for collision and returns the collision status and the MTV.
        Returns: (bool, mtv_vector or None)
        """
        for polygon in self.polygons:
            p1_transformed = polygon.apply_transform(self.word_position)
            for other_polygon in other.polygons:
                p2_transformed = other_polygon.apply_transform(other.word_position)

                colliding, mtv = _sat_collision(p1_transformed.vertices, p2_transformed.vertices)
                if colliding:
                    return True, mtv
        return False, None

    def compile_numba_functions(self):
        """
        Compiles numba functions to improve performance.
        """
        if Collider.compiled:
            return

        _sat_collision(self.polygons[0].vertices, self.polygons[0].vertices)
        _ray_polygon_intersection_numba(
            np.array([0, 0], dtype=np.float64),
            np.array([1, 1], dtype=np.float64),
            np.array([[0, 0], [1, 1], [2, 2], [4, 4]], dtype=np.float64),
            10
        )

        print("Collider functions compiled")
        Collider.compiled = True

    # ... (rest of the class methods like bounding_box, ray_cast, etc. remain the same) ...
    def bounding_box(self) -> pg.Rect:
        """
        Retorna o menor retângulo que contém o collider
        """
        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf

        for polygon in self.polygons:
            polygon = polygon.apply_transform(self.word_position)
            for vertex in polygon.vertices:
                min_x = min(min_x, vertex[0])
                min_y = min(min_y, vertex[1])
                max_x = max(max_x, vertex[0])
                max_y = max(max_y, vertex[1])

        return pg.Rect(min_x, min_y, max_x - min_x, max_y - min_y)

    def ray_cast(
            self,
            origin: Vec2[float],
            direction: Vec2[float],
            max_distance: float,
    ) -> 'tuple[Vec2[float], Vec2[float]] | None':
        """
        Usa coordenadas globais
        Retorna:
        Vec2: Ponto de interseção
        Vec2: Normal da superfície atingida
        """
        origin_array = np.array([origin.x, origin.y], dtype=np.float64)
        direction_array = np.array([direction.x, direction.y], dtype=np.float64)

        closest_point = None
        closest_normal = None
        closest_distance = max_distance

        for polygon in self.polygons:
            polygon = polygon.apply_transform(self.word_position)
            intersection, normal, distance = _ray_polygon_intersection_numba(origin_array, direction_array,
                                                                             polygon.vertices, max_distance)

            if intersection is not None and distance < closest_distance:
                closest_point = Vec2(intersection[0], intersection[1])
                closest_normal = Vec2(normal[0], normal[1])
                closest_distance = distance

        if closest_point:
            if np.dot(closest_normal.to_tuple, direction_array) > 0:
                closest_normal *= -1
            return closest_point, closest_normal

        return None

    @staticmethod
    def ray_cast_static(
            origin: Vec2[float],
            direction: Vec2[float],
            max_distance: float,
            mask: int
    ) -> 'tuple[Collider, Vec2[float], Vec2[float]] | None':
        """
        Retorna:
        Collider: Collider atingido pelo raio
        Vec2: Ponto de interseção
        Vec2: Normal da superfície atingida
        """
        origin_array = np.array([origin.x, origin.y], dtype=np.float64)
        direction_array = np.array([direction.x, direction.y], dtype=np.float64)

        closest_collider = None
        closest_point = None
        closest_normal = None
        closest_distance = max_distance

        # Itera sobre todos os colliders
        for collider in Collider.colliders:
            if collider.mask & mask == 0:
                continue

            for polygon in collider.polygons:
                polygon = polygon.apply_transform(collider.word_position)
                intersection, normal, distance = _ray_polygon_intersection_numba(origin_array, direction_array,
                                                                                 polygon.vertices, max_distance)

                if intersection is not None and distance < closest_distance:
                    closest_collider = collider
                    closest_point = Vec2(intersection[0], intersection[1])
                    closest_normal = Vec2(normal[0], normal[1])
                    closest_distance = distance

        if closest_collider:
            if np.dot(closest_normal.to_tuple, direction_array) > 0:
                closest_normal *= -1
            return closest_collider, closest_point, closest_normal

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
        edge_normal = np.array([-edge[1], edge[0]])  # Normal ortogonal à aresta

        # Calcular a interseção do raio com a aresta (v1, v2)
        denom = np.dot(direction, edge_normal)
        if np.abs(denom) < 1e-6:  # Raio é paralelo à aresta
            continue

        t = np.dot(v1 - origin, edge_normal) / denom
        if t < 0 or t > max_distance:  # Interseção acontece fora do alcance ou atrás do raio continue
            continue

        intersection_point = origin + direction * t

        # Verifica se o ponto de interseção está dentro dos limites da aresta
        edge_direction = (v2 - v1) / np.linalg.norm(v2 - v1)
        proj = np.dot(intersection_point - v1, edge_direction)
        if proj < 0 or proj > np.linalg.norm(v2 - v1):
            continue

        if t < closest_distance:
            closest_distance = t
            closest_intersection = intersection_point
            closest_normal = edge_normal / np.linalg.norm(edge_normal)  # Normalizar

    return closest_intersection, closest_normal, closest_distance


@njit()
def project_polygon(vertices, axis):
    """
    Projects the vertices of a polygon onto an axis.
    """
    min_proj = np.inf
    max_proj = -np.inf
    for i in range(len(vertices)):
        projection = np.dot(vertices[i], axis)
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    return min_proj, max_proj


@njit()
def _sat_collision(vertices_a, vertices_b):
    """
    Uses the Separating Axis Theorem (SAT) to check for collision between two convex polygons.
    If a collision occurs, it returns True and the Minimum Translation Vector (MTV).
    Otherwise, it returns False and None.
    """
    min_overlap = np.inf
    mtv_axis = None

    # Get all axes to test (normals of both polygons)
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

    # Test all axes
    for i in range(len(axes)):
        axis = axes[i]
        minA, maxA = project_polygon(vertices_a, axis)
        minB, maxB = project_polygon(vertices_b, axis)

        # Check for separation
        if maxA < minB or maxB < minA:
            return False, None  # Separation detected

        # Calculate overlap
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap < min_overlap:
            min_overlap = overlap
            mtv_axis = axis

    # If we are here, a collision occurred.
    # The MTV is the axis with the minimum overlap, scaled by that overlap.

    center_a = np.sum(vertices_a, axis=0) / vertices_a.shape[0]
    center_b = np.sum(vertices_b, axis=0) / vertices_b.shape[0]
    direction = center_b - center_a
    if np.dot(direction, mtv_axis) < 0:
        mtv_axis = -mtv_axis

    mtv = mtv_axis * min_overlap
    return True, mtv
