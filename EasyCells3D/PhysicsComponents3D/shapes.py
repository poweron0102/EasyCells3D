"""Shapes de colisão 3D.

Cada shape é uma dataclass com uma *tag de tipo* (``kind``) e sabe construir o
``collisionShape`` correspondente no Bullet via :meth:`build`. Os campos são
primitivos/``Vec3`` e há ``to_dict``/``from_dict`` para deixar a serialização
pronta desde já, mesmo sem integrar o ``SceneLoader`` na v1.

Convenção de eixos: o mundo é Y-up. As primitivas capsule/cylinder do Bullet
são alinhadas ao eixo Z; por isso recebem um ``up_axis`` (padrão ``"y"``) e,
quando Y, são rotacionadas -90° em X para ficarem em pé.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import pybullet as p

from ..Geometry import Vec3

# Quaternion (Bullet xyzw) que rotaciona o eixo Z para o eixo Y (-90° em X).
_Z_TO_Y = [-math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]


@dataclass
class CollisionShape:
    """Base abstrata de todas as shapes."""

    kind: ClassVar[str] = "shape"

    def build(self, client: int, scale: Vec3) -> int:
        """Cria o collisionShape no Bullet e retorna seu id.

        A ``scale`` (escala global do transform) é embutida nas dimensões, pois
        o pybullet não expõe ``setLocalScaling`` para primitivas.
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict) -> "CollisionShape":
        kind = data.get("kind")
        cls = _SHAPE_REGISTRY.get(kind)
        if cls is None:
            raise ValueError(f"Shape desconhecida: {kind!r}")
        return cls._from_dict(data)


@dataclass
class BoxShape(CollisionShape):
    kind: ClassVar[str] = "box"
    half_extents: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.5))

    def build(self, client: int, scale: Vec3) -> int:
        he = [
            abs(self.half_extents.x * scale.x),
            abs(self.half_extents.y * scale.y),
            abs(self.half_extents.z * scale.z),
        ]
        return p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=client)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "half_extents": self.half_extents.to_tuple}

    @classmethod
    def _from_dict(cls, data: dict) -> "BoxShape":
        return cls(Vec3.from_tuple(data["half_extents"]))


@dataclass
class SphereShape(CollisionShape):
    kind: ClassVar[str] = "sphere"
    radius: float = 0.5

    def build(self, client: int, scale: Vec3) -> int:
        # Esfera só suporta escala uniforme; usamos o maior componente.
        s = max(abs(scale.x), abs(scale.y), abs(scale.z))
        return p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.radius * s, physicsClientId=client
        )

    def to_dict(self) -> dict:
        return {"kind": self.kind, "radius": self.radius}

    @classmethod
    def _from_dict(cls, data: dict) -> "SphereShape":
        return cls(float(data["radius"]))


@dataclass
class CapsuleShape(CollisionShape):
    kind: ClassVar[str] = "capsule"
    radius: float = 0.5
    # Altura do segmento cilíndrico (entre os centros das duas semiesferas).
    height: float = 1.0
    up_axis: str = "y"

    def build(self, client: int, scale: Vec3) -> int:
        r = self.radius * max(abs(scale.x), abs(scale.z))
        h = self.height * abs(scale.y)
        kwargs = dict(radius=r, height=h, physicsClientId=client)
        if self.up_axis == "y":
            kwargs["collisionFrameOrientation"] = _Z_TO_Y
        return p.createCollisionShape(p.GEOM_CAPSULE, **kwargs)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "radius": self.radius, "height": self.height, "up_axis": self.up_axis}

    @classmethod
    def _from_dict(cls, data: dict) -> "CapsuleShape":
        return cls(float(data["radius"]), float(data["height"]), data.get("up_axis", "y"))


@dataclass
class CylinderShape(CollisionShape):
    kind: ClassVar[str] = "cylinder"
    radius: float = 0.5
    height: float = 1.0
    up_axis: str = "y"

    def build(self, client: int, scale: Vec3) -> int:
        r = self.radius * max(abs(scale.x), abs(scale.z))
        h = self.height * abs(scale.y)
        kwargs = dict(radius=r, height=h, physicsClientId=client)
        if self.up_axis == "y":
            kwargs["collisionFrameOrientation"] = _Z_TO_Y
        return p.createCollisionShape(p.GEOM_CYLINDER, **kwargs)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "radius": self.radius, "height": self.height, "up_axis": self.up_axis}

    @classmethod
    def _from_dict(cls, data: dict) -> "CylinderShape":
        return cls(float(data["radius"]), float(data["height"]), data.get("up_axis", "y"))


@dataclass
class ConvexHullShape(CollisionShape):
    """Casco convexo a partir de uma nuvem de vértices. Pode ser DYNAMIC."""

    kind: ClassVar[str] = "convex_hull"
    vertices: list = field(default_factory=list)  # lista de (x, y, z)

    def build(self, client: int, scale: Vec3) -> int:
        return p.createCollisionShape(
            p.GEOM_MESH,
            vertices=[list(v) for v in self.vertices],
            meshScale=[scale.x, scale.y, scale.z],
            physicsClientId=client,
        )

    def to_dict(self) -> dict:
        return {"kind": self.kind, "vertices": [list(v) for v in self.vertices]}

    @classmethod
    def _from_dict(cls, data: dict) -> "ConvexHullShape":
        return cls([tuple(v) for v in data["vertices"]])


@dataclass
class TriangleMeshShape(CollisionShape):
    """Malha triangular côncava (``btBvhTriangleMeshShape``). Só STATIC."""

    kind: ClassVar[str] = "triangle_mesh"
    vertices: list = field(default_factory=list)  # lista de (x, y, z)
    indices: list = field(default_factory=list)   # lista plana de ints

    def build(self, client: int, scale: Vec3) -> int:
        return p.createCollisionShape(
            p.GEOM_MESH,
            vertices=[list(v) for v in self.vertices],
            indices=list(self.indices),
            meshScale=[scale.x, scale.y, scale.z],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=client,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "vertices": [list(v) for v in self.vertices],
            "indices": list(self.indices),
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "TriangleMeshShape":
        return cls([tuple(v) for v in data["vertices"]], list(data["indices"]))


@dataclass
class CompoundChild:
    shape: CollisionShape
    position: Vec3 = field(default_factory=Vec3.zero)
    # Quaternion no formato Bullet [x, y, z, w].
    orientation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])


@dataclass
class CompoundShape(CollisionShape):
    """Compõe várias primitivas num único corpo.

    Implementado via ``createCollisionShapeArray``; suporta apenas filhos
    primitivos (box/sphere/capsule/cylinder).
    """

    kind: ClassVar[str] = "compound"
    children: list = field(default_factory=list)  # list[CompoundChild]

    _ARRAY_TYPE = {
        "box": "GEOM_BOX",
        "sphere": "GEOM_SPHERE",
        "capsule": "GEOM_CAPSULE",
        "cylinder": "GEOM_CYLINDER",
    }

    def build(self, client: int, scale: Vec3) -> int:
        shape_types, radii, half_extents, lengths = [], [], [], []
        frame_pos, frame_orn = [], []
        for child in self.children:
            shp = child.shape
            type_name = self._ARRAY_TYPE.get(shp.kind)
            if type_name is None:
                raise ValueError(
                    f"CompoundShape suporta apenas primitivas; recebido {shp.kind!r}"
                )
            shape_types.append(getattr(p, type_name))
            radii.append(getattr(shp, "radius", 0.0) * max(abs(scale.x), abs(scale.z)))
            if isinstance(shp, BoxShape):
                half_extents.append(
                    [abs(shp.half_extents.x * scale.x),
                     abs(shp.half_extents.y * scale.y),
                     abs(shp.half_extents.z * scale.z)]
                )
            else:
                half_extents.append([0.0, 0.0, 0.0])
            lengths.append(getattr(shp, "height", 0.0) * abs(scale.y))
            frame_pos.append([child.position.x * scale.x,
                              child.position.y * scale.y,
                              child.position.z * scale.z])
            frame_orn.append(list(child.orientation))

        return p.createCollisionShapeArray(
            shapeTypes=shape_types,
            radii=radii,
            halfExtents=half_extents,
            lengths=lengths,
            collisionFramePositions=frame_pos,
            collisionFrameOrientations=frame_orn,
            physicsClientId=client,
        )

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "children": [
                {
                    "shape": c.shape.to_dict(),
                    "position": c.position.to_tuple,
                    "orientation": list(c.orientation),
                }
                for c in self.children
            ],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "CompoundShape":
        children = [
            CompoundChild(
                CollisionShape.from_dict(c["shape"]),
                Vec3.from_tuple(c["position"]),
                list(c["orientation"]),
            )
            for c in data["children"]
        ]
        return cls(children)


_SHAPE_REGISTRY = {
    cls.kind: cls
    for cls in (
        BoxShape, SphereShape, CapsuleShape, CylinderShape,
        ConvexHullShape, TriangleMeshShape, CompoundShape,
    )
}


def extract_mesh_geometry(model) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Extrai (vertices, indices) de um ``rl.Model`` (todas as meshes juntas).

    Usado para auto-construir colisores de ``StaticModel``.
    """
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    mesh_count = int(getattr(model, "mesh_count", getattr(model, "meshCount", 0)) or 0)
    offset = 0
    for m in range(mesh_count):
        mesh = model.meshes[m]
        vertex_count = int(getattr(mesh, "vertexCount", getattr(mesh, "vertex_count", 0)) or 0)
        triangle_count = int(getattr(mesh, "triangleCount", getattr(mesh, "triangle_count", 0)) or 0)
        verts_ptr = mesh.vertices
        for i in range(vertex_count):
            vertices.append((verts_ptr[i * 3], verts_ptr[i * 3 + 1], verts_ptr[i * 3 + 2]))

        idx_ptr = getattr(mesh, "indices", None)
        if idx_ptr:
            for i in range(triangle_count * 3):
                indices.append(offset + int(idx_ptr[i]))
        else:
            for i in range(triangle_count * 3):
                indices.append(offset + i)
        offset += vertex_count
    return vertices, indices
