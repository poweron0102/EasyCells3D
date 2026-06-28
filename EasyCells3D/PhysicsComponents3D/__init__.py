"""Física 3D do EasyCells3D (backend PyBullet).

Uso típico no ``init()`` de um level::

    from EasyCells3D.PhysicsComponents3D import (
        BulletPhysicsWorld, PhysicsBody3D, BodyType, BoxShape,
    )

    def init(game):
        game.physics_world = BulletPhysicsWorld()
        chao = game.CreateItem()
        chao.AddComponent(PhysicsBody3D(BoxShape(Vec3(10, 0.5, 10)),
                                        body_type=BodyType.STATIC))

O ``game`` ticka o mundo automaticamente (hook no ``Game.run``) e o demole na
troca de cena.
"""
from .world import BodyType, BulletPhysicsWorld, PhysicsWorld, RaycastHit
from .body import PhysicsBody3D
from .character import CharacterController3D
from .debug import PhysicsDebugRenderer
from .shapes import (
    BoxShape, CapsuleShape, CompoundChild, CompoundShape, ConvexHullShape,
    CylinderShape, SphereShape, TriangleMeshShape, CollisionShape,
)

__all__ = [
    "PhysicsWorld",
    "BulletPhysicsWorld",
    "BodyType",
    "RaycastHit",
    "PhysicsBody3D",
    "CharacterController3D",
    "PhysicsDebugRenderer",
    "CollisionShape",
    "BoxShape",
    "SphereShape",
    "CapsuleShape",
    "CylinderShape",
    "ConvexHullShape",
    "TriangleMeshShape",
    "CompoundShape",
    "CompoundChild",
]
