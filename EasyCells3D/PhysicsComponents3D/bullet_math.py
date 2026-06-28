"""Helpers de conversão entre os tipos da engine e o PyBullet.

PyBullet usa quaternions no formato ``[x, y, z, w]`` e vetores como listas
``[x, y, z]``. A engine usa ``Quaternion(w, x, y, z)`` e ``Vec3(x, y, z)``.
Tudo que cruza essa fronteira passa por aqui para manter a conversão num
único lugar.
"""
from __future__ import annotations

from ..Geometry import Quaternion, Vec3


def vec3_to_bullet(v: Vec3) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def bullet_to_vec3(t) -> Vec3:
    return Vec3(float(t[0]), float(t[1]), float(t[2]))


def quat_to_bullet(q: Quaternion) -> list[float]:
    """Quaternion(w, x, y, z) -> [x, y, z, w] (formato do Bullet)."""
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def bullet_to_quat(t) -> Quaternion:
    """[x, y, z, w] (Bullet) -> Quaternion(w, x, y, z)."""
    return Quaternion(float(t[3]), float(t[0]), float(t[1]), float(t[2]))
