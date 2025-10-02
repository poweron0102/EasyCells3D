import math

from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Ray, HitInfo, Vec3, Vec2
from EasyCells3D.Material import Material


class SphereHittable(Hittable):
    def __init__(self, radius: float = 0.5, material: Material = None):
        super().__init__()
        self.radius = radius
        self.material = material if material is not None else Material()

    @staticmethod
    def get_sphere_uv(normal: Vec3[float]) -> Vec2[float]:
        """Calcula as coordenadas UV para um ponto numa esfera, dada a normal."""
        # normal é um vetor unitário.
        # phi = atan2(z, x)
        # theta = asin(y)
        # u = 1 - (phi + pi) / (2 * pi)
        # v = (theta + pi/2) / pi
        phi = math.atan2(normal.z, normal.x)
        theta = math.asin(normal.y)
        u = 1 - (phi + math.pi) / (2 * math.pi)
        v = (theta + math.pi / 2) / math.pi
        return Vec2(u, v)

    def intersect(self, ray: Ray) -> HitInfo | None:
        center = self.transform.position
        oc = ray.origin - center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrt_d = math.sqrt(discriminant)
        root = (-half_b - sqrt_d) / a
        if root <= 0.001:
            root = (-half_b + sqrt_d) / a
            if root <= 0.001:
                return None

        point = ray.point_at(root)
        normal = (point - center).normalize()
        uv = self.get_sphere_uv(normal)

        return HitInfo(point=point, normal=normal, distance=root, hit=True, uv=uv)
