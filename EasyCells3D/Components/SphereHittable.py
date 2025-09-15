import math

from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Ray, HitInfo


class SphereHittable(Hittable):
    def __init__(self, radius: float = 0.5):
        super().__init__()
        self.radius = radius

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
        return HitInfo(point=point, normal=normal, distance=root, hit=True)
