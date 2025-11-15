import math

from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Ray, HitInfo, Vec3, Vec2
from EasyCells3D.Material import Material


class SphereHittable(Hittable):
    def __init__(self, radius: float = 0.5, material: Material = None):
        super().__init__()
        self.radius = radius
        self.material = material if material is not None else Material()
        # 'word_position' será definido no 'init' para garantir que o transform já existe.
        self.word_position = None

    def init(self):
        super().init()
        self.word_position = self.transform

    def get_sphere_uv(self, world_normal: Vec3[float]) -> Vec2[float]:
        """
        Calcula as coordenadas UV para um ponto na esfera, considerando a rotação do objeto.
        A normal do mundo é transformada para o espaço local do objeto antes do cálculo.
        """
        # Usa a rotação global atual para transformar a normal.
        local_normal = self.word_position.rotation.inverse().rotate_vector(world_normal)

        # Evita erro de domínio para math.asin
        clamped_y = max(-1.0, min(1.0, local_normal.y))
        theta = math.asin(clamped_y)
        phi = math.atan2(local_normal.z, local_normal.x)

        u = 1 - (phi + math.pi) / (2 * math.pi)
        v = (theta + math.pi / 2) / math.pi
        return Vec2(u, v)

    def intersect(self, ray: Ray) -> HitInfo | None:
        """
        Calcula a interseção de um raio com a esfera no espaço global.
        """
        # Usa a posição e escala globais atuais para o cálculo.
        center = self.word_position.position
        radius = self.radius * self.word_position.scale.x # Assume escala uniforme

        oc = ray.origin - center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - radius * radius
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

        return HitInfo(point=point, normal=normal, distance=root, hit=True, uv=uv, material=self.material)

    def loop(self):
        """
        Atualiza a transformação global da esfera a cada frame.
        Isso é crucial para que a renderização (CPU ou GPU) use a posição,
        rotação e escala mais recentes.
        """
        self.word_position = self.transform.Global
