import math

import pygame as pg

from .Component import Component, Transform
from .. import Game
from ..Geometry import Vec3, Ray, HitInfo


class Hittable(Component):
    cameras: list['Camera']

    def __init__(self):
        self.cameras = []
        if Camera.instance():
            self.cameras.append(Camera.instance())

    def init(self):
        if Camera.instance():
            Camera.instance().add_hittable(self)

    def on_destroy(self):
        for camera in self.cameras:
            camera.remove_hittable(self)
        self.on_destroy = lambda: None

    def add_camera(self, camera: 'Camera'):
        if camera not in self.cameras:
            camera.add_hittable(self)
            self.cameras.append(camera)

    def remove_camera(self, camera: 'Camera'):
        if camera in self.cameras:
            self.cameras.remove(camera)
        camera.remove_hittable(self)

    def clear_cameras(self):
        for camera in self.cameras:
            camera.remove_hittable(self)
        self.cameras.clear()

    def remove_main_camera(self):
        if Camera.instance():
            self.remove_camera(Camera.instance())

    def intersect(self, ray: Ray) -> HitInfo:
        """Método abstrato para calcular a interseção do raio com o objeto."""
        raise NotImplementedError("O método de interseção deve ser implementado por subclasses de Hittable.")


class Camera(Component):
    instances: dict[int, 'Camera'] = {}

    @staticmethod
    def instance() -> 'Camera | None':
        return Camera.instances.get(Game.current_instance)

    @property
    def screen(self):
        return self._screen if self._screen is not None else self.game.screen

    @property
    def image_width(self) -> int:
        return self.screen.get_width()

    @property
    def image_height(self) -> int:
        return self.screen.get_height()

    def __init__(self, screen: pg.Surface = None, vfov: float = 90.0):
        """
        Cria uma câmara 3D para ray tracing.
        :param screen: A superfície do pygame para renderizar. Se for None, usa o ecrã principal do jogo.
        :param vfov: Campo de visão vertical em graus.
        """
        super().__init__()
        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        self.hittables: list[Hittable] = []
        self._screen = screen

        # Propriedades da Câmara
        self.vfov = vfov
        self.center = Vec3(0.0, 0.0, 0.0)

        # Valores calculados no 'init' ou 'loop'
        self.pixel00_loc = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_u = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_v = Vec3(0.0, 0.0, 0.0)

    def on_destroy(self):
        if Game.current_instance in Camera.instances and Camera.instances[Game.current_instance] == self:
            del Camera.instances[Game.current_instance]
        self.on_destroy = lambda: None

    def add_hittable(self, hittable: Hittable):
        if hittable not in self.hittables:
            self.hittables.append(hittable)

    def remove_hittable(self, hittable: Hittable):
        if hittable in self.hittables:
            self.hittables.remove(hittable)

    def _update_camera_geometry(self):
        """Calcula a geometria da câmara para a renderização."""
        self.center = Transform.Global.position

        # Determina as dimensões do viewport
        focal_length = 1.0
        theta = math.radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2 * h * focal_length
        viewport_width = viewport_height * (self.image_width / self.image_height)

        # Calcula os vetores da base ortonormal u,v,w para a orientação da câmara
        # Assumindo que a câmara olha para -Z por padrão. A rotação do transform irá ajustá-la.
        forward = Vec3(0.0, 0.0, -1.0)
        up = Vec3(0.0, 1.0, 0.0)

        w = self.transform.rotation.rotate_vector(forward).normalize()
        u = self.transform.rotation.rotate_vector(up).cross(w).normalize()
        v = w.cross(u)

        # Calcula os vetores através do viewport horizontal e vertical
        viewport_u = u * viewport_width
        viewport_v = -v * viewport_height  # Negativo porque a origem dos pixels é no canto superior esquerdo

        # Calcula os deltas de pixel horizontal e vertical
        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        # Calcula a localização do pixel do canto superior esquerdo
        viewport_upper_left = self.center - (w * focal_length) - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + (self.pixel_delta_u + self.pixel_delta_v) * 0.5

    def loop(self):
        """O loop principal de renderização. Lança raios para cada pixel."""
        self._update_camera_geometry()

        render_array = pg.surfarray.pixels3d(self.screen)
        for j in range(self.image_height):
            for i in range(self.image_width):
                ray = self._get_ray(i, j)
                color_vec = self._ray_color(ray)

                r = int(255.999 * color_vec.x)
                g = int(255.999 * color_vec.y)
                b = int(255.999 * color_vec.z)

                render_array[i, j] = (r, g, b)
            pg.display.flip()

    def _get_ray(self, i: int, j: int) -> Ray:
        """Obtém um raio da câmara para um pixel específico."""
        pixel_center = self.pixel00_loc + (self.pixel_delta_u * i) + (self.pixel_delta_v * j)
        ray_direction = (pixel_center - self.center).normalize()
        return Ray(self.center, ray_direction)

    def _ray_color(self, ray: Ray) -> Vec3[float]:
        """Calcula a cor de um raio."""
        closest_hit: HitInfo | None = None
        min_dist = float('inf')

        for hittable in self.hittables:
            if hittable.enable:
                hit_info = hittable.intersect(ray)
                if hit_info and hit_info.hit and 0.001 < hit_info.distance < min_dist:
                    min_dist = hit_info.distance
                    closest_hit = hit_info

        if closest_hit:
            # Se atingiu algo, colore com base na normal da superfície
            n = closest_hit.normal
            return Vec3(n.x + 1, n.y + 1, n.z + 1) * 0.5

        # Cor de fundo (gradiente do céu)
        unit_direction = ray.direction.normalize()
        a = 0.5 * (unit_direction.y + 1.0)
        return Vec3(1.0, 1.0, 1.0) * (1.0 - a) + Vec3(0.5, 0.7, 1.0) * a

        # Make a xadrez
        # checker_size = 0.1
        #
        # # if (math.floor(ray.direction.x / checker_size) + math.floor(ray.direction.y / checker_size)) % 2 == 0:
        # #     return Vec3(1.0, 1.0, 1.0)
        # # else:
        # #     return Vec3(0.0, 0.0, 0.0)
