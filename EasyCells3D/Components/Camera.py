import math
import numpy as np
import pygame as pg

from .Component import Component
from .. import Game
from ..Geometry import Vec3, Ray, HitInfo
from EasyCells3D.CudaRenderer import CudaRenderer
import pycuda.driver as cuda

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

    def __init__(self, screen: pg.Surface = None, vfov: float = 90.0, use_cuda=True, light_direction: Vec3 = None, ambient_light: Vec3 = Vec3(0.1, 0.1, 0.1)):
        super().__init__()
        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        self.hittables: list[Hittable] = []
        self._screen = screen
        self.use_cuda = use_cuda
        self.cuda_renderer = None

        self.vfov = vfov
        self.center = Vec3(0.0, 0.0, 0.0)
        self.ambient_light = ambient_light
        self.light_direction = light_direction.normalize() if light_direction else Vec3(0.0, 1.0, -1.0).normalize()

        self.pixel00_loc = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_u = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_v = Vec3(0.0, 0.0, 0.0)

        self._render_array_np = None

    def init(self):
        if self.use_cuda:
            # try:
            #     # A inicialização do PyCUDA é tratada no CudaRenderer
            #     self.cuda_renderer = CudaRenderer(self.image_width, self.image_height)
            #     print("Dispositivo CUDA detectado. A renderização será feita na GPU.")
            # except Exception as e:
            #     print(f"Erro ao inicializar o PyCUDA: {e}")
            #     print("A renderizar na CPU.")
            #     self.use_cuda = False

            self.cuda_renderer = CudaRenderer(self.image_width, self.image_height)
            print("Dispositivo CUDA detectado. A renderização será feita na GPU.")

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
        self.center = self.transform.Global.position

        focal_length = 1.0
        theta = math.radians(self.vfov)
        h = math.tan(theta / 2)
        viewport_height = 2 * h * focal_length
        viewport_width = viewport_height * (self.image_width / self.image_height)

        forward = Vec3(0.0, 0.0, 1.0)
        up = Vec3(0.0, 1.0, 0.0)

        w = self.transform.rotation.rotate_vector(forward).normalize()
        u = self.transform.rotation.rotate_vector(up).cross(w).normalize()
        v = w.cross(u)

        viewport_u = u * viewport_width
        viewport_v = -v * viewport_height

        self.pixel_delta_u = viewport_u / self.image_width
        self.pixel_delta_v = viewport_v / self.image_height

        viewport_upper_left = self.center - (w * focal_length) - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + (self.pixel_delta_u + self.pixel_delta_v) * 0.5

        if self._render_array_np is None or self._render_array_np.shape[1] != self.image_width or \
                self._render_array_np.shape[0] != self.image_height:
            self._render_array_np = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            if self.use_cuda and self.cuda_renderer:
                self.cuda_renderer.width = self.image_width
                self.cuda_renderer.height = self.image_height

    def loop(self):
        self._update_camera_geometry()

        if self.use_cuda and self.cuda_renderer:
            self._render_cuda()
        else:
            self._render_cpu()

        pg.surfarray.blit_array(self.screen, np.transpose(self._render_array_np, (1, 0, 2)))

    def _render_cuda(self):
        # A cena é composta pelos hittables adicionados à câmara
        scene = self.game.scene
        self._render_array_np = self.cuda_renderer.render(self, scene, self.light_direction, self.ambient_light)

    def _render_cpu(self):
        render_array = np.transpose(self._render_array_np, (1, 0, 2))
        for j in range(self.image_height):
            for i in range(self.image_width):
                ray = self._get_ray(i, j)
                color_vec = self._ray_color(ray)

                r = int(255.999 * np.clip(color_vec.x, 0, 1))
                g = int(255.999 * np.clip(color_vec.y, 0, 1))
                b = int(255.999 * np.clip(color_vec.z, 0, 1))

                render_array[i, j] = (r, g, b)

    def _get_ray(self, i: int, j: int) -> Ray:
        pixel_center = self.pixel00_loc + (self.pixel_delta_u * i) + (self.pixel_delta_v * j)
        ray_direction = (pixel_center - self.center).normalize()
        return Ray(self.center, ray_direction)

    def _ray_color(self, ray: Ray) -> Vec3[float]:
        closest_hit: HitInfo | None = None
        closest_hittable: Hittable | None = None
        min_dist = float('inf')

        for hittable in self.hittables:
            if hittable.enable:
                hit_info = hittable.intersect(ray)
                if hit_info and hit_info.hit and 0.001 < hit_info.distance < min_dist:
                    min_dist = hit_info.distance
                    closest_hit = hit_info
                    closest_hittable = hittable

        if closest_hit:
            if hasattr(closest_hittable, 'material') and closest_hit.uv is not None:
                material = closest_hittable.material
                albedo = material.get_color_at(closest_hit.uv.x, closest_hit.uv.y)
                emissive = material.emissive_color
                light_dir = self.light_direction
                diffuse_intensity = max(0.0, closest_hit.normal.dot(light_dir))
                diffuse = albedo * diffuse_intensity
                view_dir = (ray.origin - closest_hit.point).normalize()
                half_vector = (light_dir + view_dir).normalize()
                specular_intensity = pow(max(0.0, closest_hit.normal.dot(half_vector)), material.shininess)
                specular = Vec3(1.0, 1.0, 1.0) * material.specular * specular_intensity
                final_color = emissive + diffuse + specular
                return final_color
            else:
                n = closest_hit.normal
                return Vec3(n.x + 1, n.y + 1, n.z + 1) * 0.5

        unit_direction = ray.direction.normalize()
        a = 0.5 * (unit_direction.y + 1.0)
        return Vec3(1.0, 1.0, 1.0) * (1.0 - a) + Vec3(0.5, 0.7, 1.0) * a
