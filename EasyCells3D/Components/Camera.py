import math
import numpy as np
import pygame as pg
import taichi as ti

from .Component import Component
from .. import Game
from ..Geometry import Vec3, Ray, HitInfo
# Importar o renderizador Taichi
from EasyCells3D import CudaRenderer

# --- Taichi Struct Definitions ---
Sphere = ti.types.struct(
    center=ti.types.vector(3, ti.f32),
    radius=ti.f32,
    rotation=ti.types.vector(4, ti.f32),  # Quaternion (w, x, y, z)
    material_index=ti.i32
)

Material = ti.types.struct(
    diffuse_color=ti.types.vector(3, ti.f32),
    specular=ti.f32,
    shininess=ti.f32,
    emissive_color=ti.types.vector(3, ti.f32),
    texture_index=ti.i32  # -1 se não houver textura
)


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

    def __init__(self, screen: pg.Surface = None, vfov: float = 90.0, use_cuda=True, light_direction: Vec3 = None, ambient_light: Vec3 = Vec3(0.1, 0.1, 0.1)):
        """
        Cria uma câmara 3D para ray tracing.
        :param screen: A superfície do pygame para renderizar. Se for None, usa o ecrã principal do jogo.
        :param vfov: Campo de visão vertical em graus.
        :param use_cuda: Se verdadeiro, tenta usar a GPU (Taichi) para renderizar.
        :param light_direction: A direção da luz principal na cena.
        """
        super().__init__()
        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        self.hittables: list[Hittable] = []
        self._screen = screen

        # Com Taichi, a inicialização é feita no módulo do renderizador.
        self.use_cuda = use_cuda
        if self.use_cuda:
            print("Taichi backend selecionado. A renderização será feita na GPU se disponível.")
        else:
            print("A renderizar na CPU.")

        # Propriedades da Câmara
        self.vfov = vfov
        self.center = Vec3(0.0, 0.0, 0.0)
        self.ambient_light = ambient_light
        if light_direction is None:
            self.light_direction = Vec3(0.0, 1.0, -1.0).normalize()
        else:
            self.light_direction = light_direction.normalize()

        # Valores calculados
        self.pixel00_loc = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_u = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_v = Vec3(0.0, 0.0, 0.0)

        # Arrays e campos Taichi
        self._render_array_np = None
        self._pixels_field = None
        self._spheres_field = None
        self._materials_field = None
        self._texture_field = None
        self._has_texture = False


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

        if self._render_array_np is None or self._render_array_np.shape[0] != self.image_width or \
                self._render_array_np.shape[1] != self.image_height:
            self._render_array_np = np.zeros((self.image_width, self.image_height, 3), dtype=np.uint8)
            self._pixels_field = None
            self._spheres_field = None
            self._materials_field = None
            self._texture_field = None

    def loop(self):
        """O loop principal de renderização. Lança raios para cada píxel."""
        self._update_camera_geometry()

        if self.use_cuda:
            self._render_taichi()
        else:
            self._render_cpu()

        pg.surfarray.blit_array(self.screen, self._render_array_np)

    def _prepare_taichi_scene(self):
        """Prepara os dados da cena em campos Taichi para a GPU."""
        from .SphereHittable import SphereHittable
        spheres = [h for h in self.hittables if isinstance(h, SphereHittable) and h.enable]

        if not spheres:
            return False

        if self._spheres_field is None or self._spheres_field.shape[0] != len(spheres):
            self._spheres_field = Sphere.field(shape=(len(spheres),))
            self._materials_field = Material.field(shape=(len(spheres),))

        first_texture_data = None
        for sphere in spheres:
            if hasattr(sphere, 'material') and sphere.material.gpu_data is not None:
                first_texture_data = sphere.material.gpu_data
                break
        
        self._has_texture = first_texture_data is not None
        if self._has_texture:
            if self._texture_field is None or \
               self._texture_field.shape[0] != first_texture_data.shape[0] or \
               self._texture_field.shape[1] != first_texture_data.shape[1]:
                self._texture_field = ti.Vector.field(3, dtype=ti.u8, shape=first_texture_data.shape[:2])
            self._texture_field.from_numpy(first_texture_data)
        else:
            if self._texture_field is None:
                 self._texture_field = ti.Vector.field(3, dtype=ti.u8, shape=(1, 1))

        for i, sphere in enumerate(spheres):
            self._spheres_field[i].center = sphere.word_position.position.to_numpy(np.float32)
            self._spheres_field[i].radius = sphere.radius * sphere.word_position.scale.x
            self._spheres_field[i].rotation = sphere.word_position.rotation.to_numpy(np.float32)
            self._spheres_field[i].material_index = i

            mat = sphere.material
            self._materials_field[i].diffuse_color = mat.diffuse_color.to_numpy(np.float32)
            self._materials_field[i].specular = mat.specular
            self._materials_field[i].shininess = mat.shininess
            self._materials_field[i].emissive_color = mat.emissive_color.to_numpy(np.float32)
            
            if self._has_texture and mat.gpu_data is not None:
                self._materials_field[i].texture_index = 0
            else:
                self._materials_field[i].texture_index = -1
        
        return True

    def _render_taichi(self):
        """Renderiza a cena usando o kernel Taichi."""
        scene_ready = self._prepare_taichi_scene()

        if not scene_ready:
            self._render_array_np.fill(0)
            return

        if self._pixels_field is None:
            self._pixels_field = ti.Vector.field(3, dtype=ti.u8, shape=(self.image_width, self.image_height))

        CudaRenderer.render_kernel(
            self._pixels_field,
            self.center.to_numpy(np.float32),
            self.pixel00_loc.to_numpy(np.float32),
            self.pixel_delta_u.to_numpy(np.float32),
            self.pixel_delta_v.to_numpy(np.float32),
            self._spheres_field,
            self._materials_field,
            self._texture_field,
            self.light_direction.to_numpy(np.float32),
            self.ambient_light.to_numpy(np.float32)
        )

        self._pixels_field.to_numpy(self._render_array_np)

    def _render_cpu(self):
        """Renderiza a cena usando a CPU (código original)."""
        render_array_transposed = np.transpose(self._render_array_np, (1, 0, 2))
        for j in range(self.image_height):
            for i in range(self.image_width):
                ray = self._get_ray(i, j)
                color_vec = self._ray_color(ray)

                r = int(255.999 * np.clip(color_vec.x, 0, 1))
                g = int(255.999 * np.clip(color_vec.y, 0, 1))
                b = int(255.999 * np.clip(color_vec.z, 0, 1))

                render_array_transposed[j, i] = (r, g, b)
        self._render_array_np[:, :, :] = np.transpose(render_array_transposed, (1, 0, 2))


    def _get_ray(self, i: int, j: int) -> Ray:
        """Obtém um raio da câmara para um pixel específico."""
        pixel_center = self.pixel00_loc + (self.pixel_delta_u * i) + (self.pixel_delta_v * j)
        ray_direction = (pixel_center - self.center).normalize()
        return Ray(self.center, ray_direction)

    def _ray_color(self, ray: Ray) -> Vec3[float]:
        """Calcula a cor de um raio."""
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
            if hasattr(closest_hittable, 'material'):
                material = closest_hittable.material
                
                albedo = material.get_color_at(closest_hit.uv.x, closest_hit.uv.y) if closest_hit.uv else material.diffuse_color

                ambient = albedo * self.ambient_light
                emissive = material.emissive_color
                light_dir = self.light_direction
                diffuse_intensity = max(0.0, closest_hit.normal.dot(light_dir))
                diffuse = albedo * diffuse_intensity

                view_dir = (ray.origin - closest_hit.point).normalize()
                half_vector = (light_dir + view_dir).normalize()
                specular_intensity = pow(max(0.0, closest_hit.normal.dot(half_vector)), material.shininess)
                specular = Vec3(1.0, 1.0, 1.0) * material.specular * specular_intensity

                final_color = emissive + ambient + diffuse + specular
                return final_color
            else:
                n = closest_hit.normal
                return Vec3(n.x + 1, n.y + 1, n.z + 1) * 0.5

        unit_direction = ray.direction.normalize()
        a = 0.5 * (unit_direction.y + 1.0)
        return Vec3(1.0, 1.0, 1.0) * (1.0 - a) + Vec3(0.5, 0.7, 1.0) * a
