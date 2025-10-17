import math
import numpy as np
import pygame as pg
from numba import cuda

from .Component import Component, Transform
from .. import Game
from ..Geometry import Vec3, Ray, HitInfo
# Importar o renderizador CUDA
from EasyCells3D import CudaRenderer



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

    def __init__(self, screen: pg.Surface = None, vfov: float = 90.0, use_cuda=True, light_direction: Vec3 = None):
        """
        Cria uma câmara 3D para ray tracing.
        :param screen: A superfície do pygame para renderizar. Se for None, usa o ecrã principal do jogo.
        :param vfov: Campo de visão vertical em graus.
        :param use_cuda: Se verdadeiro, tenta usar a GPU para renderizar.
        :param light_direction: A direção da luz principal na cena.
        """
        super().__init__()
        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        self.hittables: list[Hittable] = []
        self._screen = screen

        # Tenta usar CUDA, se falhar, volta para CPU
        self.use_cuda = use_cuda
        if self.use_cuda:
            try:
                cuda.detect()
                print("Dispositivo CUDA detectado. A renderização será feita na GPU.")
            except cuda.CudaSupportError:
                print("Nenhum dispositivo CUDA encontrado. A renderizar na CPU.")
                self.use_cuda = False

        # Propriedades da Câmara
        self.vfov = vfov
        self.center = Vec3(0.0, 0.0, 0.0)
        if light_direction is None:
            self.light_direction = Vec3(0.0, 1.0, -1.0).normalize()
        else:
            self.light_direction = light_direction.normalize()


        # Valores calculados
        self.pixel00_loc = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_u = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_v = Vec3(0.0, 0.0, 0.0)

        # Array para a imagem renderizada (usado por CPU e CUDA)
        self._render_array_np = None
        # Array de dispositivo para a imagem renderizada (para CUDA)
        self._render_array_device = None


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

        # Inicializa o array de renderização se o tamanho do ecrã mudou
        if self._render_array_np is None or self._render_array_np.shape[1] != self.image_width or \
                self._render_array_np.shape[0] != self.image_height:
            self._render_array_np = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            self._render_array_device = None # Força a recriação do array de dispositivo

    def loop(self):
        """O loop principal de renderização. Lança raios para cada píxel."""
        self._update_camera_geometry()

        if self.use_cuda:
            self._render_cuda()
        else:
            self._render_cpu()

        # Desenha o array numpy no ecrã do pygame
        pg.surfarray.blit_array(self.screen, np.transpose(self._render_array_np, (1, 0, 2)))

    def _render_cuda(self):
        """Renderiza a cena usando o kernel CUDA."""
        # 1. Preparar os dados para a GPU

        # Cria ou reutiliza o array de pixels na memória da GPU
        if self._render_array_device is None:
            self._render_array_device = cuda.to_device(self._render_array_np)

        # Prepara os dados de objetos e materiais para a GPU
        spheres_np, materials_np, textures_np = self.make_np_objects_and_materials(self.hittables)

        if spheres_np is None or spheres_np.size == 0:
            # Se não há esferas, não há nada para renderizar na GPU.
            # Podemos limpar o array de renderização e retornar.
            self._render_array_np.fill(0) # Ou preencher com a cor de fundo
            return

        # Copia os dados das esferas para a GPU
        spheres_device = cuda.to_device(spheres_np)
        materials_device = cuda.to_device(materials_np)

        # Numba não suporta arrays de arrays de forma direta e eficiente para texturas.
        # Uma abordagem comum é criar um "texture atlas" ou, para este caso,
        # vamos passar a primeira textura encontrada como exemplo.
        # Uma implementação mais robusta lidaria com múltiplas texturas de forma mais genérica.
        textures_device = cuda.to_device(textures_np)
        
        # Converte vetores da câmara para arrays numpy
        camera_center_np = self.center.to_numpy(dtype=np.float32)
        pixel00_loc_np = self.pixel00_loc.to_numpy(dtype=np.float32)
        pixel_delta_u_np = self.pixel_delta_u.to_numpy(dtype=np.float32)
        pixel_delta_v_np = self.pixel_delta_v.to_numpy(dtype=np.float32)
        light_direction_np = self.light_direction.to_numpy(dtype=np.float32)

        # 2. Configurar a execução do Kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(self.image_width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(self.image_height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # 3. Executar o Kernel
        CudaRenderer.render_kernel[blocks_per_grid, threads_per_block](
            self._render_array_device,
            camera_center_np,
            pixel00_loc_np,
            pixel_delta_u_np,
            pixel_delta_v_np,
            spheres_device,
            materials_device,
            textures_device,
            light_direction_np,
        )
        # 4. Copiar o resultado de volta para a CPU para exibição
        self._render_array_device.copy_to_host(self._render_array_np)

    @staticmethod
    def make_np_objects_and_materials(hittables) -> 'tuple[np.ndarray, np.ndarray, list[cuda.device_array]]':
        from .SphereHittable import SphereHittable
        spheres = [h for h in hittables if isinstance(h, SphereHittable)]

        if not spheres:
            return None, None, None

        # Estrutura para os dados da esfera na GPU
        sphere_dtype = np.dtype([
            ('center', np.float32, 3),
            ('radius', np.float32),
            ('rotation', np.float32, 4),  # Quaternion (w, x, y, z)
            ('material_index', np.int32)
        ])

        # Estrutura para os dados do material na GPU
        material_dtype = np.dtype([
            ('diffuse_color', np.float32, 3),
            ('specular', np.float32),
            ('shininess', np.float32),
            ('emissive_color', np.float32, 3),
            ('texture_index', np.int32) # -1 se não houver textura
        ])

        spheres_np = np.empty(len(spheres), dtype=sphere_dtype)
        materials_np = np.empty(len(spheres), dtype=material_dtype)
        textures_np = []

        for i, sphere in enumerate(spheres):
            spheres_np[i]['center'] = sphere.transform.position.to_numpy(dtype=np.float32)
            spheres_np[i]['radius'] = sphere.radius
            spheres_np[i]['rotation'] = sphere.transform.rotation.to_numpy(dtype=np.float32) # (w,x,y,z)
            spheres_np[i]['material_index'] = i

            mat = sphere.material
            materials_np[i]['diffuse_color'] = mat.diffuse_color.to_numpy(dtype=np.float32)
            materials_np[i]['specular'] = mat.specular
            materials_np[i]['shininess'] = mat.shininess
            materials_np[i]['emissive_color'] = mat.emissive_color.to_numpy(dtype=np.float32)

            if mat.gpu_data is None:
                materials_np[i]['texture_index'] = -1
            else:
                materials_np[i]['texture_index'] = len(textures_np)
                textures_np.append(mat.gpu_data)


        return spheres_np, materials_np, textures_np

    def _render_cpu(self):
        """Renderiza a cena usando a CPU (código original)."""
        render_array = np.transpose(self._render_array_np, (1, 0, 2))  # surfarray usa (width, height)
        for j in range(self.image_height):
            for i in range(self.image_width):
                ray = self._get_ray(i, j)
                color_vec = self._ray_color(ray)

                r = int(255.999 * np.clip(color_vec.x, 0, 1))
                g = int(255.999 * np.clip(color_vec.y, 0, 1))
                b = int(255.999 * np.clip(color_vec.z, 0, 1))

                render_array[i, j] = (r, g, b)

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
            # Verifica se o objeto atingido tem um material e coordenadas UV
            if hasattr(closest_hittable, 'material') and closest_hit.uv is not None:
                material = closest_hittable.material

                # Cor base da textura ou difusa
                albedo = material.get_color_at(closest_hit.uv.x, closest_hit.uv.y)

                # Componente emissiva
                emissive = material.emissive_color

                # Componente difusa
                light_dir = self.light_direction
                diffuse_intensity = max(0.0, closest_hit.normal.dot(light_dir))
                diffuse = albedo * diffuse_intensity

                # Componente especular (Blinn-Phong)
                view_dir = (ray.origin - closest_hit.point).normalize()
                half_vector = (light_dir + view_dir).normalize()
                specular_intensity = pow(max(0.0, closest_hit.normal.dot(half_vector)), material.shininess)
                specular = Vec3(1.0, 1.0, 1.0) * material.specular * specular_intensity

                # Combina as componentes
                final_color = emissive + diffuse + specular
                return final_color
            else:
                # Fallback: colore com base na normal da superfície
                n = closest_hit.normal
                return Vec3(n.x + 1, n.y + 1, n.z + 1) * 0.5

        # Cor de fundo (gradiente do céu)
        unit_direction = ray.direction.normalize()
        a = 0.5 * (unit_direction.y + 1.0)
        return Vec3(1.0, 1.0, 1.0) * (1.0 - a) + Vec3(0.5, 0.7, 1.0) * a
