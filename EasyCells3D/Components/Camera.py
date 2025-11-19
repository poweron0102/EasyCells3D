import math
import os
from pathlib import Path

import numpy as np
import pygame as pg

from EasyCells3D.Components import Component
from EasyCells3D import Game
from EasyCells3D.Geometry import Vec3
import pycuda.driver as cuda
import pycuda.autoinit # importante se não ocorre SIGSEV
from pycuda.compiler import SourceModule

from ..Material import Texture



SphereHittable: type
VoxelsHittable: type


def LoadKernel(file: str, options: list[str] = None) -> SourceModule:
    """
    Carrega e compila um kernel CUDA a partir do caminho 'file'.
    Ele busca o arquivo na pasta `./kernels`.
    Retorna um objeto pycuda.compiler.SourceModule.
    """
    path = Path("kernels/" + file)
    if not path.exists():
        raise FileNotFoundError(f"Kernel file not found: {path}")

    src = path.read_text(encoding='utf-8')

    # opções padrão para compilação podem ser sobrescritas pelo caller
    compile_options = options if options is not None else ['-use_fast_math']

    try:
        module = SourceModule(
            src,
            options=compile_options,
            include_dirs=[os.getcwd() + '/kernels/'],  # Onde procurar os #include "..."
            no_extern_c=True  # Opcional: útil se estiver usando templates C++
        )
        return module
    except Exception as e:
        # inclui o conteúdo do arquivo no erro pode ser verboso; aqui apenas repassa mensagem
        raise RuntimeError(f"Falha ao compilar o kernel CUDA '{path}': {e}")



class Hittable(Component):
    cameras: list['Camera']
    dtype = None

    def __init__(self, cameras: list['Camera'] = None):
        self.cameras: list['Camera'] = []
        if Camera.instance() and cameras is None:
            self.cameras.append(Camera.instance())
        else:
            self.cameras = cameras if cameras is not None else []

    def to_numpy(self):
        pass


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

    def __init__(
            self,
            sky_box: Texture,
            screen: pg.Surface = None,
            vfov: float = 90.0,
            light_direction: Vec3 = Vec3(0.0, 1.0, -1.0),
            ambient_light: Vec3 = Vec3(0.1, 0.1, 0.1),
    ):
        super().__init__()
        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        from EasyCells3D.Components.SphereHittable import SphereHittable as sh
        from EasyCells3D.Components.VoxelsHittable import VoxelsHittable as vh

        global SphereHittable
        global VoxelsHittable
        SphereHittable = sh
        VoxelsHittable = vh

        self._screen = screen

        self.kernel_module = LoadKernel("render_kernel.cu")
        self.kernel = self.kernel_module.get_function("kernel")

        self.sky_box = sky_box
        self.vfov = vfov
        self.center = Vec3(0.0, 0.0, 0.0)
        self.ambient_light = ambient_light
        self.light_direction = light_direction.normalize()

        self.pixel00_loc = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_u = Vec3(0.0, 0.0, 0.0)
        self.pixel_delta_v = Vec3(0.0, 0.0, 0.0)

        self._render_array_np = None
        self._render_array_device = None

    def init(self):
        self._render_array_np = np.ascontiguousarray(pg.surfarray.pixels3d(self.screen))
        self._render_array_device = cuda.mem_alloc_like(self._render_array_np)

    def on_destroy(self):
        if Game.current_instance in Camera.instances and Camera.instances[Game.current_instance] == self:
            Camera.instances.pop(Game.current_instance)
        self.on_destroy = lambda: None

    def update_camera_geometry(self):
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

        if self._render_array_np.shape != (self.image_height, self.image_width, 3):
            self._render_array_np = np.ascontiguousarray(pg.surfarray.pixels3d(self.screen))
            self._render_array_device = cuda.mem_alloc_like(self._render_array_np)

    def loop(self):
        self.update_camera_geometry()
        self.render_cuda()


    def render_cuda(self):
        """
        Renderiza a cena usando o kernel CUDA.
        """
        _, textures_gpu = Texture.get_all_textures_as_numpy_array()

        spheres = [sphere for sphere in SphereHittable.instances if self in sphere.cameras]
        num_spheres = np.int32(len(spheres))
        spheres_np = np.array([sphere.to_numpy() for sphere in spheres], dtype=SphereHittable.dtype)
        spheres_gpu = cuda.to_device(spheres_np)  if len(spheres_np) > 0 else np.uintp(0)

        voxels = [voxel for voxel in VoxelsHittable.instances if self in voxel.cameras]
        num_voxels = np.int32(len(voxels))
        voxels_np = np.array([voxel.to_numpy() for voxel in voxels], dtype=VoxelsHittable.dtype)
        voxels_gpu = cuda.to_device(voxels_np) if len(voxels_np) > 0 else np.uintp(0)

        camera_center = self.center.to_numpy()
        pixel00_loc = self.pixel00_loc.to_numpy()
        pixel_delta_u = self.pixel_delta_u.to_numpy()
        pixel_delta_v = self.pixel_delta_v.to_numpy()
        width, height, _ = self._render_array_np.shape
        sky_box_index = self.sky_box.index
        light_direction = self.light_direction.to_numpy()
        ambient_light = self.ambient_light.to_numpy()

        # 3. Lançar o kernel
        block_size = (16, 16, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )

        self.kernel(
            self._render_array_device,
            np.int32(width),
            np.int32(height),
            camera_center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
            spheres_gpu,
            num_spheres,
            voxels_gpu,
            num_voxels,
            textures_gpu,
            np.int32(sky_box_index),
            light_direction,
            ambient_light,
            block=block_size,
            grid=grid_size
        )

        cuda.memcpy_dtoh(self._render_array_np, self._render_array_device)
        pg.surfarray.blit_array(self.screen, self._render_array_np)
