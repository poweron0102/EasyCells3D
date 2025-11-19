import numpy as np
import midvoxio.voxio as voxio
import pycuda.driver as cuda
from midvoxio.vox import Vox

from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material

voxels_dtype = np.dtype([
    ("position", Vec3.dtype),
    ("rotation", Quaternion.dtype),
    ("scale", Vec3.dtype),
    ("voxels_ptr", np.uintp),
    ("voxels_size", (np.uint32, 3)),
    ("materials_ptr", np.uintp),
    ("num_materials", np.uint32)
])


class VoxelsHittable(Hittable):
    word_position: Transform
    dtype = voxels_dtype
    instances: list['VoxelsHittable'] = []

    def __init__(self, vox_file_name: str):
        super().__init__()
        VoxelsHittable.instances.append(self)
        
        # Carrega o arquivo .vox
        full_vox_path = "Assets/" + vox_file_name
        try:
            vox_model: Vox = voxio.Vox.from_file(full_vox_path)
        except FileNotFoundError:
            print(f"Erro: Arquivo .vox não encontrado em: {full_vox_path}")
            self.voxels_data = None
            self.materials = []
            self.voxels_gpu = None
            return
        
        # Converte a paleta de cores para materiais
        self.materials: list[Material] = []
        for color in vox_model.palette:
            r, g, b, _ = color
            diffuse_color = Vec3(r / 255.0, g / 255.0, b / 255.0)
            self.materials.append(Material(diffuse_color=diffuse_color))
        
        # Prepara os dados dos voxels com índices de material
        # O índice 0 no arquivo .vox significa vazio, então o ajustamos para -1
        self.voxels_data = (vox_model.models[0] - 1).astype(np.int32)
        
        # Aloca memória na GPU
        self.voxels_gpu = cuda.to_device(self.voxels_data)
        
        materials_np = np.array([mat.to_numpy() for mat in self.materials], dtype=Material.dtype)
        self.materials_gpu = cuda.to_device(materials_np)
        self._is_gpu_voxels_valid = True

    def init(self):
        super().init()
        self.word_position = self.transform

    def loop(self):
        self.word_position = Transform.Global

    def on_destroy(self):
        if self in VoxelsHittable.instances:
            VoxelsHittable.instances.remove(self)
        
        if hasattr(self, 'voxels_gpu') and self.voxels_gpu:
            self.voxels_gpu.free()
        if hasattr(self, 'materials_gpu') and self.materials_gpu:
            self.materials_gpu.free()
            
        self.on_destroy = lambda: None # Evita chamadas repetidas

    def get_voxel(self, x: int, y: int, z: int) -> int:
        """Retorna o índice do material para um voxel nas coordenadas (x, y, z)."""
        if self.voxels_data is None:
            return -1
        return self.voxels_data[x, y, z]

    def set_voxel(self, x: int, y: int, z: int, material_index: int):
        """
        Define o índice do material para um voxel nas coordenadas (x, y, z).
        Isso invalida os dados na GPU, que serão atualizados no próximo ciclo.
        """
        if self.voxels_data is None:
            return
        try:
            self.voxels_data[x, y, z] = material_index
            self._is_gpu_voxels_valid = False  # Marca os dados da GPU como inválidos
        except IndexError:
            print(f"Erro: Coordenadas ({x}, {y}, {z}) fora dos limites para set_voxel.")

    def to_numpy(self):
        if not self._is_gpu_voxels_valid:
            # Libera a memória antiga antes de alocar a nova, se existir
            if self.voxels_gpu:
                self.voxels_gpu.free()
            # Reenvia os dados atualizados dos voxels para a GPU
            self.voxels_gpu = cuda.to_device(self.voxels_data)
            self._is_gpu_voxels_valid = True

        return np.array([(
            self.word_position.position.to_numpy(),
            self.word_position.rotation.to_numpy(),
            self.word_position.scale.to_numpy(),
            self.voxels_gpu.ptr,
            self.voxels_data.shape,
            self.materials_gpu.ptr,
            len(self.materials)
        )], dtype=voxels_dtype)