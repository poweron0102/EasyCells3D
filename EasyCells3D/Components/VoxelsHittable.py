import numpy as np
from midvoxio.voxio import vox_to_arr
import pycuda.driver as cuda

from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material

voxels_dtype = np.dtype([
    ("voxels_ptr", np.uintp),       # 8 bytes
    ("materials_ptr", np.uintp),    # 8 bytes
    ("position", Vec3.dtype),       # 12 bytes
    ("rotation", Quaternion.dtype), # 16 bytes
    ("scale", Vec3.dtype),          # 12 bytes
    ("voxels_size", (np.uint32, 3)),# 12 bytes
    ("num_materials", np.uint32)    # 4 bytes
])


class VoxelsHittable(Hittable):
    word_position: Transform
    dtype = voxels_dtype
    instances: list['VoxelsHittable'] = []

    def __init__(self, vox_file_name: str, emissive_amount = 0.6):
        super().__init__()
        VoxelsHittable.instances.append(self)
        
        # Carrega o arquivo .vox
        full_vox_path = "Assets/" + vox_file_name
        try:
            # vox_to_arr retorna um array (x, y, z, 4) com cores RGBA normalizadas (0-1)
            voxels_rgba = vox_to_arr(full_vox_path)
            voxels_rgba = np.transpose(voxels_rgba, (0, 2, 1, 3))[:, ::-1, :, :]
        except FileNotFoundError:
            print(f"Erro: Arquivo .vox não encontrado em: {full_vox_path}")
            self.voxels_data = None
            self.materials = []
            self.voxels_gpu = None
            return
        
        # Extrai a paleta de cores únicas a partir dos dados dos voxels (ignorando voxels vazios)
        # e converte para materiais. As cores em voxels_rgba já estão normalizadas (0-1).
        all_colors = voxels_rgba.reshape(-1, 4)
        non_empty_mask = all_colors[:, 3] > 0
        palette_normalized = np.unique(all_colors[non_empty_mask], axis=0)
        
        self.materials: list[Material] = []
        for r, g, b, a in palette_normalized:
            diffuse_color = Vec3(r, g, b)
            self.materials.append(Material(diffuse_color=diffuse_color, emissive_color=diffuse_color * emissive_amount, specular=0.01))
        
        # Cria uma matriz interna para os dados dos voxels
        voxels_data_inner = np.full(voxels_rgba.shape[:3], -1, dtype=np.int32)

        # Para cada cor única na paleta, encontra todos os voxels que têm essa cor e atribui o índice da paleta.
        for material_index, color in enumerate(palette_normalized):
            matching_voxels = np.all(voxels_rgba == color, axis=3)
            voxels_data_inner[matching_voxels] = material_index

        # Adiciona uma camada de padding com -1 em volta da matriz de voxels.
        # Isso simplifica o acesso na GPU, evitando a necessidade de verificações de limites.
        self.voxels_data = np.pad(voxels_data_inner, pad_width=1, mode='constant', constant_values=-1)


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

        # print(f"Voxels_gpu: {self.voxels_gpu}, Materials_gpu: {self.materials_gpu}")
        # print(f"Voxels_gpu: {int(self.voxels_gpu)}, Materials_gpu: {int(self.materials_gpu)}")
        return np.array((
            int(self.voxels_gpu),  # voxels_ptr
            int(self.materials_gpu),  # materials_ptr
            self.word_position.position.to_numpy(),  # position
            self.word_position.rotation.to_numpy(),  # rotation
            self.word_position.scale.to_numpy(),  # scale
            self.voxels_data.shape,  # voxels_size
            len(self.materials)  # num_materials
        ), dtype=voxels_dtype)