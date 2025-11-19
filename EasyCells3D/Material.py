import pygame as pg
import numpy as np
import pycuda.driver as cuda

from .Geometry import Vec3, DeviceAllocation

texture_dtype = np.dtype([
    ("data_ptr", np.uintp),
    ("width", np.uint32),
    ("height", np.uint32)
])


class Texture:
    _loaded_textures: dict[str, 'Texture'] = {}
    _texture_list: list['Texture'] = []
    _all_textures_np: np.ndarray | None = None
    _all_textures_gpu: DeviceAllocation | None = None
    _is_np_array_valid: bool = False

    dtype = texture_dtype

    # O construtor não deve ser chamado diretamente, use Texture.get()
    def __init__(self, texture_path: str, _private_constructor_token=None):
        if _private_constructor_token is None:
            raise RuntimeError("Use Texture.get(texture_path) para criar ou obter uma textura.")
        
        self.texture_path = texture_path
        full_texture_path = "Assets/" + texture_path
        try:
            # Carrega a imagem para a CPU e GPU
            surface = pg.image.load(full_texture_path).convert_alpha()

            texture_data = pg.surfarray.pixels3d(surface)
            self.width, self.height, _ = texture_data.shape
            self.data = cuda.to_device(texture_data)

            self.index = len(Texture._texture_list)
            Texture._texture_list.append(self)
            Texture._is_np_array_valid = False  # Invalida o cache

        except pg.error as e:
            print(f"Erro ao carregar a textura: {full_texture_path} - {e}")
            self.data = None
            self.width = 0
            self.height = 0
            self.index = -1  # Indica uma textura inválida

    @staticmethod
    def get(texture_path: str) -> 'Texture':
        """
        Obtém uma instância de textura. Se a textura já foi carregada,
        retorna a instância existente. Caso contrário, cria uma nova.
        """
        if texture_path in Texture._loaded_textures:
            return Texture._loaded_textures[texture_path]
        else:
            new_texture = Texture(texture_path, _private_constructor_token=object())
            Texture._loaded_textures[texture_path] = new_texture
            return new_texture

    def __del__(self):
        # Chamado quando a contagem de referências do objeto chega a 0
        if self.texture_path in Texture._loaded_textures:
            Texture._loaded_textures.pop(self.texture_path, None)
            if self in Texture._texture_list:
                # Atualiza os índices das texturas subsequentes
                removed_index = self.index
                Texture._texture_list.pop(removed_index)
                for i in range(removed_index, len(Texture._texture_list)):
                    Texture._texture_list[i].index = i

            if self.data:
                self.data.free()

            Texture._is_np_array_valid = False  # Invalida o cache

    def to_numpy(self) -> np.ndarray:
        """Converte os dados da textura para um array estruturado numpy."""
        return np.array([(self.data, self.width, self.height)], dtype=texture_dtype)

    @staticmethod
    def get_all_textures_as_numpy_array() -> tuple[np.ndarray, DeviceAllocation]:
        """
        Retorna um array numpy de todas as texturas carregadas.
        E um ponteiro delas na GPU.
        """
        if Texture._is_np_array_valid and Texture._all_textures_np is not None:
            return Texture._all_textures_np, Texture._all_textures_gpu

        all_textures = np.empty(len(Texture._texture_list), dtype=texture_dtype)
        for texture_instance in Texture._texture_list:
            all_textures[texture_instance.index] = texture_instance.to_numpy()
        
        Texture._all_textures_np = all_textures
        Texture._all_textures_gpu = cuda.to_device(all_textures)
        Texture._is_np_array_valid = True
        
        return Texture._all_textures_np, Texture._all_textures_gpu


material_dtype = np.dtype([
    ("texture_index", np.int32),
    ("diffuse_color", Vec3.dtype),
    ("specular", np.float32),
    ("shininess", np.float32),
    ("emissive_color", Vec3.dtype)
])


class Material:
    """
    Representa as propriedades do material de um objeto.
    """
    
    dtype = material_dtype

    def __init__(self,
                 texture_path: str = None,
                 diffuse_color: Vec3 = Vec3(1.0, 1.0, 1.0),
                 specular: float = 0.5,
                 shininess: float = 32.0,
                 emissive_color: Vec3 = Vec3(0.0, 0.0, 0.0)
        ):
        """
        Representa as propriedades do material de um objeto.
        :param texture_path: Caminho para o arquivo de textura.
        :param diffuse_color: A cor base do material (usada se não houver textura).
        :param specular: A intensidade do brilho especular.
        :param shininess: O quão concentrado é o brilho especular.
        :param emissive_color: A cor que o material emite (luz própria).
        """
        self.texture = Texture.get(texture_path) if texture_path else None

        self.diffuse_color = diffuse_color
        self.specular = specular
        self.shininess = shininess
        self.emissive_color = emissive_color

    def to_numpy(self) -> np.ndarray:
        """Converte as propriedades do material para um array estruturado numpy."""
        texture_index = self.texture.index if self.texture and self.texture.data is not None else -1
        return np.array(
            (texture_index,
                self.diffuse_color.to_tuple,
                self.specular,
                self.shininess,
                self.emissive_color.to_tuple),
            dtype=material_dtype)
