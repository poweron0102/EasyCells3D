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
    LoadedTextures: dict[str, tuple[int, DeviceAllocation, int, int, int]] = {}
    TextureList: list[str] = []

    dtype = texture_dtype

    def __init__(self, texture_path: str):

        self.texture_path = texture_path

        if texture_path in Texture.LoadedTextures:
            cont, self.data, self.width, self.height, self.index = Texture.LoadedTextures[texture_path]
            Texture.LoadedTextures[texture_path] = (cont + 1, self.data, self.width, self.height, self.index)
        else:
            full_texture_path = "Assets/" + texture_path
            try:
                # Carrega a imagem para a CPU e GPU
                surface = pg.image.load(full_texture_path).convert_alpha()

                texture_data = pg.surfarray.pixels3d(surface)
                self.width, self.height, _ = texture_data.shape
                self.data = cuda.to_device(texture_data)

                self.index = len(Texture.TextureList)
                Texture.TextureList.append(texture_path)
                Texture.LoadedTextures[texture_path] = (1, self.data, self.width, self.height, self.index)

            except pg.error as e:
                print(f"Erro ao carregar a textura: {full_texture_path} - {e}")
                self.data = None
                self.width = 0
                self.height = 0
                self.index = -1 # Indica uma textura inválida

    def __del__(self):
        if self.texture_path in Texture.LoadedTextures:
            cont, data, width, height, index = Texture.LoadedTextures[self.texture_path]
            if cont > 1:
                Texture.LoadedTextures[self.texture_path] = (cont - 1, data, width, height, index)
            else:
                # A remoção da lista e a reorganização dos índices é complexa
                # e pode não ser necessária se as texturas persistirem durante a vida útil do aplicativo.
                # Por simplicidade, apenas removemos do dicionário.
                # todo
                Texture.LoadedTextures.pop(self.texture_path)

    def to_numpy(self) -> np.ndarray:
        """Converte os dados da textura para um array estruturado numpy."""
        return np.array([(self.data, self.width, self.height)], dtype=texture_dtype)

    @staticmethod
    def get_all_textures_as_numpy_array() -> np.ndarray:
        """Retorna um array numpy de todas as texturas carregadas."""
        all_textures = np.empty(len(Texture.TextureList), dtype=texture_dtype)
        for path in Texture.TextureList:
            _, data, width, height, index = Texture.LoadedTextures[path]
            all_textures[index] = (data, width, height)
        return all_textures


material_dtype = np.dtype([
    ("texture_index", np.uint32),
    ("diffuse_color", Vec3.dtype),
    ("specular", np.float32),
    ("shininess", np.float32),
    ("emissive_color", Vec3.dtype)
])


class Material:
    """
    Representa as propriedades do material de um objeto.
    """

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
        self.texture = Texture(texture_path)

        self.diffuse_color = diffuse_color
        self.specular = specular
        self.shininess = shininess
        self.emissive_color = emissive_color

    def to_numpy(self) -> np.ndarray:
        """Converte as propriedades do material para um array estruturado numpy."""
        texture_index = self.texture.index if self.texture and self.texture.data is not None else -1
        return np.array([(texture_index,
                          self.diffuse_color.to_tuple(),
                          self.specular,
                          self.shininess,
                          self.emissive_color.to_tuple())], dtype=material_dtype)
