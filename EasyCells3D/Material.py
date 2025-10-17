import pygame as pg
from numba import cuda
import numpy as np
from .Geometry import Vec3
from .Game import Game

class Material:
    def __init__(self, texture_path: str = None, diffuse_color: Vec3 = Vec3(1.0, 1.0, 1.0), specular: float = 0.5, shininess: float = 32.0, emissive_color: Vec3 = Vec3(0.0, 0.0, 0.0)):
        """
        Representa as propriedades do material de um objeto.
        :param texture_path: Caminho para o arquivo de textura.
        :param diffuse_color: A cor base do material (usada se não houver textura).
        :param specular: A intensidade do brilho especular.
        :param shininess: O quão concentrado é o brilho especular.
        :param emissive_color: A cor que o material emite (luz própria).
        """
        self.texture = None
        self.gpu_data = None
        self.use_gpu = Game.instance().use_gpu if Game.instance() else False

        if texture_path:
            texture_path = "Assets/" + texture_path
            try:
                self.texture = pg.image.load(texture_path)
                if self.use_gpu:
                    texture_surface = pg.transform.flip(self.texture, False, True)
                    texture_data = pg.surfarray.pixels3d(texture_surface)
                    self.gpu_data = cuda.to_device(np.ascontiguousarray(texture_data))
            except pg.error as e:
                print(f"Erro ao carregar a textura: {texture_path} - {e}")

        self.diffuse_color = diffuse_color
        self.specular = specular
        self.shininess = shininess
        self.emissive_color = emissive_color


    def get_color_at(self, u: float, v: float) -> Vec3:
        """
        Obtém a cor da textura nas coordenadas UV.
        Se não houver textura, retorna a cor difusa.
        """
        if self.texture:
            # Converte coordenadas UV (0-1) para coordenadas de pixel da imagem
            tex_x = int(u * (self.texture.get_width() - 1))
            tex_y = int(v * (self.texture.get_height() - 1))
            # Pygame's get_at returns (r, g, b, a)
            color = self.texture.get_at((tex_x, tex_y))
            return Vec3(color.r / 255.0, color.g / 255.0, color.b / 255.0)
        else:
            return self.diffuse_color

