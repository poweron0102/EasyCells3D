import pygame as pg
import numpy as np

from .Geometry import Vec3

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
        self.texture = None  # Será a própria instância se a textura existir, para ser usada como chave.
        self.surface = None  # Superfície do Pygame para a renderização da CPU.
        self.data = None     # Array Numpy para a renderização da GPU.

        if texture_path:
            full_texture_path = "Assets/" + texture_path
            try:
                # Carrega a imagem para a CPU e GPU
                self.surface = pg.image.load(full_texture_path).convert_alpha()

                # Prepara o array numpy para a GPU.
                # pg.surfarray.pixels3d retorna (largura, altura, canais)
                # O CudaRenderer espera (altura, largura, canais)
                texture_data_whc = pg.surfarray.pixels3d(self.surface)
                texture_data_hwc = np.transpose(texture_data_whc, (1, 0, 2))
                self.data = np.ascontiguousarray(texture_data_hwc, dtype=np.uint8)
                
                # Define a si mesmo como a chave da textura
                self.texture = self

            except pg.error as e:
                print(f"Erro ao carregar a textura: {full_texture_path} - {e}")

        self.diffuse_color = diffuse_color
        self.specular = specular
        self.shininess = shininess
        self.emissive_color = emissive_color

    def get_color_at(self, u: float, v: float) -> Vec3:
        """
        Obtém a cor da textura nas coordenadas UV para a renderização da CPU.
        Se não houver textura, retorna a cor difusa.
        """
        if self.surface:
            # Converte coordenadas UV (0-1) para coordenadas de pixel da imagem
            tex_x = int(u * (self.surface.get_width() - 1))
            tex_y = int(v * (self.surface.get_height() - 1))
            # get_at do Pygame retorna (r, g, b, a)
            color = self.surface.get_at((tex_x, tex_y))
            return Vec3(color.r / 255.0, color.g / 255.0, color.b / 255.0)
        else:
            return self.diffuse_color
