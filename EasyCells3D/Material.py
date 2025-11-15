import pygame as pg
import numpy as np

from .Components.Camera import Camera
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
        self.texture = None
        self.gpu_data = None # Armazena os dados da textura como um array numpy

        if texture_path:
            full_path = "Assets/" + texture_path
            try:
                self.texture = pg.image.load(full_path).convert_alpha()
                # Prepara os dados para a GPU (Taichi) se a câmera estiver usando CUDA.
                # A transferência real para a GPU agora é gerenciada pela câmera.
                if Camera.instance() and Camera.instance().use_cuda:
                    # Taichi espera a origem no canto inferior esquerdo, então não precisamos inverter verticalmente aqui.
                    # O shader fará a inversão da coordenada V.
                    self.gpu_data = pg.surfarray.pixels3d(self.texture).astype(np.uint8)

            except pg.error as e:
                print(f"Erro ao carregar a textura: {full_path} - {e}")

        self.diffuse_color = diffuse_color
        self.specular = specular
        self.shininess = shininess
        self.emissive_color = emissive_color

    def get_color_at(self, u: float, v: float) -> Vec3:
        """
        Obtém a cor da textura nas coordenadas UV para renderização de CPU.
        Se não houver textura, retorna a cor difusa.
        """
        if self.texture:
            # Converte coordenadas UV (0-1) para coordenadas de pixel da imagem
            tex_x = int(u * (self.texture.get_width() - 1))
            # Inverte v para corresponder ao sistema de coordenadas do Pygame (origem no topo esquerdo)
            tex_y = int((1.0 - v) * (self.texture.get_height() - 1))

            # Garante que as coordenadas estão dentro dos limites
            tex_x = max(0, min(self.texture.get_width() - 1, tex_x))
            tex_y = max(0, min(self.texture.get_height() - 1, tex_y))

            color = self.texture.get_at((tex_x, tex_y))
            return Vec3(color.r / 255.0, color.g / 255.0, color.b / 255.0)
        else:
            return self.diffuse_color
