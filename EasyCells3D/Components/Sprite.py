import math

import raylibpy as rl

from .Camera2D import Renderable2D
from .Component import Transform


class Sprite(Renderable2D):
    texture: rl.Texture2D
    index_x: int = 0
    index_y: int = 0
    size: tuple[int, int] = (0, 0)

    def __init__(self, image_path: str | rl.Texture2D, size: tuple[int, int] = None):
        super().__init__()
        if isinstance(image_path, rl.Texture2D):
            self.texture = image_path
        else:
            self.texture = rl.load_texture(f"Assets/{image_path}")

        self.size = size if size else (self.texture.width, self.texture.height)

        self.horizontal_flip = False
        self.vertical_flip = False


    def render(self):
        if not self.texture.id:
            return

        transform = self.global_transform
        
        # Define a área da textura a ser desenhada (Source Rectangle)
        # Lida com animação (index_x) e espelhamento (flip)
        src_width = self.size[0]
        src_height = self.size[1]
        
        # Em Raylib, largura/altura negativa no source_rec inverte a imagem
        if self.horizontal_flip:
            src_width *= -1
        if self.vertical_flip:
            src_height *= -1

        source_rec = rl.Rectangle(
            self.index_x * abs(src_width), self.index_y * abs(src_height),
            src_width, src_height
        )

        # Define onde desenhar no mundo (Destination Rectangle)
        dest_width = abs(self.size[0]) * transform.scale.x
        dest_height = abs(self.size[1]) * transform.scale.y
        
        dest_rec = rl.Rectangle(
            transform.position.x, transform.position.y,
            dest_width, dest_height
        )

        # Define o pivô de rotação (Centro do sprite)
        origin = rl.Vector2(dest_width / 2, dest_height / 2)

        rl.draw_texture_pro(self.texture, source_rec, dest_rec, origin, transform.angle, rl.WHITE)
