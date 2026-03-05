import json
from typing import Callable

import pyray as rl

from .Camera2D import Renderable2D
from .Component import Component


def matrix_from_csv(file_path: str) -> list[list[int]]:
    matrix = []
    with open(f"Assets/{file_path}", 'r') as file:
        for line in file:
            row = [int(value.strip()) for value in line.split(',')]
            matrix.append(row)
    return matrix

def solids_set_from_tsj(file_path: str) -> set[int]:
    solids_set = set()
    with open(f"Assets/{file_path}", 'r') as file:
        json_str = file.read()
        data = json.loads(json_str)

        tiles = data["tiles"]
        for tile in tiles:
            if tile["type"] == "Collider":
                solids_set.add(tile["id"])
    return solids_set


class TileMap(Component):

    def __init__(self, matrix: list[list[int]]):
        self.matrix = matrix
        self.size = (len(matrix[0]), len(matrix))

        self.on_tile_change: list[Callable[[int, int, int], None]] = []

    def get_tile(self, x: int, y: int) -> int:
        return self.matrix[y][x]

    def set_tile(self, x: int, y: int, value: int):
        self.matrix[y][x] = value
        for callback in self.on_tile_change:
            callback(x, y, value)


class TileMapRenderer(Renderable2D):
    tile_map: TileMap
    tile_set: rl.Texture
    render_texture: rl.RenderTexture = None
    dirty: bool = True

    def __init__(self, tile_set, tile_size: int):
        """
        tile_set = str | rl.Texture2D
        """
        super().__init__()
        if isinstance(tile_set, str):
            self.tile_set = rl.load_texture(f"Assets/{tile_set}")
        else:
            self.tile_set = tile_set
            
        self.tile_size = tile_size
        
        # Calcula quantos tiles cabem no tileset (colunas, linhas)
        self.matrix_size = (self.tile_set.width // tile_size, self.tile_set.height // tile_size)

    def init(self):
        super().init()
        self.tile_map = self.GetComponent(TileMap)
        if self.tile_map:
            self.tile_map.on_tile_change.append(self._on_tile_change)
            self.dirty = True

    def _on_tile_change(self, x, y, val):
        self.dirty = True

    def int2coord(self, value: int) -> tuple[int, int]:
        return value % self.matrix_size[0], value // self.matrix_size[0]

    def update_cache(self):
        """Redesenha o mapa inteiro para a RenderTexture."""
        if not self.tile_map:
            return

        width = self.tile_size * self.tile_map.size[0]
        height = self.tile_size * self.tile_map.size[1]

        # Inicializa a RenderTexture se necessário
        if self.render_texture is None or self.render_texture.texture.width != width or self.render_texture.texture.height != height:
            self.render_texture = rl.load_render_texture(width, height)

        rl.begin_texture_mode(self.render_texture)
        rl.clear_background(rl.BLANK)

        for y, row in enumerate(self.tile_map.matrix):
            for x, tile_index in enumerate(row):
                if tile_index == -1: continue
                tx, ty = self.int2coord(tile_index)
                
                source_rec = rl.Rectangle(tx * self.tile_size, ty * self.tile_size, self.tile_size, self.tile_size)
                dest_rec = rl.Rectangle(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                
                rl.draw_texture_pro(self.tile_set, source_rec, dest_rec, rl.Vector2(0, 0), 0, rl.WHITE)

        rl.end_texture_mode()
        self.dirty = False

    def render(self):
        if not self.tile_map:
            return

        if self.dirty or self.render_texture is None:
            self.update_cache()

        transform = self.global_transform
        tex = self.render_texture.texture

        # Source Rect: Inverte Y porque RenderTextures são armazenadas invertidas no OpenGL
        source_rec = rl.Rectangle(0, 0, tex.width, -tex.height)

        # Dest Rect: Tamanho no mundo escalado
        dest_width = tex.width * transform.scale.x
        dest_height = tex.height * transform.scale.y
        
        dest_rec = rl.Rectangle(transform.position.x, transform.position.y, dest_width, dest_height)

        # Origin: Centro do mapa para rotação correta
        origin = rl.Vector2(dest_width / 2, dest_height / 2)

        rl.draw_texture_pro(tex, source_rec, dest_rec, origin, transform.angle, rl.WHITE)
