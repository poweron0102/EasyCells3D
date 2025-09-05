import math
from typing import Callable

import pygame as pg

from .Camera import Drawable, Camera
from .Component import Component, Transform
from ..Geometry import Vec2


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


class TileMapRenderer(Drawable):
    tile_map: TileMap

    def __init__(self, tile_set: str | pg.Surface, tile_size: int):
        super().__init__()
        if isinstance(tile_set, str):
            self.tile_set = pg.image.load(f"Assets/{tile_set}").convert_alpha()
        else:
            self.tile_set = tile_set
        self.tile_size = tile_size

        size = self.tile_set.get_size()
        self.word_position = Transform()
        self.matrix_size = (size[0] // tile_size, size[1] // tile_size)
        Camera.instance().to_draw.append(self)

    def init(self):
        self.tile_map = self.GetComponent(TileMap)

    def int2coord(self, value: int) -> tuple[int, int]:
        return value % self.matrix_size[0], value // self.matrix_size[0]

    def coord2int(self, coord: tuple[int, int]) -> int:
        return coord[0] + coord[1] * self.matrix_size[0]

    def get_tile(self, x: int, y: int) -> pg.Surface:
        return self.tile_set.subsurface((x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))

    def get_tile_word_position(self, x: int, y: int) -> Vec2[float]:
        x_new = self.word_position.x + self.tile_size * (x - self.tile_map.size[0] // 2)
        y_new = self.word_position.y + self.tile_size * (y - self.tile_map.size[1] // 2)

        if self.tile_map.size[0] % 2 == 0: x_new += self.tile_size // 2
        if self.tile_map.size[1] % 2 == 0: y_new += self.tile_size // 2

        return Vec2(x_new, y_new)

    def loop(self):
        self.word_position = Transform.Global

    def draw(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        position = self.word_position * scale
        position.scale *= scale

        # Create a new surface to draw the tile map
        image = pg.Surface(
            (self.tile_size * self.tile_map.size[0], self.tile_size * self.tile_map.size[1]),
            pg.SRCALPHA
        )

        # Draw the tile map
        for y, row in enumerate(self.tile_map.matrix):
            for x, tile in enumerate(row):
                image.blit(self.get_tile(*self.int2coord(tile)), (x * self.tile_size, y * self.tile_size))

        # Get size and apply nearest neighbor scaling
        original_size = image.get_size()
        new_size = (int(original_size[0] * position.scale), int(original_size[1] * position.scale))
        image = pg.transform.scale(image, new_size)

        # Rotate base_image
        image = pg.transform.rotate(image, -math.degrees(position.angle))

        # Draw base_image
        size = image.get_size()
        camera.screen.blit(
            image,
            (
                position.x - cam_x - size[0] // 2,
                position.y - cam_y - size[1] // 2
            )
        )
