import os
from functools import cache
from typing import Callable

import pygame as pg

from .Component import Transform, Component
from .Sprite import Sprite
from ..Geometry import Vec2


class TileMap3D(Component):
    def __init__(self, matrix: list[list[list[int]]]):
        self.matrix = matrix
        self.size: tuple[int, int, int] = (len(matrix[0][0]), len(matrix[0]), len(matrix))

        self.on_tile_change: list[Callable[[int, int, int, int], None]] = []

    def get_tile(self, x: int, y: int, z) -> int:
        return self.matrix[z][y][x]

    def set_tile(self, x: int, y: int, z: int, value: int):
        self.matrix[z][y][x] = value
        for callback in self.on_tile_change:
            callback(x, y, z, value)

    @staticmethod
    def load_from_csv(dir_path: str) -> 'TileMap3D':
        mat: list[list[list[int]]] = []

        for file in sorted(os.listdir(f"Assets/{dir_path}")):
            with open(f"Assets/{dir_path}/{file}") as f:
                mat.append([list(map(int, line.split(","))) for line in f.readlines()])

        for index, mat0 in enumerate(mat):
            mat1: list[list[int]] = [[-1 for _ in range(len(mat0[0]))] for _ in range(len(mat0))]
            for i in range(index, len(mat0) - index):
                for j in range(index, len(mat0[i]) - index):
                    mat1[i][j] = mat0[i - index][j - index]

            mat[index] = mat1

        return TileMap3D(mat)


class TileMapIsometricRenderer(Component):
    tile_map: TileMap3D

    def __init__(self, tile_set: str | pg.Surface, tile_size: tuple[int, int]):
        if isinstance(tile_set, str):
            self.tile_set = pg.image.load(f"Assets/{tile_set}").convert_alpha()
        else:
            self.tile_set = tile_set

        self.tile_size: tuple[int, int] = tile_size

        size = self.tile_set.get_size()
        self.word_position = Transform()
        self.matrix_size = (size[0] // tile_size[0], size[1] // tile_size[1])
        self.sprites: list[Sprite] = []

    def init(self):
        self.tile_map = self.GetComponent(TileMap3D)
        self.update_image()

    def int2coord(self, value: int) -> tuple[int, int]:
        return value % self.matrix_size[0], value // self.matrix_size[0]

    def coord2int(self, coord: tuple[int, int]) -> int:
        return coord[0] + coord[1] * self.matrix_size[0]

    @cache
    def get_tile(self, x: int, y: int) -> pg.Surface:
        return self.tile_set.subsurface(
            (x * self.tile_size[0], y * self.tile_size[1], self.tile_size[0], self.tile_size[1])
        )

    def get_tile_word_position(self, word_x: float, word_y: float, z_index: int) -> tuple[float, float] | None:
        # Calculate hy (same as in update_image)
        hy = (sum(self.tile_map.size) * self.tile_size[1] // 4 + self.tile_size[1] // 2) // 4

        # Reverse the transformation
        adjusted_y = word_y + hy + z_index * self.tile_size[1] / 2

        # Calculate x + y and x - y
        sum_x_y = (adjusted_y * 4) / self.tile_size[1]
        diff_x_y = (word_x * 2) / self.tile_size[0]

        # Solve for x and y
        x_index = (sum_x_y + diff_x_y) / 2
        y_index = (sum_x_y - diff_x_y) / 2

        return x_index, y_index

    def world_to_isometric(self, x: float, y: float, z: float) -> tuple[int, int]:
        pass  # TODO

    def get_draw_order(self, x: float, y: float, z: float) -> float:
        return -0.01 * (x + y * self.tile_map.size[0] + z * self.tile_map.size[0] * self.tile_map.size[1])

    def update_image(self):
        hy = (sum(self.tile_map.size) * self.tile_size[1] // 4 + self.tile_size[1] // 2) // 4

        draw_ord = 0.01
        for z in range(self.tile_map.size[2]):
            for y in range(self.tile_map.size[1]):
                for x in range(self.tile_map.size[0]):
                    tile = self.tile_map.get_tile(x, y, z)
                    draw_ord -= 0.01
                    # print(f"x: {x}, y: {y}, z: {z}, draw_ord: {draw_ord}, draw_ord_from func: {self.get_draw_order(x, y, z)}")
                    if tile == -1:
                        continue

                    sprite = self.item.CreateChild().AddComponent(Sprite(
                        self.get_tile(*self.int2coord(tile))
                    ))
                    sprite.transform.position = Vec2(
                        (x - y) * self.tile_size[0] // 2,
                        ((x + y) * self.tile_size[1] // 4 - z * self.tile_size[1] // 2) - hy
                    )
                    sprite.transform.z = draw_ord

