from .Collider import Collider, Polygon
from ..Components.Component import Transform
from ..Components.TileMap import TileMap


class TileMapCollider(Collider):
    def __init__(self, solids: set[int], tile_size: int, mask: int = 1, debug: bool = False):
        self.word_position = Transform()
        self.mask = mask
        self.solids = solids
        self.tile_size = tile_size
        self.debug = debug
        Collider.colliders.append(self)

    def init(self):
        self.word_position = self.CalculateGlobalTransform()
        polygons = []
        tile_map = self.GetComponent(TileMap)
        matrix: list[list[int]] = tile_map.matrix
        visited = [[False for _ in row] for row in matrix]

        size2 = self.tile_size * tile_map.size[0] / 2, self.tile_size * tile_map.size[1] / 2

        def find_max_rectangle(x_start, y_start):
            """ Encontra o maior retângulo possível começando em (x_start, y_start) """
            max_width = 0
            max_height = 0
            for y in range(y_start, len(matrix)):
                if matrix[y][x_start] not in self.solids or visited[y][x_start]:
                    break
                width = 0
                for x in range(x_start, len(matrix[0])):
                    if matrix[y][x] in self.solids and not visited[y][x]:
                        width += 1
                    else:
                        break
                if max_width == 0:
                    max_width = width
                else:
                    max_width = min(max_width, width)
                max_height += 1

            return max_width, max_height

        for y, row in enumerate(matrix):
            for x, tile in enumerate(row):
                if tile in self.solids and not visited[y][x]:
                    width, height = find_max_rectangle(x, y)

                    # Marcar os tiles cobertos pelo retângulo como visitados
                    for i in range(y, y + height):
                        for j in range(x, x + width):
                            visited[i][j] = True

                    # Criar o polígono para o retângulo
                    vertices = [
                        [x * self.tile_size - size2[0], y * self.tile_size - size2[1]],
                        [(x + width) * self.tile_size - size2[0], y * self.tile_size - size2[1]],
                        [(x + width) * self.tile_size - size2[0], (y + height) * self.tile_size - size2[1]],
                        [x * self.tile_size - size2[0], (y + height) * self.tile_size - size2[1]]
                    ]
                    polygons.append(Polygon(vertices))

        self.polygons = polygons
        self.compile_numba_functions()
