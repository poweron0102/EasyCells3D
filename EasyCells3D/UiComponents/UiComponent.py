import pygame as pg
from enum import Enum

from ..Components import Camera
from ..Components.Camera import Hittable
from ..Components.Component import Transform
from ..Geometry import Vec2


def panel_maker(size: Vec2[int], base_panel: pg.Surface) -> pg.Surface:
    pieces_size = (base_panel.get_width() // 3, base_panel.get_height() // 3)
    pieces = [
        [
            base_panel.subsurface(
                (x * pieces_size[0], y * pieces_size[1], pieces_size[0], pieces_size[1])
            )
            for x in range(3)
        ]
        for y in range(3)
    ]

    # size = first multiple of pieces_size that is bigger than size
    size = Vec2(
        (size.x // pieces_size[0] + 1) * pieces_size[0],
        (size.y // pieces_size[1] + 1) * pieces_size[1]
    )

    panel = pg.Surface(size.to_tuple, pg.SRCALPHA)
    # panel = pg.Surface(size.to_tuple)

    num_of_pieces = (size.x // pieces_size[0], size.y // pieces_size[1])
    for y in range(1, num_of_pieces[1] - 1):
        for x in range(1, num_of_pieces[0] - 1):
            panel.blit(pieces[1][1], (x * pieces_size[0], y * pieces_size[1]))

    for x in range(1, num_of_pieces[0] - 1):
        panel.blit(pieces[0][1], (x * pieces_size[0], 0))
        panel.blit(pieces[2][1], (x * pieces_size[0], (num_of_pieces[1] - 1) * pieces_size[1]))

    for y in range(1, num_of_pieces[1] - 1):
        panel.blit(pieces[1][0], (0, y * pieces_size[1]))
        panel.blit(pieces[1][2], ((num_of_pieces[0] - 1) * pieces_size[0], y * pieces_size[1]))

    panel.blit(pieces[0][0], (0, 0))
    panel.blit(pieces[2][0], (0, (num_of_pieces[1] - 1) * pieces_size[1]))
    panel.blit(pieces[0][2], ((num_of_pieces[0] - 1) * pieces_size[0], 0))
    panel.blit(pieces[2][2], ((num_of_pieces[0] - 1) * pieces_size[0], (num_of_pieces[1] - 1) * pieces_size[1]))

    return panel


class UiAlignment(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    CENTER = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    GAME_SPACE = 5


class UiComponent(Hittable):
    _draw_on_screen_space = True

    @property
    def draw_on_screen_space(self):
        return self._draw_on_screen_space

    @draw_on_screen_space.setter
    def draw_on_screen_space(self, value: bool):
        self._draw_on_screen_space = value
        self.draw = self.draw_screen_space if value else self.draw_world_space

    def __init__(self, position: Vec2[float], image: pg.Surface, z: int = -101,
                 alignment: UiAlignment = UiAlignment.TOP_LEFT):
        super().__init__()
        self.word_position = Transform()
        self.image = image
        self.position = position
        self.z = z
        self.draw_on_screen_space = False if alignment == UiAlignment.GAME_SPACE else True
        self.alignment = alignment

    def init(self):
        super().init()
        self.transform.z = self.z
        self.transform.position = self.position
        del self.position
        del self.z

    def calculate_screen_offset(self) -> Vec2:
        if self.alignment == UiAlignment.TOP_LEFT:
            return Vec2(0, 0)
        elif self.alignment == UiAlignment.TOP_RIGHT:
            return Vec2(self.game.screen.get_width(), 0)
        elif self.alignment == UiAlignment.CENTER:
            return Vec2(self.game.screen.get_width() // 2, self.game.screen.get_height() // 2)
        elif self.alignment == UiAlignment.BOTTOM_LEFT:
            return Vec2(0, self.game.screen.get_height())
        elif self.alignment == UiAlignment.BOTTOM_RIGHT:
            return Vec2(self.game.screen.get_width(), self.game.screen.get_height())
        elif self.alignment == UiAlignment.GAME_SPACE:
            return Vec2(0, 0)

    def draw_screen_space(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        position = self.transform.position + self.calculate_screen_offset()
        camera.screen.blit(
            self.image,
            (
                position.x - self.image.get_width() // 2,
                position.y - self.image.get_height() // 2
            )
        )

    def draw_world_space(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        position = self.word_position * scale
        position.scale *= scale

        # Get size and apply nearest neighbor scaling
        original_size = self.image.get_size()
        new_size = (int(original_size[0] * position.scale), int(original_size[1] * position.scale))
        image = pg.transform.scale(self.image, new_size)

        # Draw base_image
        size = image.get_size()
        camera.screen.blit(
            image,
            (
                position.x - cam_x - size[0] // 2,
                position.y - cam_y - size[1] // 2
            )
        )

    def loop(self):
        self.word_position = Transform.Global
