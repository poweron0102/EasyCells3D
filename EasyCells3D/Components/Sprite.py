import math

from . import Camera
from .Camera import Drawable
from .Component import Transform

import pygame as pg


def convert_to_grayscale(surface: pg.Surface, strength: float = 1) -> pg.Surface:
    # Cria uma nova superfície com o mesmo tamanho e formato
    grayscale_surface = pg.Surface(surface.get_size(), pg.SRCALPHA)

    # Converte para um formato apropriado para píxel access
    surface_locked = surface.copy()

    # Percorre cada pixel da superfície
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            # Obtém a cor do pixel
            r, g, b, a = surface_locked.get_at((x, y))

            # Calcula a tonalidade de cinza usando a média
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)

            # Define a nova cor (tons de cinza) na nova superfície
            grayscale_surface.set_at(
                (x, y),
                (
                    int(r * (1 - strength) + gray * strength),
                    int(g * (1 - strength) + gray * strength),
                    int(b * (1 - strength) + gray * strength),
                    a
                )
            )

    return grayscale_surface


class Sprite(Drawable):
    image: pg.Surface
    draw_image: pg.Surface
    index: int = 0
    size: tuple[int, int] = (0, 0)

    def __init__(self, image_path: str | pg.Surface, size: tuple[int, int] = None):
        super().__init__()
        if isinstance(image_path, pg.Surface):
            self.image = image_path
        else:
            self.image = pg.image.load(f"Assets/{image_path}").convert_alpha()

        self.size = size if size else self.image.get_size()

        self.draw_image = pg.Surface(self.size, pg.SRCALPHA)

        self.horizontal_flip = False
        self.vertical_flip = False

        self.word_position = Transform()

    def loop(self):
        self.word_position = Transform.Global

    def draw(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        # Calculate the sprite's position and scaled size
        position = self.word_position * scale
        position.scale *= scale
        original_size = self.size  # Get original image size
        scaled_size = (int(original_size[0] * position.scale), int(original_size[1] * position.scale))

        # Calculate the sprite's bounding box on the screen
        left = position.x - scaled_size[0] // 2
        right = position.x + scaled_size[0] // 2
        top = position.y - scaled_size[1] // 2
        bottom = position.y + scaled_size[1] // 2

        # Get the screen dimensions (viewport)
        screen_width, screen_height = camera.screen.get_size()

        # Check if the sprite is completely outside the viewport
        if right < cam_x or left > cam_x + screen_width or bottom < cam_y or top > cam_y + screen_height:
            return  # Skip drawing if out of bounds

        # Crop base_image without losing alpha channel
        self.draw_image.fill((0, 0, 0, 0))
        self.draw_image.blit(
            self.image,
            (0, 0),
            (self.index * self.size[0], 0, self.size[0], self.size[1])
        )

        # Flip base_image
        if self.horizontal_flip or self.vertical_flip:
            self.draw_image = pg.transform.flip(self.draw_image, self.horizontal_flip, self.vertical_flip)

        # Apply nearest neighbor scaling
        image = pg.transform.scale(self.draw_image, scaled_size)

        # Rotate the image
        image = pg.transform.rotate(image, -math.degrees(position.angle))

        # Draw the image
        size = image.get_size()
        camera.screen.blit(
            image,
            (
                position.x - cam_x - size[0] // 2,
                position.y - cam_y - size[1] // 2
            )
        )


class SimpleSprite(Sprite):
    """
    A Sprite with that don't rotate, scale, flip, or be animated.
    """

    def draw(self, cam_x: float, cam_y: float, scale: float, camera: Camera):
        # Calculate the sprite's position and scaled size
        position = self.word_position * scale
        position.scale *= scale

        # Calculate the sprite's bounding box on the screen
        left = position.x - self.size[0] // 2
        right = position.x + self.size[0] // 2
        top = position.y - self.size[1] // 2
        bottom = position.y + self.size[1] // 2

        # Get the screen dimensions (viewport)
        screen_width, screen_height = camera.screen.get_size()

        # Check if the sprite is completely outside the viewport
        if right < cam_x or left > cam_x + screen_width or bottom < cam_y or top > cam_y + screen_height:
            return  # Skip drawing if out of bounds

        # Draw the image
        camera.screen.blit(
            self.image,
            (
                position.x - cam_x - self.size[0] // 2,
                position.y - cam_y - self.size[1] // 2
            )
        )


