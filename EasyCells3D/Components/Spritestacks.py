import math

import pygame as pg
from midvoxio.voxio import vox_to_arr

from . import Camera
from .Camera import Drawable
from .Component import Transform


class SpriteStacks(Drawable):

    @property
    def size(self) -> tuple[int, int]:
        return self.images[int(self.transform.angle_deg // self.angle)].get_size()

    @property
    def image(self) -> pg.Surface:
        return self.images[int(self.transform.angle_deg // self.angle)]

    def image_at(self, angle_deg: float) -> pg.Surface:
        return self.images[int(angle_deg // self.angle)]

    def __init__(self, image_path: str | pg.Surface, size: tuple[int, int] = None, angle_deg: float = 15.0):
        super().__init__()
        if isinstance(image_path, pg.Surface):
            image = image_path
        else:
            image = pg.image.load(f"Assets/{image_path}").convert_alpha()

        self.images = SpriteStacks.spritestacks_from_img(image, size, angle_deg)
        self.angle = angle_deg

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

        # Apply nearest neighbor scaling
        # image = pg.transform.scale(
        #     self.image_at(self.word_position.angle_deg - Transform.Global.angle_deg),
        #     scaled_size
        # )

        image = pg.transform.scale(self.image, scaled_size)

        # Draw the image
        size = image.get_size()
        camera.screen.blit(
            image,
            (
                position.x - cam_x - size[0] // 2,
                position.y - cam_y - size[1] // 2
            )
        )

    @staticmethod
    def spritestacks_from_img(image: pg.Surface, size: tuple[int, int], angle: float) -> list[pg.Surface]:
        layers_orig = [
            image.subsurface((x * size[0], 0, size[0], size[1]))
            for x in range(image.get_width() // size[0])
        ]

        # Calculate the diagonal size for the rotation bounding box
        diagonal = math.ceil(math.sqrt(size[0] ** 2 + size[1] ** 2))
        size = (diagonal, diagonal + len(layers_orig))

        result: list[pg.Surface] = []
        a = 0.0
        while a < 360.0:
            layer_r = pg.Surface(size, pg.SRCALPHA)
            for index, layer in enumerate(layers_orig):
                rotated = pg.transform.rotate(layer, -a)
                layer_r.blit(
                    rotated,
                    (
                        size[0] // 2 - rotated.get_width() // 2,
                        (size[1] - rotated.get_height() // 2 - size[1] // 2) - index + ((len(layers_orig)) / 2)
                    )
                )
            result.append(layer_r)
            a += angle

        return result

    @staticmethod
    def voxel2img(file_path) -> tuple[pg.Surface, tuple[int, int]]:
        """
        Reads a .vox voxel model file and converts it to a 3D list of pygame.Color objects.

        Args:
            file_path (str): The path to the .vox file.

        Returns:
            pg.Surface: A 3D list of pygame.Color objects representing the voxel model.
            tuple[int, int]: The size of the voxel model.
        """
        # Parse the .vox file
        vox_arr = vox_to_arr(file_path)
        size_x, size_y, size_z, _ = vox_arr.shape

        surface = pg.Surface((size_x * size_z, size_y), pg.SRCALPHA)

        # Populate the 3D list with pygame.Color instances
        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    surface.set_at(
                        (z * size_x + x, y),
                        pg.Color(*map(lambda a: a * 255, vox_arr[x, y, z]))
                    )

        return surface, (size_x, size_y)