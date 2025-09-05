import math
from typing import Callable

import pygame as pg

from .Component import Component, Transform
from .. import Game
from ..Geometry import Vec2


class Drawable(Component):
    cameras: list['Camera']

    def __init__(self):
        self.cameras = [Camera.instance()]

    def init(self):
        Camera.instance().to_draw.append(self)

    def draw(self, cam_x: float, cam_y: float, scale: float, camera: 'Camera'):
        pass

    def on_destroy(self):
        for camera in self.cameras:
            camera.to_draw.remove(self)

        self.on_destroy = lambda: None

    def add_camera(self, camera: 'Camera'):
        if camera not in self.cameras:
            camera.to_draw.append(self)
            self.cameras.append(camera)

    def remove_camera(self, camera: 'Camera'):
        if camera in self.cameras:
            self.cameras.remove(camera)
        if self in camera.to_draw:
            camera.to_draw.remove(self)

    def clear_cameras(self):
        for camera in self.cameras:
            camera.to_draw.remove(self)
        self.cameras.clear()

    def remove_main_camera(self):
        self.remove_camera(Camera.instance())


class Camera(Component):
    instances: dict[int, 'Camera'] = {}

    @staticmethod
    def instance() -> 'Camera':
        return Camera.instances[Game.current_instance]

    @property
    def scale(self):
        return self.screen.get_size()[self.scale_with] / self.size[self.scale_with]

    @property
    def screen(self):
        return self._screen if self._screen is not None else self.game.screen

    def __init__(self, size: None | tuple[float, float] = None, scale_with: int = 0, screen: pg.Surface = None,
                 fill_color: tuple[int, int, int, int] | None = None):
        """
        scale_with[0: width, 1: height]
        This camera will be considered the main camera if it is the first camera to be created
        """

        if Game.current_instance not in Camera.instances:
            Camera.instances[Game.current_instance] = self

        self.scale_with = scale_with
        self.to_draw: list[Drawable] = []
        self.debug_draws: list[Callable] = []
        self.word_position = Transform()
        self.size = size
        self._screen = screen
        self.fill_color = fill_color

    def init(self):
        if self.size is None:
            self.size = self.screen.get_size()

    def on_destroy(self):
        if Game.current_instance in Camera.instances and Camera.instances[Game.current_instance] == self:
            del Camera.instances[Game.current_instance]

        self.on_destroy = lambda: None

    def loop(self):
        self.word_position = Transform.Global
        self.to_draw.sort(key=lambda drawable: -drawable.transform.z)

        # Correct to camera size
        scale = self.scale

        position = Transform.Global
        cam_x = position.x * scale - self.screen.get_width() / 2
        cam_y = position.y * scale - self.screen.get_height() / 2

        if self._screen is not None and self.fill_color is not None:
            # Clear screen with transparent color
            self._screen.fill(self.fill_color)

        for drawable in self.to_draw:
            if drawable.enable:
                drawable.draw(cam_x, cam_y, scale, self)

        for function in self.debug_draws:
            function(cam_x, cam_y, scale, self)

        self.debug_draws.clear()

    @staticmethod
    def draw_debug_line(start: Vec2[float], end: Vec2[float], color: pg.Color, width: int = 1):
        def draw(cam_x: float, cam_y: float, scale: float, camera: 'Camera'):
            pg.draw.line(
                Camera.instance().screen,
                color,
                (start - Vec2(cam_x, cam_y)).to_tuple,
                (end - Vec2(cam_x, cam_y)).to_tuple,
                width
            )

        Camera.instance().debug_draws.append(draw)

    @staticmethod
    def draw_debug_ray(start: Vec2[float], angle: float, length: float, color: pg.Color, width: int = 1):
        Camera.draw_debug_line(
            start,
            Vec2(start.x + length * math.cos(angle), start.y + length * math.sin(angle)),
            color,
            width
        )

    @staticmethod
    def get_global_mouse_position() -> Vec2[float]:
        mouse = pg.mouse.get_pos()
        position = Camera.instance().word_position
        scale = Camera.instance().scale
        return Vec2(
            (mouse[0] + position.x * scale - Camera.instance().game.screen.get_width() / 2) / scale,
            (mouse[1] + position.y * scale - Camera.instance().game.screen.get_height() / 2) / scale
        )
