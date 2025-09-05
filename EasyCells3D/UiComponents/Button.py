import pygame as pg

from typing import Callable

from ..Components.Camera import Camera
from ..Components.Component import Transform
from ..Geometry import Vec2
from .UiComponent import UiComponent, panel_maker, UiAlignment

pg.font.init()


class Button(UiComponent):
    def __init__(
            self,
            position: Vec2[float],
            text: str,
            base_panel: pg.Surface,
            font_size: int = 32,
            font_color: pg.Color = pg.Color("Black"),
            z: int = -101,
            hover_panel: pg.Surface = None,
            on_click: Callable = None,
            on_hover: Callable = None,
            font: str = None,
            alignment: UiAlignment = UiAlignment.TOP_LEFT,
    ):
        self.is_clicked = False

        font = pg.font.Font(f"Assets/{font}", font_size) if font is not None else pg.font.Font(None, font_size)
        text_surface = font.render(text, True, font_color)

        self.base_image = panel_maker(
            Vec2(text_surface.get_width() + font_size, text_surface.get_height() + font_size), base_panel
        )

        self.base_image.blit(
            text_surface,
            (
                self.base_image.get_width() / 2 - text_surface.get_width() / 2,
                self.base_image.get_height() / 2 - text_surface.get_height() / 2
            )
        )
        if hover_panel is None:
            self.hover_image = self.base_image
        else:
            self.hover_image = panel_maker(
                Vec2(text_surface.get_width() + font_size, text_surface.get_height() + font_size), hover_panel
            )
            self.hover_image.blit(
                text_surface,
                (
                    self.hover_image.get_width() / 2 - text_surface.get_width() / 2,
                    self.hover_image.get_height() / 2 - text_surface.get_height() / 2
                )
            )

        self.on_click = on_click if on_click is not None else lambda: None
        self.on_hover = on_hover if on_hover is not None else lambda: None

        super().__init__(position, self.base_image, z, alignment)

    def loop(self):
        super().loop()

        if not pg.mouse.get_pressed()[0]:
            self.is_clicked = False

        if self.is_mouse_over():
            self.on_hover()
            self.image = self.hover_image
            if pg.mouse.get_pressed()[0] and not self.is_clicked:
                self.is_clicked = True
                self.on_click()
        else:
            self.image = self.base_image

    def is_mouse_over(self) -> bool:
        if self.draw_on_screen_space:
            position = self.calculate_screen_offset() + self.transform.position
            return self.base_image.get_rect(
                topleft=(
                    position.x - self.base_image.get_width() // 2,
                    position.y - self.base_image.get_height() // 2
                )
            ).collidepoint(pg.mouse.get_pos())
        else:
            mouse = Camera.get_global_mouse_position()
            size = Vec2(*self.image.get_size()) * self.transform.scale
            top_left = Transform.Global.position - Vec2(size.x // 2, size.y // 2)

            return pg.Rect(top_left.to_tuple, size.to_tuple).collidepoint(mouse.to_tuple)
