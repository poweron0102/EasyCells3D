import pygame as pg

from typing import Callable

from ..Game import Game
from ..Components.Camera import Camera
from ..Components.Component import Transform
from ..Geometry import Vec2
from .UiComponent import UiComponent, panel_maker, UiAlignment
from ..scheduler import Tick

pg.font.init()


class TextInput(UiComponent):
    def __init__(
            self,
            position: Vec2[float],
            inactive_text: str,
            empty_panel: pg.Surface,
            font_size: int = 32,
            size: Vec2[int] = Vec2(200, 50),
            inactive_font_color: pg.Color = pg.Color("Grey"),
            active_font_color: pg.Color = pg.Color("Black"),
            z: int = -101,
            # hover_panel: pg.Surface = None,
            active_panel: pg.Surface = None,
            on_active: Callable = None,
            # on_hover: Callable = None,
            on_write: Callable = None,
            on_enter: Callable = None,
            on_inactive: Callable = None,
            font: str = None,
            alignment: UiAlignment = UiAlignment.TOP_LEFT,
    ):
        pg.scrap.init()
        self.is_active = False
        self.text = ""
        self.write_tick = Tick(0.01)

        self.font = pg.font.Font(f"Assets/{font}", font_size) if font is not None else pg.font.Font(None, font_size)

        inactive_text = self.font.render(inactive_text, True, inactive_font_color)
        self.empty_panel = panel_maker(size, empty_panel)

        if active_panel is None:
            self.active_panel = self.empty_panel.copy()
        else:
            self.active_panel = panel_maker(size, active_panel)
        # if hover_panel is None:
        #     self.hover_panel = self.empty_panel.copy()
        # else:
        #     self.hover_panel = panel_maker(size, hover_panel)

        self.empty_panel.blit(
            inactive_text,
            (
                self.empty_panel.get_width() / 2 - inactive_text.get_width() / 2,
                self.empty_panel.get_height() / 2 - inactive_text.get_height() / 2
            )
        )

        self.active_font_color = active_font_color
        self.on_active = on_active if on_active is not None else lambda: None
        # self.on_hover = on_hover if on_hover is not None else lambda: None
        self.on_write = on_write if on_write is not None else lambda: None
        self.on_enter = on_enter if on_enter is not None else lambda: None
        self.on_inactive = on_inactive if on_inactive is not None else lambda: None

        super().__init__(position, self.empty_panel, z, alignment)

    def loop(self):
        super().loop()

        if self.is_mouse_over():
            if pg.mouse.get_pressed()[0] and not self.is_active:
                self.is_active = True
                self.on_active()
        else:
            if pg.mouse.get_pressed()[0] and self.is_active:
                self.is_active = False
                self.on_inactive()

        if self.is_active and self.write_tick.on:
            if pg.key.get_mods() & pg.KMOD_CTRL:
                for event in Game.events:
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_v:
                            self.text += pg.scrap.get(pg.SCRAP_TEXT)[:-1].decode("utf-8")
                            print(f"Ctrl + V: \"{self.text}\"")
                            if self.on_write.__code__.co_argcount == 1:
                                self.on_write(self.text)
                            else:
                                self.on_write()
                            self.write_tick.reset()
            else:
                for event in Game.events:
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_RETURN:
                            if self.on_enter.__code__.co_argcount == 1:
                                self.on_enter(self.text)
                            else:
                                self.on_enter()

                        elif event.key == pg.K_BACKSPACE:
                            self.text = self.text[:-1]

                        else:
                            self.text += event.unicode

                        if self.on_write.__code__.co_argcount == 1:
                            self.on_write(self.text)
                        else:
                            self.on_write()

                        self.write_tick.reset()

            text_surface = self.font.render(self.text, True, self.active_font_color)
            active_panel = self.active_panel.copy()
            active_panel.blit(
                text_surface,
                (
                    self.active_panel.get_width() / 2 - text_surface.get_width() / 2,
                    self.active_panel.get_height() / 2 - text_surface.get_height() / 2
                )
            )
            self.image = active_panel

        if self.text == "":
            self.image = self.empty_panel

    def is_mouse_over(self) -> bool:
        if self.draw_on_screen_space:
            position = self.calculate_screen_offset() + self.transform.position
            return self.image.get_rect(
                topleft=(
                    position.x - self.image.get_width() // 2,
                    position.y - self.image.get_height() // 2
                )
            ).collidepoint(pg.mouse.get_pos())
        else:
            mouse = Camera.get_global_mouse_position()
            size = Vec2(*self.image.get_size()) * self.transform.scale
            top_left = Transform.Global.position - Vec2(size.x // 2, size.y // 2)

            return pg.Rect(top_left.to_tuple, size.to_tuple).collidepoint(mouse.to_tuple)
