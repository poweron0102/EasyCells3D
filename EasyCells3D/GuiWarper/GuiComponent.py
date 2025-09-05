from pygame_gui import UIManager

from EasyCells3D import Vec2
from EasyCells3D.UiComponents import UiComponent, UiAlignment
import pygame as pg


class GuiComponent(UiComponent):
    def __init__(
            self,
            position: Vec2[float],
            size: Vec2[float],
            theme_path: str,
            z: int = -101,
            alignment: UiAlignment = UiAlignment.TOP_LEFT,
            enable_live_theme_updates: bool = False,
    ):
        self.img = pg.Surface(size.to_tuple, pg.SRCALPHA)
        self.ui_manager = UIManager(
            size.to_tuple,
            theme_path,
            enable_live_theme_updates,
        )

        super().__init__(position, self.img, z, alignment)

    def loop(self):
        for event in self.game.events:
            # subtract the mouse position from the event
            if event.type == pg.MOUSEMOTION:
                event.pos = (event.pos[0] - self.position.x, event.pos[1] - self.position.y)

            self.ui_manager.process_events(event)

        self.ui_manager.update(self.game.delta_time)

        self.img.fill((0, 0, 0, 0))
        self.ui_manager.draw_ui(self.img)
        super().loop()
