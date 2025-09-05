import asyncio
import sys
from importlib import import_module
from types import ModuleType
from typing import Callable, TYPE_CHECKING

import pygame as pg
from pygame.event import Event

from EasyCells3D.NewGame import NewGame

if TYPE_CHECKING:
    from EasyCells3D.Components import Item

ItemClass: type

pg.init()


class Game:
    instances: dict[int, 'Game'] = {}
    instances_count: int = 0
    current_instance: int = 0

    @staticmethod
    def instance() -> 'Game':
        return Game.instances[Game.current_instance]

    level: ModuleType
    events: list[Event]

    _game_name: str

    @property
    def game_name(self) -> str:
        return self._game_name

    @game_name.setter
    def game_name(self, value: str):
        self._game_name = value
        if not self.show_fps:
            pg.display.set_caption(value)

    def __init__(
            self,
            start_level: str | ModuleType,
            game_name: str,
            show_fps: bool = False,
            screen_resolution: tuple[int, int] = (800, 600),
            screen_flag: int = 0,
            screen: pg.Surface | None = None,
    ):
        # imports: -=-=-=-=-
        global ItemClass
        from EasyCells3D.scheduler import Scheduler
        from EasyCells3D.Components import Item
        ItemClass = Item
        # imports: -=-=-=-=-

        self.my_instance = Game.instances_count
        Game.instances[self.my_instance] = self
        Game.instances_count += 1

        if screen is None:
            self.screen: pg.Surface = pg.display.set_mode(screen_resolution, screen_flag)
        else:
            self.screen: pg.Surface = screen

        Game.events = pg.event.get()

        self.show_fps = show_fps
        self.game_name = game_name

        self.clock = pg.time.Clock()
        self.time = pg.time.get_ticks()
        self.last_time = pg.time.get_ticks()
        self.delta_time = 0
        self.run_time = 0
        self.scheduler = Scheduler(self)
        self.item_list: list[Item] = []
        self.to_init: list[Callable] = []
        self.new_game(start_level, supress=True)
        # pg.mouse.set_visible

        self.current_level = "Level_name"

    def new_game(self, level: str | ModuleType, supress=False):
        if type(level) == ModuleType:
            self.level = level
            self.current_level = level.__name__
        else:
            self.level = import_module(f".{level}", "Levels")
            self.current_level = level


        self.run_time = 0

        for item in list(self.item_list):
            if item.destroy_on_load:
                item.Destroy()

        self.scheduler.clear()
        self.level.init(self)

        self.update()

        if not supress:
            raise NewGame

    def CreateItem(self) -> 'Item':
        return ItemClass(self)

    def update(self):
        Game.events = pg.event.get()
        for event in Game.events:
            if event.type == pg.QUIT:  # or (event.type == pg.KEYDOWN and event.key == pg.k_ESCAPE):
                pg.quit()
                sys.exit()

        pg.display.flip()
        self.screen.fill((30, 30, 30))  # Cinza
        self.clock.tick(1000) # Limitando a 30 FPS
        self.last_time = self.time
        self.time = pg.time.get_ticks()
        self.delta_time = (self.time - self.last_time) / 1000.0
        self.run_time += self.delta_time

        if self.show_fps:
            pg.display.set_caption(f'{self.game_name}   FPS: {self.clock.get_fps():.0f}')

    def run(self):
        while True:
            self.update()
            try:
                for function in self.to_init:
                    function()
                self.to_init.clear()

                for item in list(self.item_list):
                    item.update()

                self.level.loop(self)

                self.scheduler.update()
            except NewGame:
                pass

    async def run_async(self):
        while True:
            self.update()
            try:
                for function in self.to_init:
                    function()
                self.to_init.clear()

                for item in list(self.item_list):
                    item.update()

                self.level.loop(self)

                self.scheduler.update()
            except NewGame:
                pass

            await asyncio.sleep(0)


    def run_once(self):
        previous_instance: int = Game.current_instance
        Game.current_instance = self.my_instance

        self.update()
        try:
            for function in self.to_init:
                function()
            self.to_init.clear()

            for item in list(self.item_list):
                item.update()

            self.level.loop(self)

            self.scheduler.update()
        except NewGame:
            pass

        Game.current_instance = previous_instance
