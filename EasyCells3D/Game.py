import asyncio
from importlib import import_module
from types import ModuleType
from typing import Callable, TYPE_CHECKING

import pyray as rl

from EasyCells3D.NewGame import NewGame

if TYPE_CHECKING:
    from EasyCells3D.Components import Item

ItemClass: type


class Camera:
    def render(self):
        pass

    def add_to_game(self, game: 'Game', priority: int = -1):
        if priority < 0:
            game.cameras.append(self)
        else:
            game.cameras.insert(priority, self)



class Game:
    instance: 'Game' = None
    level: ModuleType
    _game_name: str

    @property
    def game_name(self) -> str:
        return self._game_name

    @game_name.setter
    def game_name(self, value: str):
        self._game_name = value
        if not self.show_fps:
            rl.set_window_title(value)

    def __init__(
            self,
            start_level: str | ModuleType,
            game_name: str,
            show_fps: bool = False,
            screen_resolution: tuple[int, int] = (800, 600),
            dynamic_resolution: bool = False
    ):
        # imports: -=-=-=-=-
        global ItemClass
        from EasyCells3D.scheduler import Scheduler
        from EasyCells3D.Components import Item
        ItemClass = Item
        # imports: -=-=-=-=-

        if Game.instance is None:
            Game.instance = self

        if dynamic_resolution:
            rl.set_config_flags(rl.FLAG_WINDOW_RESIZABLE)

        rl.init_window(screen_resolution[0], screen_resolution[1], game_name)
        rl.set_exit_key(rl.KeyboardKey.KEY_NULL)
        #rl.set_target_fps(60)

        self.show_fps = show_fps
        self.game_name = game_name

        self.time = rl.get_time() * 1000
        self.last_time = rl.get_time() * 1000
        self.delta_time = 0
        self.run_time = 0
        self.scheduler = Scheduler(self)
        self.current_level = "Name"
        self.cameras: list[Camera] = []
        self.item_list: list[Item] = []
        self.to_init: list[Callable] = []
        self.new_game(start_level, supress=True)

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
        rl.clear_background(rl.Color(30, 30, 30, 255))  # Cinza

        self.last_time = self.time
        self.time = rl.get_time() * 1000
        self.delta_time = rl.get_frame_time()
        self.run_time += self.delta_time

        if self.show_fps:
            rl.set_window_title(f'{self.game_name}   FPS: {rl.get_fps()}')

    def run(self):
        while not rl.window_should_close():
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

            # Início do frame de renderização
            rl.begin_drawing()
            for camera in self.cameras:
                camera.render()
            rl.end_drawing()
        rl.close_window()

    async def run_async(self):
        while not rl.window_should_close():
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

            rl.end_drawing()
            await asyncio.sleep(0)
        rl.close_window()

    def run_once(self):
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

        # Início do frame de renderização
        rl.begin_drawing()
        for camera in self.cameras:
            camera.render()
        rl.end_drawing()

    def __enter__(self):
        self.previous_game = Game.instance
        Game.instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Game.instance = self.previous_game
