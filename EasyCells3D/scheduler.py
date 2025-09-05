import traceback
from typing import Generator, Callable

from .NewGame import NewGame
from .Game import Game


class Scheduler:
    instance: 'Scheduler' = None

    def __init__(self, game: Game):
        self.game = game

        self._generators: list[Generator] = []
        self._generators_times: list[float] = []
        self._generators_dict: dict[any, Generator] = {}
        self._generators_dict_times: dict[any, float] = {}

        self._times: list[float] = []
        self._functions: list[Callable] = []

        self._times_dict: dict[any, float] = {}
        self._functions_dict: dict[any, Callable] = {}

        if not Scheduler.instance:
            Scheduler.instance = self

    def update(self):
        # functions
        for index, function in enumerate(self._functions):
            if self._times[index] < self.game.run_time:
                try:
                    function()
                except (KeyboardInterrupt, SystemExit, NewGame) as e:
                    raise e
                except Exception as e:
                    print(f"Error in {function}:\n    {e}")
                    traceback.print_exc()
                try:
                    self._times.pop(index)
                    self._functions.pop(index)
                except IndexError:
                    pass

        for key in list(self._times_dict.keys()):
            if self._times_dict[key] < self.game.run_time:
                try:
                    self._functions_dict[key]()
                except (KeyboardInterrupt, SystemExit, NewGame) as e:
                    raise e
                except Exception as e:
                    print(f"Error in {self._functions_dict[key]}:\n    {e}")
                    traceback.print_exc()
                try:
                    self._times_dict.pop(key)
                    self._functions_dict.pop(key)
                except KeyError:
                    pass

        # generators
        for index, generator in enumerate(self._generators):
            if self._generators_times[index] > self.game.run_time:
                continue
            try:
                next_time = next(generator)
                if next_time:
                    self._generators_times[index] = self.game.run_time + next_time
            except StopIteration:
                self._generators.remove(generator)
                self._generators_times.pop(index)
            except (KeyboardInterrupt, SystemExit, NewGame) as e:
                raise e
            except Exception as e:
                print(f"Error in {generator}:\n    {e}")
                traceback.print_exc()

        for key in list(self._generators_dict.keys()):
            generator = self._generators_dict[key]
            if self._generators_dict_times[key] > self.game.run_time:
                continue
            try:
                next_time = next(generator)
                if next_time:
                    self._generators_dict_times[key] = self.game.run_time + next_time
            except StopIteration:
                self._generators_dict.pop(key)
                self._generators_dict_times.pop(key)
            except (KeyboardInterrupt, SystemExit, NewGame) as e:
                raise e
            except Exception as e:
                print(f"Error in {generator}:\n    {e}")
                traceback.print_exc()

    def add(self, time: float, function: Callable):
        self._times.append(self.game.run_time + time)
        self._functions.append(function)

    def remove(self, function: Callable):
        index = self._functions.index(function)
        self._times.pop(index)
        self._functions.pop(index)

    def add_dict(self, key, time: float, function: Callable):
        self._times_dict[key] = self.game.run_time + time
        self._functions_dict[key] = function

    def remove_dict(self, key):
        self._times_dict.pop(key)
        self._functions_dict.pop(key)

    def add_generator(self, generator: Generator, time: float = 0):
        self._generators.append(generator)
        self._generators_times.append(self.game.run_time + time)

    def remove_generator(self, generator: Generator):
        index = self._generators.index(generator)
        self._generators.pop(index)
        self._generators_times.pop(index)

    def add_dict_generator(self, key, generator: Generator, time: float = 0):
        self._generators_dict[key] = generator
        self._generators_dict_times[key] = self.game.run_time + time

    def remove_dict_generator(self, key):
        try:
            self._generators_dict.pop(key)
            self._generators_dict_times.pop(key)
        except KeyError:
            return KeyError

    def change_time_dict_generator(self, key, time: float):
        self._generators_dict_times[key] = self.game.run_time + time

    def change_time_generator(self, generator: Generator, time: float):
        index = self._generators.index(generator)
        self._generators_times[index] = self.game.run_time + time

    def change_time_dict(self, key, time: float):
        self._times_dict[key] = self.game.run_time + time

    def change_time(self, function: Callable, time: float):
        index = self._functions.index(function)
        self._times[index] = self.game.run_time + time

    def clear(self):
        self._times.clear()
        self._functions.clear()
        self._times_dict.clear()
        self._functions_dict.clear()
        self._generators.clear()
        self._generators_times.clear()
        self._generators_dict.clear()
        self._generators_dict_times.clear()


class Tick:
    def __init__(self, time: float):
        self.time = time
        self.on = True

    def turn_off(self):
        self.on = False

    def turn_on(self):
        self.on = True

    def reset(self):
        self.on = False
        Scheduler.instance.add(self.time, self.turn_on)

    def __call__(self) -> bool:
        if self.on:
            self.on = False
            Scheduler.instance.add(self.time, self.turn_on)
            return True
        return False
