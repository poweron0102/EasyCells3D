"""
Dependencies:
    - pygame-ce
    - numpy
    - numba
    - scipy
    - pyfmodex
    - midvoxio, matplotlib
    - pygame_gui

    pip install pygame-ce numpy numba scipy pyfmodex midvoxio matplotlib pygame_gui
"""
# import asyncio

from EasyCells3D import Game
import pygame as pg

import Levels.test

if __name__ == '__main__':
    #GAME = Game(Levels.space_selector, "Spaceship", True, (1280, 720))
    GAME = Game(Levels.test, "RayTracing", True, (1280, 720), screen_flag=pg.RESIZABLE)
    GAME.run()
    # asyncio.run(GAME.run_async())
