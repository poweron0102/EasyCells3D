"""
Dependencies:
    - raylib
    - numpy
    - numba
    - scipy

    pip install raylib numpy numba scipy
"""
# import asyncio

from EasyCells3D import Game
from EasyCells3D import EditorLevel

if __name__ == '__main__':
    #GAME = Game(Levels.space_selector, "Spaceship", True, (1280, 720))
    GAME = Game(EditorLevel, "Editor", True, (1280, 720), True)
    GAME.run()
    # asyncio.run(GAME.run_async())
