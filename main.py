"""
Dependencies:
    - raylib-py
    - numpy
    - numba
    - scipy

    pip install raylib-py numpy numba scipy
"""
# import asyncio

from EasyCells3D import Game

#import Levels.test
import Levels.test_rigidbody

if __name__ == '__main__':
    #GAME = Game(Levels.space_selector, "Spaceship", True, (1280, 720))
    GAME = Game(Levels.test_rigidbody, "RayLib-test", True, (1280, 720))
    GAME.run()
    # asyncio.run(GAME.run_async())
