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

import Levels.test_rigidbody
import Levels.solar
import Levels.platform
import Levels.blender_scene
import Levels.blockbench_scene

if __name__ == '__main__':
    #GAME = Game(Levels.space_selector, "Spaceship", True, (1280, 720))
    GAME = Game(Levels.blender_scene, "blockbench_scene", True, (1280, 720), True)
    GAME.run()
    # asyncio.run(GAME.run_async())
