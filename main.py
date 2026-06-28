"""
Dependencies:
    - raylib
    - numpy
    - numba
    - scipy

    pip install raylib numpy numba scipy
"""
from EasyCells3D import Game

import Levels.test_rigidbody
import Levels.solar
import Levels.platform
import Levels.physics3d_demo


if __name__ == '__main__':
    #GAME = Game(Levels.space_selector, "Spaceship", True, (1280, 720))
    GAME = Game(Levels.platform, "physics3d_demo", True, (1280, 720), True)
    GAME.run()
