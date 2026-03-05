from EasyCells3D.Components.CameraUI import CameraUI
from EasyCells3D.EditorComponents.ComponentsReloader import ComponentsReloader
from EasyCells3D.EditorComponents.EditorUI import EditorUI
from EasyCells3D.Game import Game

import Levels.platform


running_game: Game

def init(game: Game):
    global running_game
    game.CreateItem().AddComponent(ComponentsReloader())

    camera = game.CreateItem().AddComponent(CameraUI())

    running_game = Game(Levels.platform, "Platform", True, (1280, 720), True)

    game.CreateItem().AddComponent(EditorUI(running_game))



def loop(game: Game):
    with running_game:
        running_game.run_once()



