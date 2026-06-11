import pyray as rl

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

    render = rl.load_render_texture(800, 600)
    running_game = Game(Levels.platform, "Platform",  render_target=render)

    game.CreateItem().AddComponent(EditorUI(running_game))



def loop(game: Game):
    with running_game:
        running_game.run_once()



