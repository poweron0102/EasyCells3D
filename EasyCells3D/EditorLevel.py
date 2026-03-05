from EasyCells3D.Components.CameraUI import CameraUI
from EasyCells3D.EditorComponents.ComponentsReloader import ComponentsReloader
from EasyCells3D.EditorComponents.EditorUI import EditorUI
from EasyCells3D.Game import Game



def init(game: Game):
    game.CreateItem().AddComponent(ComponentsReloader())

    camera = game.CreateItem().AddComponent(CameraUI())

    game.CreateItem().AddComponent(EditorUI())


def loop(game: Game):
    pass



