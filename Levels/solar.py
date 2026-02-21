from EasyCells3D import Game
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.Sphere import Sphere


def init(game: Game):
    camera = game.CreateItem()
    camera.AddComponent(Camera3D())
    camera.AddComponent(FreeCam())

    sphere = game.CreateItem()
    sphere.AddComponent(Sphere(
        1,
        texture_path="copper_bulb_lit_powered.png"
    ))

def loop(game: Game):
    pass