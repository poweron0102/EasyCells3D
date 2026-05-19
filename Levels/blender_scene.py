from EasyCells3D import Game, SceneLoader
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Geometry import Vec3


def init(game: Game):
    camera = game.CreateItem()
    camera.name = "Camera"
    camera.transform.position = Vec3(0, 2, 6)
    camera.AddComponent(Camera3D())
    camera.AddComponent(FreeCam())

    SceneLoader(game).load("Assets/Blender/scene.glb")


def loop(game: Game):
    pass
