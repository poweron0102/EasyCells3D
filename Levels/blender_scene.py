from EasyCells3D import ComponentRegistry, Game, SceneLoader
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Geometry import Vec3
from UserComponents.ratating_obj import RotatingObj


def init(game: Game):
    ComponentRegistry.register("RotatingObj", RotatingObj)

    camera = game.CreateItem()
    camera.name = "Camera"
    camera.transform.position = Vec3(0, 2, 6)
    camera.AddComponent(Camera3D())
    camera.AddComponent(FreeCam())

    SceneLoader(game).load("Assets/Blender/FoodPack.glb")


def loop(game: Game):
    pass
