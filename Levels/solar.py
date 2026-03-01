from EasyCells3D import Game
from EasyCells3D.Components import Camera3D, Item
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.Sphere import Sphere
from EasyCells3D.Components.StaticModel import StaticModel
from EasyCells3D.Geometry import Vec3
from UserComponents.ratating_obj import RotatingObj

sphere: Item

def init(game: Game):
    global sphere
    camera = game.CreateItem()
    camera.AddComponent(Camera3D())
    camera.AddComponent(FreeCam())

    sphere = game.CreateItem()
    sphere.AddComponent(Sphere(
        1,
        texture_path="copper_bulb_lit_powered.png"
    ))
    sphere.AddComponent(RotatingObj(5))

    sphere2 = sphere.CreateChild()
    sphere2.AddComponent(Sphere(
        1,
        texture_path="copper_bulb_lit_powered.png",
    ))
    sphere2.AddComponent(RotatingObj(10))
    sphere2.transform.position += Vec3(4, 0, 0)

    # mapa = game.CreateItem()
    # mapa.AddComponent(StaticModel("model/Floor_Dirt.gltf"))


def loop(game: Game):
    sphere.transform.angle += 10 * game.delta_time