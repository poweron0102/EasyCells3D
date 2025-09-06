from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Geometry import Vec3

bola: Item

def init(game: Game):
    camera = game.CreateItem()
    camera.AddComponent(Camera(aspect_ratio=1.0))

    global bola
    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(0, 0, 2)



def loop(game: Game):
    bola.transform.position += Vec3(1, 0, 0)
