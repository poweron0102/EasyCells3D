from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Geometry import Vec3, Quaternion

import pygame as pg

bola: Item
camera: Item

def init(game: Game):
    global camera
    camera = game.CreateItem()
    camera.AddComponent(Camera(aspect_ratio=1.0))

    global bola
    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(0, 0, 2)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(1, 0, 2)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(-1, 0, 2)



def loop(game: Game):
    #bola.transform.position += Vec3(0.1, 0, 0)

    offset: Vec3[float] = Vec3(0, 0, 0)
    if pg.key.get_pressed()[pg.K_w]:
        offset += Vec3(0, 0, 0.1)
    if pg.key.get_pressed()[pg.K_s]:
        offset += Vec3(0, 0, -0.1)
    if pg.key.get_pressed()[pg.K_a]:
        offset += Vec3(-0.1, 0, 0)
    if pg.key.get_pressed()[pg.K_d]:
        offset += Vec3(0.1, 0, 0)
    camera.transform.position += offset

    rotation = Vec3(0, 0, 0)
    if pg.key.get_pressed()[pg.K_UP]:
        rotation += Vec3(-1, 0, 0)
    if pg.key.get_pressed()[pg.K_DOWN]:
        rotation += Vec3(1, 0, 0)
    if pg.key.get_pressed()[pg.K_LEFT]:
        rotation += Vec3(0, -1, 0)
    if pg.key.get_pressed()[pg.K_RIGHT]:
        rotation += Vec3(0, 1, 0)
    camera.transform.rotation = camera.transform.rotation * Quaternion.from_axis_angle(rotation.normalize(), 0.05)
