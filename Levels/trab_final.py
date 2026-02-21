import pygame as pg
import math

from EasyCells3D import Game, Tick
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Components.VoxelsHittable import VoxelsHittable
from EasyCells3D.Geometry import Vec3
from EasyCells3D.Material import Material, Texture
from UserComponents.ratating_obj import RotatingObj

camera: Item
camera_component: Camera

def init(game: Game):
    global camera
    global camera_component
    camera = game.CreateItem()
    camera.AddComponent(FreeCam())
    camera_component = camera.AddComponent(
        Camera(sky_box=Texture.get("sky1.jpg"), vfov=60, ambient_light=Vec3(0.05, 0.05, 0.05))
    )


    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(material=Material(
        diffuse_color=Vec3(1, 0.7, 0.7),
        specular=0.6,
        shininess=4
    )))
    bola.transform.position = Vec3(0, 5, 0)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(material=Material(
        diffuse_color=Vec3(1, 0, 0),
        specular=0.6,
        shininess=4
    )))
    bola.transform.position = Vec3(0, 5, 2)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(material=Material(
        diffuse_color=Vec3(0, 1, 0),
        specular=0.6,
        shininess=4
    )))
    bola.transform.position = Vec3(0, 7, 0)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(material=Material(
        diffuse_color=Vec3(0, 0, 1),
        specular=0.6,
        shininess=4
    )))
    bola.transform.position = Vec3(0, 7, 2)



    # --- Naves ---
    nave = game.CreateItem()
    nave.AddComponent(VoxelsHittable("SpaceShips/DualStriker.vox"))
    nave.transform.position = Vec3(0, 5, -15)
    nave.transform.scale = Vec3(2, 2, 2)

    nave = game.CreateItem()
    nave.AddComponent(VoxelsHittable("SpaceShips/UltravioletIntruder.vox"))
    nave.transform.position += Vec3(0, 5, 10)
    nave.transform.scale = Vec3(5, 5, 5)

    nave = game.CreateItem()
    nave.AddComponent(VoxelsHittable("SpaceShips/RedFighter.vox"))
    nave.transform.position += Vec3(0, 5, 20)
    nave.transform.scale = Vec3(5, 5, 5)

    mapa = game.CreateItem()
    mapa.AddComponent(VoxelsHittable("map.vox"))
    mapa.transform.position += Vec3(0, 0, 0)
    mapa.transform.scale = Vec3(15, -15, 15)

tick = Tick(0.1)
def loop(game: Game):

    if pg.key.get_pressed()[pg.K_b]:
        if tick():
            camera_component.max_bounces += 1
            print(f"Max Bounces: {camera_component.max_bounces}")
    elif pg.key.get_pressed()[pg.K_n]:
        if tick():
            camera_component.max_bounces = max(0, camera_component.max_bounces - 1)
            print(f"Max Bounces: {camera_component.max_bounces}")

    # Makes the light source rotate over time, simulating a day/night cycle
    angle = game.time * 0.00005  # Adjust the multiplier to control the speed of the cycle
    camera_component.light_direction = Vec3(
        math.sin(angle), 0.5, math.cos(angle)
    ).normalize()
