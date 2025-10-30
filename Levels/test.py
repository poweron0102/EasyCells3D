from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material
from EasyCells3D.scheduler import Tick
import math
import random

import pygame as pg

bola: Item
filha: Item
camera: Item


def init(game: Game):
    global camera
    global bola
    camera = game.CreateItem()
    camera.AddComponent(Camera(vfov=60, use_cuda=True, ambient_light=Vec3(0.7, 0.7, 0.7)))
    camera.transform.position = Vec3(0, 0, -8)
    camera.transform.forward = Vec3(0, 0, 1)
    camera.AddComponent(FreeCam())

    # Centraliza e oculta o cursor do mouse para melhor controle da câmera
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)

    for _ in range(30):
        bola = game.CreateItem()
        texture_path = "Texture/" + random.choice(['brick.png', 'command_block.png', 'dark_oak_log.png', 'enchanting_table_bottom.png', 'redstone_lamp_off.png', 'redstone_lamp_on.png', 'spruce_log.png', 'stone_andesite_smooth.png', 'stone_bricks.png', 'stone_granite_smooth.png', 'stone_slab_top.png', 'tnt_side.png', 'warped_planks.png', 'wool_colored_cyan.png', 'wool_colored_gray.png', 'wool_colored_green.png', 'wool_colored_light_blue.png', 'wool_colored_lime.png', 'wool_colored_magenta.png', 'wool_colored_orange.png'])
        bola.AddComponent(SphereHittable(
            0.5,
            Material(
                texture_path,
                diffuse_color= Vec3(random.random(), random.random(), random.random()),
                specular=random.random(),
                shininess= random.uniform(1, 100),
                #emissive_color= Vec3(random.random(), random.random(), random.random()),
            )
        ))
        bola.transform.position += Vec3(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))

    global filha
    filha = bola.CreateChild()
    filha.AddComponent(SphereHittable(
        0.5,
        Material(
            "Texture/redstone_lamp_on.png",
            diffuse_color= Vec3(random.random(), random.random(), random.random()),
            specular=random.random(),
            shininess= random.uniform(1, 100),
            #emissive_color= Vec3(a := random.random(), a, a),
        )
    ))
    filha.transform.position += Vec3(2, 0, 0)


def loop(game: Game):
    rotation_speed = math.radians(45)  # 45 graus por segundo
    angle_this_frame = rotation_speed * game.delta_time
    delta_rotation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), angle_this_frame)
    bola.transform.rotation = delta_rotation * bola.transform.rotation
