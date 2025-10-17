from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material
from EasyCells3D.scheduler import Tick
import random

import pygame as pg

bola: Item
camera: Item

tick: Tick = Tick(1)
mouse_on: bool = False


def init(game: Game):
    global camera
    camera = game.CreateItem()
    camera.AddComponent(Camera(vfov=60, use_cuda=True))
    camera.transform.position = Vec3(0, 0, -8)  # Posição inicial da câmera
    camera.transform.forward = Vec3(0, 0, 1) # Aponta para a origem

    # Centraliza e oculta o cursor do mouse para melhor controle da câmera
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)

    global bola
    # Posiciona as esferas na frente da câmera para serem visíveis no início
    # bola = game.CreateItem()
    # bola.AddComponent(SphereHittable(0.5))
    # bola.transform.position += Vec3(0, 0, -5)
    #
    # bola = game.CreateItem()
    # bola.AddComponent(SphereHittable(0.5))
    # bola.transform.position += Vec3(2, 0, -5)
    #
    # bola = game.CreateItem()
    # bola.AddComponent(SphereHittable(0.5))
    # bola.transform.position += Vec3(-2, 0, -5)

    # Spown 30 spheres in random positions near 0, 0, 0
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
                emissive_color= Vec3(random.random(), random.random(), random.random()),
            )
        ))
        bola.transform.position += Vec3(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))


def loop(game: Game):
    # --- Rotação da Câmera com o Mouse ---

    global mouse_on
    if not mouse_on:
        mouse_sensitivity = 0.2
        dx, dy = pg.mouse.get_rel()

        # Rotação Yaw (esquerda/direita) em torno do eixo Y global
        if dx != 0:
            yaw_rotation = Quaternion.from_axis_angle(Vec3(0, 1, 0), -dx * mouse_sensitivity * game.delta_time)
            camera.transform.rotation = yaw_rotation * camera.transform.rotation

        # Rotação Pitch (cima/baixo) em torno do eixo X local (vetor 'right')
        if dy != 0:
            right_vector = camera.transform.right
            pitch_rotation = Quaternion.from_axis_angle(right_vector, -dy * mouse_sensitivity * game.delta_time)

            # --- INÍCIO DA MODIFICAÇÃO ---
            # Calcula a rotação potencial para verificação
            potential_rotation = pitch_rotation * camera.transform.rotation

            # Para evitar que a câmera vire de cabeça para baixo (e inverta os controles),
            # verificamos se o vetor 'up' da câmera continua apontando para cima no espaço do mundo.
            # Se o componente Y do vetor 'up' se tornar negativo, a câmera virou.
            if potential_rotation.rotate_vector(Vec3(0, 1, 0)).y > 0:
                camera.transform.rotation = potential_rotation
            # --- FIM DA MODIFICAÇÃO ---

        # Normalizar o quaternion para evitar problemas de ponto flutuante com o tempo
        camera.transform.rotation = camera.transform.rotation.normalize()

    #print(camera.transform.forward)

    # --- Translação da Câmera com o Teclado (relativo à direção da câmera) ---
    speed = 4

    # Obtém os vetores de direção locais da câmera
    forward_vector = camera.transform.forward
    right_vector = camera.transform.right # Usando a property para consistência

    movement_input = Vec3(0.0, 0.0, 0.0)

    keys = pg.key.get_pressed()
    if keys[pg.K_w]:
        movement_input += forward_vector
    if keys[pg.K_s]:
        movement_input -= forward_vector
    if keys[pg.K_a]:
        movement_input -= right_vector
    if keys[pg.K_d]:
        movement_input += right_vector

    # Normaliza o vetor de movimento para que a velocidade seja constante em todas as direções
    if movement_input.magnitude() > 0:
        movement_input = movement_input.normalize()

    # Aplica o movimento à posição da câmera
    camera.transform.position += movement_input * (speed * game.delta_time)

    if keys[pg.K_ESCAPE] and tick():
        mouse_on = not mouse_on
        pg.mouse.set_visible(mouse_on)
        pg.event.set_grab(not mouse_on)
