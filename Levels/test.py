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
    camera.AddComponent(Camera(vfov=40))

    # Centraliza e oculta o cursor do mouse para melhor controle da câmera
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)

    global bola
    # Posiciona as esferas na frente da câmera para serem visíveis no início
    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(0, 0, -5)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(2, 0, -5)

    bola = game.CreateItem()
    bola.AddComponent(SphereHittable(0.5))
    bola.transform.position += Vec3(-2, 0, -5)


def loop(game: Game):
    # --- Rotação da Câmera com o Mouse ---
    mouse_sensitivity = 0.003
    dx, dy = pg.mouse.get_rel()
    dy *= -1

    # Rotação Yaw (esquerda/direita) em torno do eixo Y global
    if dx != 0:
        yaw_rotation = Quaternion.from_axis_angle(Vec3(0, 1, 0), -dx * mouse_sensitivity)
        camera.transform.rotation = yaw_rotation * camera.transform.rotation

    # Rotação Pitch (cima/baixo) em torno do eixo X local (vetor 'right')
    if dy != 0:
        right_vector = camera.transform.rotation.rotate_vector(Vec3(1, 0, 0))
        pitch_rotation = Quaternion.from_axis_angle(right_vector, -dy * mouse_sensitivity)
        camera.transform.rotation = pitch_rotation * camera.transform.rotation

    # Normalizar o quaternion para evitar problemas de ponto flutuante com o tempo
    camera.transform.rotation = camera.transform.rotation.normalize()

    # --- Translação da Câmera com o Teclado (relativo à direção da câmera) ---
    speed = 0.1

    # Obtém os vetores de direção locais da câmera
    forward_vector = -camera.transform.forward
    right_vector = camera.transform.rotation.rotate_vector(Vec3(1, 0, 0))

    movement_input = Vec3(0.0, 0.0, 0.0)

    keys = pg.key.get_pressed()
    if keys[pg.K_w]:
        movement_input += forward_vector
    if keys[pg.K_s]:
        movement_input -= forward_vector
    if keys[pg.K_d]:
        movement_input -= right_vector
    if keys[pg.K_a]:
        movement_input += right_vector

    # Normaliza o vetor de movimento para que a velocidade seja constante em todas as direções
    if movement_input.magnitude() > 0:
        movement_input = movement_input.normalize()

    # Aplica o movimento à posição da câmera
    camera.transform.position += movement_input * speed
