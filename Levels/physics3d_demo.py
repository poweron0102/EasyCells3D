"""Demo completa da física 3D (EasyCells3D.PhysicsComponents3D).

Mostra todos os tipos de corpo e os helpers numa cena só. As shapes ficam
visíveis pelo *debug draw* (wireframe), então a cena funciona sem assets de arte.

Controles:
    Mouse + WASD  — voar com a câmera livre (FreeCam)
    Setas         — mover o personagem (cápsula) no plano XZ
    Shift direito — pular
    ESPAÇO        — chover uma caixa/esfera dinâmica
    F             — ligar/desligar o debug draw
    R             — reiniciar a cena
"""
import math
import random

import pyray as rl

from EasyCells3D import Game
from EasyCells3D.Components import Camera3D, Item
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Geometry import Vec3
from EasyCells3D.PhysicsComponents3D import (
    BodyType, BoxShape, BulletPhysicsWorld, CapsuleShape, CharacterController3D,
    PhysicsBody3D, PhysicsDebugRenderer, SphereShape,
)

character: CharacterController3D
platform: Item
spawned: list[Item]


def init(game: Game):
    global character, platform, spawned
    spawned = []

    # Mundo de física — slot único, demolido automaticamente na troca de cena.
    game.physics_world = BulletPhysicsWorld()
    game.physics_world.debug_draw = True

    # Câmera livre + renderer de debug (desenha os wireframes dos colliders).
    cam = game.CreateItem()
    cam.transform.position = Vec3(0, 6, 14)
    cam.AddComponent(Camera3D(vfov=60.0))
    cam.AddComponent(FreeCam())
    cam.AddComponent(PhysicsDebugRenderer())

    # Chão estático.
    ground = game.CreateItem()
    ground.name = "ground"
    ground.transform.position = Vec3(0, 0, 0)
    ground.AddComponent(PhysicsBody3D(
        BoxShape(Vec3(15, 0.5, 15)), body_type=BodyType.STATIC, friction=0.8))

    # Rampa estática (pra testar max_slope do personagem).
    ramp = game.CreateItem()
    ramp.name = "ramp"
    ramp.transform.position = Vec3(-8, 1.2, 0)
    ramp.transform.rotation = rl_quat_axis(Vec3(0, 0, 1), 25)
    ramp.AddComponent(PhysicsBody3D(
        BoxShape(Vec3(4, 0.25, 4)), body_type=BodyType.STATIC, friction=0.9))

    # Plataforma KINEMATIC (vai e volta no eixo X, carregando o que estiver em cima).
    platform = game.CreateItem()
    platform.name = "platform"
    platform.transform.position = Vec3(5, 1.5, 0)
    platform.AddComponent(PhysicsBody3D(
        BoxShape(Vec3(2, 0.25, 2)), body_type=BodyType.KINEMATIC, friction=1.0))

    # Pilha inicial de caixas dinâmicas sobre a plataforma.
    for i in range(3):
        box = game.CreateItem()
        box.transform.position = Vec3(5, 2.5 + i * 1.1, 0)
        box.AddComponent(PhysicsBody3D(
            BoxShape(Vec3(0.5, 0.5, 0.5)), body_type=BodyType.DYNAMIC, mass=1.0))

    # Personagem — cápsula DYNAMIC com rotação travada.
    char_item = game.CreateItem()
    char_item.name = "character"
    char_item.transform.position = Vec3(0, 3, 5)
    char_item.AddComponent(PhysicsBody3D(
        CapsuleShape(radius=0.4, height=1.0),
        body_type=BodyType.DYNAMIC, mass=1.0,
        lock_rotation=True, allow_sleep=False, friction=0.6))
    character = char_item.AddComponent(CharacterController3D(
        move_speed=5.0, jump_height=1.4, max_slope=50.0))


def loop(game: Game):
    # Plataforma kinematic oscilando (movida só pelo Transform).
    platform.transform.position = Vec3(5 + 3 * math.sin(game.run_time), 1.5, 0)

    # Movimento do personagem (world-space; quem chama escolhe os eixos).
    move = Vec3(0, 0, 0)
    if rl.is_key_down(rl.KeyboardKey.KEY_UP):
        move = move + Vec3(0, 0, -1)
    if rl.is_key_down(rl.KeyboardKey.KEY_DOWN):
        move = move + Vec3(0, 0, 1)
    if rl.is_key_down(rl.KeyboardKey.KEY_LEFT):
        move = move + Vec3(-1, 0, 0)
    if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT):
        move = move + Vec3(1, 0, 0)
    character.move(move)
    if rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT_SHIFT):
        character.jump()

    # Chuva de objetos dinâmicos.
    if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE):
        spawn_random(game)

    # Toggle do debug draw.
    if rl.is_key_pressed(rl.KeyboardKey.KEY_F):
        game.physics_world.debug_draw = not game.physics_world.debug_draw

    if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
        game.new_game("physics3d_demo")


def spawn_random(game: Game):
    it = game.CreateItem()
    it.transform.position = Vec3(random.uniform(-3, 3), 8, random.uniform(-3, 3))
    if random.random() < 0.5:
        it.AddComponent(PhysicsBody3D(
            BoxShape(Vec3(0.4, 0.4, 0.4)), body_type=BodyType.DYNAMIC, mass=1.0,
            restitution=0.2))
    else:
        it.AddComponent(PhysicsBody3D(
            SphereShape(0.4), body_type=BodyType.DYNAMIC, mass=1.0,
            restitution=0.6))
    spawned.append(it)


def rl_quat_axis(axis: Vec3, degrees: float):
    from EasyCells3D.Geometry import Quaternion
    import math
    return Quaternion.from_axis_angle(axis, math.radians(degrees))
