from EasyCells3D import Game
from EasyCells3D.Components import Camera2D, Item, TileMapRenderer, TileMap, Animator2D
from EasyCells3D.Components.TileMap import matrix_from_csv
from EasyCells3D.PhysicsComponents import Rigidbody, TileMapCollider
from UserComponents.platform.Player import load_player

import raylibpy as rl


def init(game: Game):
    camera_item = game.CreateItem()
    camera = camera_item.AddComponent(Camera2D())

    tile_map = game.CreateItem()
    tile_map.AddComponent(TileMap(matrix_from_csv("Pixel Adventure/plataforma_mapa.csv")))
    tile_map.AddComponent(TileMapRenderer("Pixel Adventure/Terrain/Terrain (16x16).png", 16))
    tile_map_collider = tile_map.AddComponent(TileMapCollider(set(range(0, 255)), 16, debug=True))
    tile_map.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))
    tile_map.transform.scale.x = 2
    tile_map.transform.scale.y = 2

    player = load_player(game, "Pixel Adventure/Main Characters/Virtual Guy")
    player.transform.y = -100
    animator = player.GetComponent(Animator2D)

    Rigidbody.start_physics()


def loop(game: Game):
    if rl.is_key_pressed(rl.KEY_R):
        game.new_game("platform")