from EasyCells3D import Game, Vec2
from EasyCells3D.Components import Item, Camera2D, Animation2D, Animator2D, Sprite, TileMap
from EasyCells3D.Components import TileMap, TileMapRenderer
from EasyCells3D.Geometry import Vec3
from EasyCells3D.PhysicsComponents import Rigidbody, Collider, RectCollider, TileMapCollider

import pyray as rl

# --- Global variables for easy access in the loop ---
player: Item
player_rg: Rigidbody
caixa: Item
caixa_rg: Rigidbody
camera: Camera2D
tile_map: Item
player_collider: Collider
tile_map_collider: Collider


def init(game: Game):
    global player, player_rg, caixa, caixa_rg, camera, tile_map, player_collider, tile_map_collider

    # --- Player Setup ---
    player = game.CreateItem()
    player.transform.position = Vec2(-30, -80)  # Start position
    player.transform.z = 1  # Render player above other objects

    # Attach camera to player
    #camera = player.AddComponent(Camera())

    camera_item = game.CreateItem()
    camera = camera_item.AddComponent(Camera2D())

    player.AddComponent(Sprite("player32.png", (32, 32)))
    player.AddComponent(
        Animator2D(
            {
                "idle": Animation2D(100, [18]),
                "run": Animation2D(0.1, list(range(0, 7))),
                "death": Animation2D(0.2, list(range(8, 13)), 0, "idle"),
                "rising": Animation2D(0.2, list(range(13, 18)), 0, "boll"),
                "boll": Animation2D(0.2, [17]),
            },
            "idle"
        )
    )
    player_collider = player.AddComponent(RectCollider(rl.Rectangle(0, 0, 32, 32), debug=True, mask=2))
    player_rg = player.AddComponent(Rigidbody(
        mass=10,  # Give player some mass
        drag=2,  # Add some drag to make controls less slippery
        angular_drag=0.9,
        use_gravity=True,
        gravity_scale=1.0,
        restitution=100000,  # bounciness
        is_kinematic=False,
    ))

    # --- Movable Box Setup ---
    caixa = game.CreateItem()
    caixa.AddComponent(Sprite("player24.png", (24, 24))).index_x = 1
    caixa.transform.position = Vec2(80, -100)
    caixa.AddComponent(RectCollider(rl.Rectangle(0, 0, 24, 24), debug=True))
    caixa_rg = caixa.AddComponent(Rigidbody(mass=200, drag=0.2, restitution=100000, use_gravity=True, is_kinematic=False))

    # --- TileMap Setup (as a static environment) ---
    tile_map = game.CreateItem()
    tile_map.AddComponent(TileMap([
        [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
        [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
        [17, 7, 7, 7, 7, 7, 7, 7, 7, 7, 15]
    ]))
    tile_map.AddComponent(TileMapRenderer("RockSet.png", 32))
    # Add a collider for the tilemap
    tile_map_collider = tile_map.AddComponent(TileMapCollider({1, 3, 4, 5, 7, 9, 11, 15, 17}, 32, debug=True))
    # Add a KINEMATIC Rigidbody to make the tilemap a static, unmovable object
    tile_map.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))

    tile_map2 = game.CreateItem()
    tile_map2.AddComponent(TileMap([
        [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    ]))
    tile_map2.AddComponent(TileMapRenderer("RockSet.png", 32))
    tile_map2.AddComponent(TileMapCollider({3, 4, 5}, 32, debug=True))
    tile_map2.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))
    tile_map2.transform.position = Vec2(0, 180)

    Rigidbody.start_physics()


def loop(game: Game):
    # --- Player Controls ---
    # Instead of setting velocity, we apply forces for smoother, physics-based movement.
    move_force = 800.0  # The amount of force to apply for movement
    jump_impulse = 340.0  # An instant force for jumping

    # Horizontal Movement
    move_direction = 0
    if rl.is_key_down(rl.KEY_A):
        move_direction -= 1
        player.GetComponent(Sprite).horizontal_flip = True
    if rl.is_key_down(rl.KEY_D):
        move_direction += 1
        player.GetComponent(Sprite).horizontal_flip = False
    player_rg.add_force(Vec2(move_direction * move_force, 0))

    # Rotate Map
    if rl.is_key_down(rl.KEY_Q):
        tile_map.transform.angle += 5 * game.delta_time
    if rl.is_key_down(rl.KEY_E):
        tile_map.transform.angle -= 5 * game.delta_time

    if rl.is_key_pressed(rl.KEY_R):
        game.new_game("test_rigidbody")  # Restart the game


    # Jumping (example)
    if rl.is_key_pressed(rl.KEY_SPACE):
        # A simple way to check if the player is on the ground is to do a short raycast downwards.
        origin = player.transform.position
        hit_info = Collider.ray_cast_static(origin, Vec2(0, 1), 20, mask=1)  # Raycast 20 pixels down
        if hit_info:
            player_rg.add_impulse(Vec2(0.0, -jump_impulse))  # Apply an upward impulse

    # --- Player Animations ---
    animator = player.GetComponent(Animator2D)
    # Check if moving horizontally
    if abs(player_rg.velocity.x) > 10:
        if animator.current_animation != "run":
            animator.current_animation = "run"
    else:
        animator.current_animation = "idle"

    # --- Debug Raycasting (Example from your original code) ---
    player_pos = player.transform.position
    mouse_pos = Camera2D.get_mouse_world_position()
    direction = (mouse_pos - player_pos).normalize()

    # Draw a line from player to mouse
    # print(f"Player Position: {player_pos}, Mouse Position: {mouse_pos}, Direction: {direction}")
    Camera2D.main.draw_debug_line(player_pos, mouse_pos, rl.GREEN)

    # Perform the raycast
    inf = Collider.ray_cast_static(player_pos, direction, 1000, mask=1)
    if inf:
        col, point, normal = inf
        # Draw a red line to the hit point
        Camera2D.main.draw_debug_line(player_pos, point, rl.RED)
        # Draw the surface normal in blue
        Camera2D.main.draw_debug_line(point, point + normal * 25, rl.BLUE)

    # Drag box with mouse
    if rl.is_mouse_button_down(rl.MOUSE_BUTTON_LEFT) and caixa_rg.collider.is_point_inside(mouse_pos):
        # caixa_rg.add_impulse((mouse_pos - caixa_rg.transform.position) * 1000)
        caixa_rg.is_kinematic = True
        caixa_rg.transform.position = mouse_pos
    else:
        caixa_rg.is_kinematic = False


    # --- Camera Zoom ---
    if rl.is_key_down(rl.KEY_Z):
        camera.zoom -= 1.0 * game.delta_time
    if rl.is_key_down(rl.KEY_X):
        camera.zoom += 1.0 * game.delta_time
    if rl.is_key_pressed(rl.KEY_R):
        camera.init()
