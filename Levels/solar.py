from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Components.VoxelsHittable import VoxelsHittable
from EasyCells3D.Geometry import Vec3
from EasyCells3D.Material import Material, Texture
from UserComponents.SpaceShip import SpaceShip
from UserComponents.ratating_obj import RotatingObj

camera: Item


def init(game: Game):
    # --- Configuração da Cena e Variáveis de Controle ---
    speed_scale = 0.5  # Multiplicador global para todas as velocidades de rotação e órbita

    def create_planet(name: str, radius: float, texture: str, distance: float, orbit_speed: float, rotation_speed: float) -> Item:
        """
        Função auxiliar para criar um planeta com sua órbita.
        """
        orbit = game.CreateItem()
        orbit.AddComponent(RotatingObj(speed=orbit_speed * speed_scale))

        planet: Item = orbit.CreateChild()
        planet.transform.position = Vec3(distance, 0, 0)
        planet.AddComponent(SphereHittable(radius, Material(texture)))
        planet.AddComponent(RotatingObj(speed=rotation_speed * speed_scale))
        return planet

    nave = game.CreateItem()
    # --- Câmera e Controles ---
    global camera
    camera = nave.CreateChild()
    camera_component = camera.AddComponent(
        Camera(sky_box=Texture.get("space.jpg"), vfov=60, ambient_light=Vec3(0.05, 0.05, 0.05))
    )

    # --- Criação dos Corpos Celestes ---

    # 1. Sol
    sun = game.CreateItem()
    sun.AddComponent(SphereHittable(
        radius=3.0,
        material=Material(
            texture_path="Texture/redstone_lamp_on.png",  # Simula uma superfície quente
            emissive_color=Vec3(1.0, 0.8, 0.2) # Cor emissiva forte para iluminar a cena
        )
    ))
    sun.AddComponent(RotatingObj(speed=5 * speed_scale))

    # 2. Planetas (usando a função auxiliar)
    # create_planet(nome, raio, textura, distância, vel_órbita, vel_rotação)
    create_planet("Mercury", 0.3, "Texture/stone_granite_smooth.png", 4.5, 47 * speed_scale, 10 * speed_scale)
    create_planet("Venus", 0.5, "Texture/wool_colored_orange.png", 7, 35 * speed_scale, -5 * speed_scale) # Rotação retrógrada lenta

    earth = create_planet("Earth", 0.6, "Texture/wool_colored_light_blue.png", 10, 30 * speed_scale, 60 * speed_scale)
    create_planet("Mars", 0.4, "Texture/brick.png", 14, 24 * speed_scale, 65 * speed_scale)
    create_planet("Jupiter", 1.5, "Texture/dark_oak_log.png", 20, 13 * speed_scale, 150 * speed_scale)
    create_planet("Saturn", 1.3, "Texture/spruce_log.png", 26, 10 * speed_scale, 140 * speed_scale)
    create_planet("Uranus", 1.0, "Texture/wool_colored_cyan.png", 31, 7 * speed_scale, 90 * speed_scale)
    create_planet("Neptune", 0.9, "Texture/wool_colored_light_blue.png", 35, 5 * speed_scale, 85 * speed_scale)

    # 3. Lua (continua como filha da Terra)
    lua = create_planet("Moon", 0.15, "Texture/stone_andesite_smooth.png", 1.5, 120 * speed_scale, 0)
    lua.SetParent(earth)
    lua.transform.scale = Vec3(3, 0.5, 1)

    # --- Naves ---
    #nave = game.CreateItem()
    nave.AddComponent(VoxelsHittable("SpaceShips/DualStriker.vox"))
    nave.transform.position = Vec3(0, 5, 0)
    nave.transform.scale = Vec3(2, 2, 2)
    # Adiciona o componente de controle da nave e passa a câmera para ele
    nave.AddComponent(SpaceShip(camera_component))
    # nave.AddChild(camera)
    camera.transform.position = Vec3(0, 3, 3)
    camera.transform.forward = Vec3(0, -0.5, -1)

    nave = game.CreateItem()
    nave.AddComponent(VoxelsHittable("SpaceShips/UltravioletIntruder.vox"))
    nave.transform.position += Vec3(0, 5, 10)
    nave.transform.scale = Vec3(5, 5, 5)


def loop(game: Game):
    pass