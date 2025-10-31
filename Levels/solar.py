import pygame as pg

from EasyCells3D import Game
from EasyCells3D.Components import Camera, Item
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.SphereHittable import SphereHittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material
from UserComponents.ratating_obj import RotatingObj


def init(game: Game):
    camera = game.CreateItem()
    # A luz ambiente é baixa para que o Sol seja a principal fonte de luz.
    camera.AddComponent(Camera(vfov=60, use_cuda=True, ambient_light=Vec3(0.05, 0.05, 0.05)))
    camera.transform.position = Vec3(0, 5, -15)
    camera.transform.forward = Vec3(0, 0, 1)
    camera.AddComponent(FreeCam())

    # Centraliza e oculta o cursor do mouse para melhor controle da câmera
    pg.mouse.set_visible(False)
    pg.event.set_grab(True)

    # Sol
    sun = game.CreateItem()
    sun.AddComponent(SphereHittable(
        radius=2.0,
        material=Material(
            texture_path="Texture/redstone_lamp_on.png", # Simula uma superfície quente
            emissive_color=Vec3(1.0, 0.8, 0.2) # Cor emissiva forte para iluminar a cena
        )
    ))
    # O Sol também pode girar em seu próprio eixo
    sun.AddComponent(RotatingObj(speed=10))

    # --- Terra e Lua ---
    # 1. Pivô de órbita da Terra: um objeto vazio que gira ao redor do Sol
    earth_orbit = game.CreateItem()
    earth_orbit.AddComponent(RotatingObj(speed=30)) # Velocidade orbital da Terra

    # 2. Planeta Terra: filho do pivô de órbita
    earth = earth_orbit.CreateChild()
    earth.transform.position = Vec3(8, 0, 0) # Distância orbital do Sol
    earth.AddComponent(SphereHittable(0.5, Material("Texture/wool_colored_light_blue.png")))
    earth.AddComponent(RotatingObj(speed=60)) # Rotação da Terra em seu próprio eixo

    # 3. Pivô de órbita da Lua: um objeto vazio que gira ao redor da Terra
    moon_orbit = earth.CreateChild()
    moon_orbit.AddComponent(RotatingObj(speed=120)) # Velocidade orbital da Lua

    # 4. Lua: filha do pivô de órbita da Lua
    moon = moon_orbit.CreateChild()
    moon.transform.position = Vec3(1.5, 0, 0) # Distância orbital da Terra
    moon.AddComponent(SphereHittable(0.2, Material("Texture/stone_granite_smooth.png")))

def loop(game: Game):
    pass