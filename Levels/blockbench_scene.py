from pathlib import Path

from EasyCells3D import BlockbenchSceneLoader, Game
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Geometry import Vec3


SCENE_PATH = Path("Assets/Blockbench/scene.ec3d.json")


def init(game: Game):
    camera = game.CreateItem()
    camera.name = "Camera"
    camera.transform.position = Vec3(0, 2, 6)
    camera.AddComponent(Camera3D())
    camera.AddComponent(FreeCam())

    if not SCENE_PATH.exists():
        print(
            f"Blockbench scene not found: {SCENE_PATH}. "
            "Export it from Blockbench with EasyCells3D > Export EasyCells3D Scene."
        )
        return

    BlockbenchSceneLoader(game).load(SCENE_PATH)


def loop(game: Game):
    pass
