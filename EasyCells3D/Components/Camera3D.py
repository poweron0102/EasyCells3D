import raylibpy as rl

from EasyCells3D.Game import Camera
from EasyCells3D.Geometry import Vec3
from EasyCells3D.Components import Component, Transform, Item

class Renderable3D(Component):
    camera: Camera3D = None

    def render(self):
        pass

    def init(self):
        if self.camera is None:
            self.camera = Camera3D.main
            self.camera.renderables.append(self)

    def on_destroy(self):
        if self in self.camera.renderables:
            self.camera.renderables.remove(self)




class Camera3D(Component, Camera):
    """
    Componente de Câmera 3D usando Raylib.
    Gerencia a visualização e renderização da cena.
    """

    main: Camera3D = None

    def __init__(self, vfov: float = 60.0, projection: int = rl.CAMERA_PERSPECTIVE):
        self.vfov = vfov
        self.projection = projection

        # Inicializa a câmera do Raylib
        self.rl_camera = rl.Camera3D(
            rl.Vector3(0, 0, 0),  # position
            rl.Vector3(0, 0, 1),  # target
            rl.Vector3(0, 1, 0),  # up
            self.vfov,
            self.projection
        )

        if Camera3D.main is None:
            Camera3D.main = self

        self.renderables: list[Renderable3D] = []

        self.global_transform = Transform()

    def init(self):
        self.add_to_game(self.game)

    def on_destroy(self):
        if self in self.game.cameras:
            self.game.cameras.remove(self)

    def loop(self):
        self.global_transform = Transform.Global

    def update_rl_camera(self):
        pos = self.global_transform.position
        self.rl_camera.position = rl.Vector3(pos.x, pos.y, pos.z)

        # Define o alvo baseado na posição + vetor forward
        forward = self.global_transform.forward
        target = pos + forward
        self.rl_camera.target = rl.Vector3(target.x, target.y, target.z)

        up = self.global_transform.up
        self.rl_camera.up = rl.Vector3(up.x, up.y, up.z)

        self.rl_camera.fovy = self.vfov
        self.rl_camera.projection = self.projection

    def render(self):
        self.update_rl_camera()
        rl.begin_mode3d(self.rl_camera)

        for ren in self.renderables:
            ren.render()

        rl.end_mode3d()