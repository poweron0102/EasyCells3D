import raylibpy as rl

from EasyCells3D.Components import Component, Transform
from EasyCells3D.Game import Camera


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

    def __init__(self, vfov: float = 60.0, projection: int = rl.CAMERA_PERSPECTIVE, render_target: rl.RenderTexture = None):
        self.vfov = vfov
        self.projection = projection
        self.render_target = render_target

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

    def init(self):
        self.add_to_game(self.game)

    def on_destroy(self):
        if self in self.game.cameras:
            self.game.cameras.remove(self)

    def update_rl_camera(self):
        self.rl_camera.position = self.global_transform.position.to_raylib()

        # Define o alvo baseado na posição + vetor forward
        forward = self.global_transform.forward
        target = self.global_transform.position + forward
        self.rl_camera.target = target.to_raylib()

        self.rl_camera.up = self.global_transform.up.to_raylib()

        self.rl_camera.fovy = self.vfov
        self.rl_camera.projection = self.projection

    def render(self):
        self.update_rl_camera()

        if self.render_target:
            rl.begin_texture_mode(self.render_target)
            rl.clear_background(rl.BLANK)

        rl.begin_mode3d(self.rl_camera)

        for ren in self.renderables:
            ren.render()

        rl.end_mode3d()

        if self.render_target:
            rl.end_texture_mode()