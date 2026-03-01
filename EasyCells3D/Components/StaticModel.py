from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera3D import Renderable3D

import raylibpy as rl


class StaticModel(Renderable3D):
    """
    Carrega um modelo 3D
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model: rl.Model = None

    def init(self):
        super().init()
        self.model = rl.load_model(f"Assets/{self.model_path}")

    def on_destroy(self):
        super().on_destroy()
        if self.model:
            rl.unload_model(self.model)

    def render(self):
        if not self.model: return
        pos = self.global_transform.position.to_raylib()
        axis = rl.Vector3(0, 1, 0)

        scale = rl.Vector3(self.global_transform.scale.x, self.global_transform.scale.y, self.global_transform.scale.z)

        rl.draw_model_ex(self.model, pos, axis, self.global_transform.angle, scale, rl.WHITE)