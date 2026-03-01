import math

import raylibpy as rl

from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera3D import Renderable3D


class Sphere(Renderable3D):
    model: rl.Model = None
    texture: rl.Texture2D = None

    def __init__(self, radius: float = 0.5, color: rl.Color = rl.WHITE, rings: int = 16, slices: int = 16, texture_path: str = None, shader: rl.Shader = None):
        super().__init__()
        self.radius = radius
        self.color = color
        self.rings = rings
        self.slices = slices
        self.texture_path = texture_path
        self.shader = shader

    def init(self):
        super().init()
        mesh = rl.gen_mesh_sphere(1.0, self.rings, self.slices)
        self.model = rl.load_model_from_mesh(mesh)

        if self.texture_path:
            self.texture = rl.load_texture(f"Assets/{self.texture_path}")
            rl.set_material_texture(self.model.materials[0], rl.MATERIAL_MAP_DIFFUSE, self.texture)

    def set_shader_value(self, name: str, value, uniform_type: int):
        if self.shader:
            loc = rl.get_shader_location(self.shader, name)
            rl.set_shader_value(self.shader, loc, value, uniform_type)

    def on_destroy(self):
        super().on_destroy()
        if self.model:
            rl.unload_model(self.model)
        if self.texture:
            rl.unload_texture(self.texture)
        if self.shader:
            rl.unload_shader(self.shader)

    def render(self):
        pos = self.global_transform.position.to_raylib()
        scale_val = self.radius * max(self.global_transform.scale.x, self.global_transform.scale.y, self.global_transform.scale.z)
        final_scale = rl.Vector3(scale_val, scale_val, scale_val)

        # Conversão de Quaternion para Axis-Angle
        q = self.global_transform.rotation
        
        # Proteção para acos (w deve estar entre -1 e 1)
        angle = 2 * math.acos(max(-1.0, min(1.0, q.w)))
        s = math.sqrt(1.0 - q.w * q.w)

        if s < 0.001:
            axis = rl.Vector3(0, 1, 0) # Eixo padrão se não houver rotação significativa
        else:
            axis = rl.Vector3(q.x / s, q.y / s, q.z / s)

        rl.draw_model_ex(self.model, pos, axis, math.degrees(angle), final_scale, self.color)
