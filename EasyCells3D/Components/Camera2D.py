import math
import pyray as rl

from .Component import Component, Transform
from ..Game import Camera
from ..Geometry import Vec2


class Renderable2D(Component):
    camera: 'Camera2D' = None

    def render(self):
        pass

    def init(self):
        if self.camera is None:
            self.camera = Camera2D.main
        if self.camera:
            self.camera.renderables.append(self)

    def on_destroy(self):
        if self.camera and self in self.camera.renderables:
            self.camera.renderables.remove(self)


class Camera2D(Component, Camera):
    """
    Componente de Câmera 2D usando Raylib.
    Gerencia a visualização 2D, zoom e rotação.
    """
    main: 'Camera2D' = None

    def __init__(self, zoom: float = 1.0, render_target: rl.RenderTexture = None):
        self.zoom = zoom
        
        # Inicializa a câmera 2D do Raylib
        self.rl_camera = rl.Camera2D()
        self.rl_camera.zoom = zoom
        self.render_target = render_target
        
        if Camera2D.main is None:
            Camera2D.main = self

        self.renderables: list[Renderable2D] = []
        self.debug_lines: list[tuple[Vec2, Vec2, rl.Color]] = []
        self.debug_polygon: list[tuple[list[Vec2], rl.Color]] = []

    def init(self):
        self.add_to_game(self.game)

    def on_destroy(self):
        if Camera2D.main == self:
            Camera2D.main = None

    def update_rl_camera(self):
        # Offset define o centro da tela (pivô da câmera)
        if self.render_target:
            self.rl_camera.offset = rl.Vector2(self.render_target.texture.width / 2, self.render_target.texture.height / 2)
        else:
            self.rl_camera.offset = rl.Vector2(rl.get_screen_width() / 2, rl.get_screen_height() / 2)
        
        # Target é a posição no mundo onde a câmera está focada
        pos = self.global_transform.position
        self.rl_camera.target = rl.Vector2(pos.x, pos.y)
        
        # Rotação: Converte a rotação Z do Quaternion para graus
        euler = self.global_transform.rotation.to_euler_angles()
        self.rl_camera.rotation = math.degrees(euler.z)
        
        # Zoom combinado com a escala X do transform
        self.rl_camera.zoom = self.zoom * self.global_transform.scale.x

    def render(self):
        self.update_rl_camera()
        
        if self.render_target:
            rl.begin_texture_mode(self.render_target)
            rl.clear_background(rl.BLANK)

        rl.begin_mode_2d(self.rl_camera)

        self.renderables.sort(key=lambda r: r.transform.position.z)

        for ren in self.renderables:
            if ren.enable:
                ren.render()

        for line in self.debug_lines:
            pos1, pos2, color = line
            rl.draw_line(
                int(pos1.x), int(pos1.y),
                int(pos2.x), int(pos2.y),
                color
            )
        self.debug_lines.clear()

        for polygon_data in self.debug_polygon:
            points_vec, color = polygon_data
            for i in range(len(points_vec) - 1):
                p1 = points_vec[i]
                p2 = points_vec[i + 1]
                rl.draw_line(int(p1.x), int(p1.y), int(p2.x), int(p2.y), color)
        self.debug_polygon.clear()

        rl.end_mode_2d()

        if self.render_target:
            rl.end_texture_mode()

    def draw_debug_line(self, position1, position2, color=rl.WHITE):
        self.debug_lines.append((position1, position2, color))

    @staticmethod
    def get_mouse_world_position() -> Vec2:
        """Retorna a posição do mouse convertida para coordenadas do mundo."""
        if Camera2D.main is None:
            return Vec2.zero()
        
        mouse_pos = rl.get_mouse_position()
        world_pos = rl.get_screen_to_world_2d(mouse_pos, Camera2D.main.rl_camera)
        return Vec2(world_pos.x, world_pos.y)
