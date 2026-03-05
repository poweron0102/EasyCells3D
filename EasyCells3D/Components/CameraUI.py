from typing import Final

import pyray as rl

from .Component import Component
from ..Game import Camera


class RenderableUI(Component):
    """
    Classe base para objetos que devem ser renderizados pela CameraUI.
    """
    camera: 'CameraUI' = None

    def render(self):
        pass

    def init(self):
        if self.camera is None:
            self.camera = CameraUI.main
        if self.camera:
            self.camera.renderables.append(self)

    def on_destroy(self):
        if self.camera and self in self.camera.renderables:
            self.camera.renderables.remove(self)


class CameraUI(Component, Camera):
    """
    Componente de Câmera para Interface de Usuário (UI).
    Renderiza objetos diretamente em coordenadas de tela (Screen Space),
    sem aplicar transformações de câmera (zoom, pan, rotação de mundo).
    """
    main: 'CameraUI' = None

    @property
    def final_render_target(self):
        if self.render_target:
            return self.render_target
        return self.game.render_target

    def __init__(self, render_target: rl.RenderTexture = None):
        self.render_target = render_target

        if CameraUI.main is None:
            CameraUI.main = self

        self.renderables: list[RenderableUI] = []

    def init(self):
        self.add_to_game(self.game)

    def on_destroy(self):
        if CameraUI.main == self:
            CameraUI.main = None
        if self in self.game.cameras:
            self.game.cameras.remove(self)

    def render(self):
        if self.final_render_target:
            rl.begin_texture_mode(self.final_render_target)
        if self.render_target:
            rl.clear_background(rl.BLANK)

        # Ordena por Z para permitir sobreposição de elementos de UI (layers)
        self.renderables.sort(key=lambda r: r.transform.position.z)

        for ren in self.renderables:
            if ren.enable:
                ren.render()

        if self.final_render_target:
            rl.end_texture_mode()