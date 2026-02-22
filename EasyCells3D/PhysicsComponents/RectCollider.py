from .Collider import Collider, Polygon
import raylibpy as rl


class RectCollider(Collider):
    def __init__(self, rect: rl.Rectangle, mask: int = 1, debug: bool = False):
        # Centraliza o retângulo em relação à posição do objeto (pivô no centro)
        x = rect.x - rect.width / 2
        y = rect.y - rect.height / 2
        w = rect.width
        h = rect.height

        # Define os vértices do polígono (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        polygon = Polygon([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ])
        super().__init__([polygon], mask, debug)
