from .Collider import Collider, Polygon
from pygame import Rect


class RectCollider(Collider):
    def __init__(self, rect: Rect, mask: int = 1, debug: bool = False):
        rect = Rect(
            rect.left - rect.width / 2,
            rect.top - rect.height / 2,
            rect.width,
            rect.height
        )
        self.rect = rect
        polygon = Polygon([
            [rect.left, rect.top],
            [rect.right, rect.top],
            [rect.right, rect.bottom],
            [rect.left, rect.bottom]
        ])
        super().__init__([polygon], mask, debug)
