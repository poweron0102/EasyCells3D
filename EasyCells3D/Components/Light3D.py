from __future__ import annotations

from EasyCells3D.Components import Component
from EasyCells3D.Geometry import Vec3


class Light3D(Component):
    """
    Dados de luz 3D importados de glTF/Blender.

    Este componente ainda nao altera o render. Ele preserva as informacoes de
    KHR_lights_punctual para que o pipeline de shader possa usa-las depois.
    """

    VALID_TYPES = {"directional", "point", "spot"}

    def __init__(
            self,
            light_type: str = "point",
            color: Vec3 | tuple[float, float, float] | list[float] | None = None,
            intensity: float = 1.0,
            range: float | None = None,
            inner_cone_angle: float = 0.0,
            outer_cone_angle: float = 0.7853981633974483,
            name: str | None = None,
    ):
        normalized_type = str(light_type or "point").lower()
        if normalized_type not in self.VALID_TYPES:
            raise ValueError(f"Light3D: tipo de luz invalido: {light_type}")

        self.light_type = normalized_type
        self.color = self._coerce_color(color)
        self.intensity = float(intensity)
        self.range = None if range is None else float(range)
        self.inner_cone_angle = float(inner_cone_angle)
        self.outer_cone_angle = float(outer_cone_angle)
        self.name = name

    def init(self):
        if not hasattr(self.game, "lights"):
            self.game.lights = []
        self.game.lights.append(self)

    def on_destroy(self):
        lights = getattr(self.game, "lights", None)
        if lights and self in lights:
            lights.remove(self)

    @property
    def direction(self) -> Vec3:
        return self.global_transform.forward

    @staticmethod
    def _coerce_color(value) -> Vec3:
        if isinstance(value, Vec3):
            return value
        if value is None:
            return Vec3(1.0, 1.0, 1.0)
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
