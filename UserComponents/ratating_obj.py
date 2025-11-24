import math

from EasyCells3D.Components import Component
from EasyCells3D.Geometry import Quaternion, Vec3


class RotatingObj(Component):
    def __init__(self, speed: float):
        self.speed = math.radians(speed)

    def loop(self):
        angle_this_frame = self.speed * self.game.delta_time
        delta_rotation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), angle_this_frame)
        self.transform.rotation = (delta_rotation * self.transform.rotation).normalize()