import math

from EasyCells3D.Components import Component
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.Serialization import SerializeField


class RotatingObj(Component):
    speed = SerializeField(default=1.0)

    def init(self):
        self.speed = math.radians(float(self.speed))

    def loop(self):
        angle_this_frame = self.speed * self.game.delta_time
        delta_rotation = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), angle_this_frame)
        self.transform.rotation = (delta_rotation * self.transform.rotation).normalize()
