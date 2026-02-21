import numpy as np

from EasyCells3D.Components import Transform
from EasyCells3D.Components.Camera import Hittable
from EasyCells3D.Geometry import Vec3, Quaternion
from EasyCells3D.Material import Material

sphere_dtype = np.dtype([
    ("radius", np.float32),
    ("material", Material.dtype),
    ("position", Vec3.dtype),
    ("rotation", Quaternion.dtype),
    ("scale", Vec3.dtype)
])

class SphereHittable(Hittable):
    word_position: Transform
    dtype = sphere_dtype

    instances: list['SphereHittable'] = []

    def __init__(self, radius: float = 0.5, material: Material = None):
        super().__init__()
        self.radius = radius
        self.material = material if material is not None else Material()
        SphereHittable.instances.append(self)


    def init(self):
        super().init()
        self.word_position = self.transform

    def loop(self):
        self.word_position = Transform.Global

    def on_destroy(self):
        SphereHittable.instances.remove(self)
        self.on_destroy = lambda: None


    def to_numpy(self):
        return np.array((
            self.radius,
            self.material.to_numpy(),
            self.word_position.position.to_numpy(),
            self.word_position.rotation.to_numpy(),
            self.word_position.scale.to_numpy()
        ), dtype=sphere_dtype)