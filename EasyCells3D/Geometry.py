import math
from dataclasses import dataclass


@dataclass
class Vec2[T]:
    x: T
    y: T

    def normalize(self) -> 'Vec2[T]':
        mag = self.magnitude()
        if mag == 0:
            return Vec2.zero()
        return Vec2(self.x / mag, self.y / mag)

    def reflect(self, normal: 'Vec2[T]') -> 'Vec2[T]':
        return self - normal * 2 * self.dot(normal)

    def dot(self, other: 'Vec2[T]') -> T:
        return self.x * other.x + self.y * other.y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: T):
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: T):
        return Vec2(self.x / other, self.y / other)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    @property
    def to_tuple(self):
        return self.x, self.y

    @property
    def to_angle(self):
        return math.atan2(self.y, self.x)

    @staticmethod
    def from_tuple(t: tuple[T, T]) -> 'Vec2[T]':
        return Vec2(t[0], t[1])

    @staticmethod
    def zero() -> 'Vec2[int]':
        return Vec2(0, 0)

    @staticmethod
    def from_angle(angle: float) -> 'Vec2[float]':
        return Vec2(math.cos(angle), math.sin(angle))

    def rotate(self, angle: T) -> 'Vec2[T]':
        return Vec2(
            self.x * math.cos(angle) - self.y * math.sin(angle),
            self.x * math.sin(angle) + self.y * math.cos(angle)
        )

    def distance(self, param: 'Vec2[T]') -> float:
        return math.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2)

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


@dataclass
class Vec3[T]:
    x: T
    y: T
    z: T

    def normalize(self) -> 'Vec3[T]':
        mag = self.magnitude()
        if mag == 0:
            return Vec3.zero()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def reflect(self, normal: 'Vec3[T]') -> 'Vec3[T]':
        return self - normal * 2 * self.dot(normal)

    def dot(self, other: 'Vec3[T]') -> T:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3[T]') -> 'Vec3[T]':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: T):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: T):
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    @property
    def to_tuple(self):
        return self.x, self.y, self.z

    @staticmethod
    def from_tuple(t: tuple[T, T, T]) -> 'Vec3[T]':
        return Vec3(t[0], t[1], t[2])

    @staticmethod
    def zero() -> 'Vec3[int]':
        return Vec3(0, 0, 0)

    def distance(self, param: 'Vec3[T]') -> float:
        return math.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2 + (self.z - param.z) ** 2)

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
