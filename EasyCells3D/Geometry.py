import taichi as ti
import taichi.math as tm  # Usar o módulo de matemática do Taichi
import numpy as np


# Definir os tipos de struct para uso em anotações de tipo
vec2 = ti.types.struct(x=float, y=float)
vec3 = ti.types.struct(x=float, y=float, z=float)
quaternion = ti.types.struct(w=float, x=float, y=float, z=float)


@ti.dataclass
class Vec2:
    x: float
    y: float

    @ti.func
    def normalize(self) -> vec2:
        mag = self.magnitude()
        # Usar uma pequena tolerância para divisão por zero
        if mag < 1e-9:
            return vec2(0.0, 0.0)
        return vec2(self.x / mag, self.y / mag)

    @ti.func
    def reflect(self, normal: vec2) -> vec2:
        return self - normal * 2 * self.dot(normal)

    @ti.func
    def dot(self, other: vec2) -> float:
        return self.x * other.x + self.y * other.y

    @ti.func
    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    @ti.func
    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    @ti.func
    def __mul__(self, other: float):
        return Vec2(self.x * other, self.y * other)

    @ti.func
    def __truediv__(self, other: float):
        return Vec2(self.x / other, self.y / other)

    @ti.func
    def __neg__(self):
        return Vec2(-self.x, -self.y)

    # Métodos de escopo Python (sem @ti.func)
    @property
    def to_tuple(self):
        return self.x, self.y

    @ti.func
    def to_angle(self):
        return tm.atan2(self.y, self.x)

    @staticmethod
    def from_tuple(t: tuple[float, float]) -> vec2:
        return Vec2(t[0], t[1])

    @staticmethod
    def zero() -> vec2:
        return Vec2(0.0, 0.0)

    @staticmethod
    @ti.func
    def from_angle(angle: float) -> vec2:
        return Vec2(tm.cos(angle), tm.sin(angle))

    @ti.func
    def rotate(self, angle: float) -> vec2:
        return Vec2(
            self.x * tm.cos(angle) - self.y * tm.sin(angle),
            self.x * tm.sin(angle) + self.y * tm.cos(angle)
        )

    @ti.func
    def distance(self, param: vec2) -> float:
        return tm.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2)

    @ti.func
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


@ti.dataclass
class Vec3:
    x: float
    y: float
    z: float

    @ti.func
    def normalize(self) -> vec3:
        mag = self.magnitude()
        if mag < 1e-9:
            return vec3(0.0, 0.0, 0.0)
        return vec3(self.x / mag, self.y / mag, self.z / mag)

    @ti.func
    def reflect(self, normal: vec3) -> vec3:
        return self - normal * 2 * self.dot(normal)

    @ti.func
    def dot(self, other: vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @ti.func
    def cross(self, other: vec3) -> vec3:
        return vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    @ti.func
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @ti.func
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @ti.func
    def __mul__(self, other: float):
        return Vec3(self.x * other, self.y * other, self.z * other)

    @ti.func
    def __truediv__(self, other: float):
        return Vec3(self.x / other, self.y / other, self.z / other)

    @ti.func
    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    # Métodos de escopo Python
    @property
    def to_tuple(self):
        return self.x, self.y, self.z

    @staticmethod
    def from_tuple(t: tuple[float, float, float]) -> vec3:
        return Vec3(t[0], t[1], t[2])

    @staticmethod
    def zero() -> vec3:
        return Vec3(0.0, 0.0, 0.0)

    @ti.func
    def distance(self, param: vec3) -> float:
        return tm.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2 + (self.z - param.z) ** 2)

    @ti.func
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    # Método de escopo Python
    def to_numpy(self, dtype):
        return np.array([self.x, self.y, self.z], dtype=dtype)


@ti.dataclass
class Quaternion:
    w: float  # = 1.0
    x: float  # = 0.0
    y: float  # = 0.0
    z: float  # = 0.0

    @staticmethod
    @ti.func
    def from_axis_angle(axis: vec3, angle: float) -> quaternion:
        """Cria um quaternion a partir de um eixo e um ângulo."""
        axis = axis.normalize()
        half_angle = angle / 2.0
        sin_half = tm.sin(half_angle)
        return Quaternion(
            w=tm.cos(half_angle),
            x=axis.x * sin_half,
            y=axis.y * sin_half,
            z=axis.z * sin_half
        )

    @ti.func
    def __mul__(self, other: quaternion) -> quaternion:
        """Multiplicação de quaternions para combinar rotações."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    @ti.func
    def conjugate(self) -> quaternion:
        """Retorna o conjugado do quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @ti.func
    def rotate_vector(self, v: vec3) -> vec3:
        """Rotaciona um vetor usando o quaternion."""
        q_v = Quaternion(0.0, v.x, v.y, v.z)
        q_rotated = self * q_v * self.conjugate()
        return vec3(q_rotated.x, q_rotated.y, q_rotated.z)

    @ti.func
    def magnitude(self) -> float:
        """Calcula a magnitude do quaternion."""
        return tm.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    @ti.func
    def normalize(self) -> quaternion:
        """Normaliza o quaternion para ter magnitude 1."""
        mag = self.magnitude()
        if mag < 1e-9:
            return Quaternion()  # Retorna quaternion de identidade
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    @ti.func
    def to_euler_angles(self) -> vec3:
        """Converte o quaternion para ângulos de Euler (roll, pitch, yaw)."""
        # Roll (rotação no eixo x)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = tm.atan2(sinr_cosp, cosr_cosp)

        # Pitch (rotação no eixo y)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = 0.0
        if tm.abs(sinp) >= 1:
            pitch = tm.copysign(tm.pi / 2, sinp)
        else:
            pitch = tm.asin(sinp)

        # Yaw (rotação no eixo z)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = tm.atan2(siny_cosp, cosy_cosp)

        return vec3(roll, pitch, yaw)

    @staticmethod
    @ti.func
    def from_euler_angles(dir: vec3) -> quaternion:
        """Cria um quaternion a partir de ângulos de Euler (roll, pitch, yaw)."""
        cy = tm.cos(dir.z * 0.5)
        sy = tm.sin(dir.z * 0.5)
        cp = tm.cos(dir.y * 0.5)
        sp = tm.sin(dir.y * 0.5)
        cr = tm.cos(dir.x * 0.5)
        sr = tm.sin(dir.x * 0.5)

        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )

    # Método de escopo Python
    def to_numpy(self, dtype):
        return np.array([self.w, self.x, self.y, self.z], dtype=dtype)

    @ti.func
    def inverse(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)


@ti.dataclass
class Color:
    r: int
    g: int
    b: int

    # Métodos de escopo Python
    def to_tuple(self) -> tuple[int, int, int]:
        return self.r, self.g, self.b

    @staticmethod
    def from_tuple(t: tuple[int, int, int]) -> 'Color':
        return Color(t[0], t[1], t[2])

    @staticmethod
    def black() -> 'Color':
        return Color(0, 0, 0)

    @staticmethod
    def white() -> 'Color':
        return Color(255, 255, 255)

    @staticmethod
    def red() -> 'Color':
        return Color(255, 0, 0)

    @staticmethod
    def green() -> 'Color':
        return Color(0, 255, 0)

    @staticmethod
    def blue() -> 'Color':
        return Color(0, 0, 255)


@ti.dataclass
class Ray:
    origin: Vec3
    direction: Vec3

    @ti.func
    def point_at(self, t: float) -> Vec3:
        return self.origin + self.direction * t

