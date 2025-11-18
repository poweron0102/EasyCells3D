import math
from dataclasses import dataclass
import numpy as np


# typedef struct { float x; float y; } Vec2f;
vec2f_dtype = np.dtype([
    ("x", np.float32),
    ("y", np.float32)
])

# typedef struct { float x; float y; float z; } Vec3f;
vec3f_dtype = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32)
])

# typedef struct { float w; float x; float y; float z; } Quaternion;
quaternion_dtype = np.dtype([
    ("w", np.float32),
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32)
])

# typedef struct { Vec3f origin; Vec3f direction; } Ray;
ray_dtype = np.dtype([
    ("origin", vec3f_dtype),    # 'origin' é do tipo vec3f_dtype
    ("direction", vec3f_dtype)  # 'direction' é do tipo vec3f_dtype
])


@dataclass
class Vec2:
    x: float
    y: float

    dtype = vec2f_dtype

    def normalize(self) -> 'Vec2':
        mag = self.magnitude()
        if mag == 0:
            return Vec2.zero()
        return Vec2(self.x / mag, self.y / mag)

    def reflect(self, normal: 'Vec2') -> 'Vec2':
        return self - normal * 2 * self.dot(normal)

    def dot(self, other: 'Vec2') -> float:
        return self.x * other.x + self.y * other.y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float):
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other: float):
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
    def from_tuple(t: tuple[float, float]) -> 'Vec2':
        return Vec2(t[0], t[1])

    @staticmethod
    def zero() -> 'Vec2':
        return Vec2(0, 0)

    @staticmethod
    def from_angle(angle: float) -> 'Vec2':
        return Vec2(math.cos(angle), math.sin(angle))

    def rotate(self, angle: float) -> 'Vec2':
        return Vec2(
            self.x * math.cos(angle) - self.y * math.sin(angle),
            self.x * math.sin(angle) + self.y * math.cos(angle)
        )

    def distance(self, param: 'Vec2') -> float:
        return math.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2)

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def to_numpy(self):
        return np.array((self.x, self.y), dtype=vec2f_dtype)



@dataclass
class Vec3:
    x: float
    y: float
    z: float

    dtype = vec3f_dtype


    def normalize(self) -> 'Vec3':
        mag = self.magnitude()
        if mag == 0:
            return Vec3.zero()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def reflect(self, normal: 'Vec3') -> 'Vec3':
        return self - normal * 2 * self.dot(normal)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other: float):
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    @property
    def to_tuple(self):
        return self.x, self.y, self.z

    @staticmethod
    def from_tuple(t: tuple[float, float, float]) -> 'Vec3':
        return Vec3(t[0], t[1], t[2])

    @staticmethod
    def zero() -> 'Vec3[int]':
        return Vec3(0, 0, 0)

    def distance(self, param: 'Vec3') -> float:
        return math.sqrt((self.x - param.x) ** 2 + (self.y - param.y) ** 2 + (self.z - param.z) ** 2)

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def to_numpy(self):
        return np.array((self.x, self.y, self.z), dtype=vec3f_dtype)


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    dtype = quaternion_dtype


    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> 'Quaternion':
        """Cria um quaternion a partir de um eixo e um ângulo."""
        axis = axis.normalize()
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        return Quaternion(
            w=math.cos(half_angle),
            x=axis.x * sin_half,
            y=axis.y * sin_half,
            z=axis.z * sin_half
        )



    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Multiplicação de quaternions para combinar rotações."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def conjugate(self) -> 'Quaternion':
        """Retorna o conjugado do quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate_vector(self, v: Vec3) -> Vec3:
        """Rotaciona um vetor usando o quaternion."""
        q_v = Quaternion(0, v.x, v.y, v.z)
        q_rotated = self * q_v * self.conjugate()
        return Vec3(q_rotated.x, q_rotated.y, q_rotated.z)

    def magnitude(self) -> float:
        """Calcula a magnitude do quaternion."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Quaternion':
        """Normaliza o quaternion para ter magnitude 1."""
        mag = self.magnitude()
        if mag == 0:
            return Quaternion()  # Retorna quaternion de identidade
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def to_euler_angles(self) -> Vec3:
        """Converte o quaternion para ângulos de Euler (roll, pitch, yaw)."""
        # Roll (rotação no eixo x)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (rotação no eixo y)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (rotação no eixo z)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vec3(roll, pitch, yaw)

    @staticmethod
    def from_euler_angles(dir: Vec3) -> 'Quaternion':
        """Cria um quaternion a partir de ângulos de Euler (roll, pitch, yaw)."""
        cy = math.cos(dir.z * 0.5)
        sy = math.sin(dir.z * 0.5)
        cp = math.cos(dir.y * 0.5)
        sp = math.sin(dir.y * 0.5)
        cr = math.cos(dir.x * 0.5)
        sr = math.sin(dir.x * 0.5)

        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )


    def to_numpy(self):
        return np.array((self.w, self.x, self.y, self.z), dtype=quaternion_dtype)

    def inverse(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)


@dataclass
class Ray:
    origin: Vec3
    direction: Vec3

    dtype = ray_dtype

    def point_at(self, t: float) -> Vec3:
        return self.origin + self.direction * t
    
    def to_numpy(self):
        return np.array((self.origin.to_numpy(), self.direction.to_numpy()), dtype=ray_dtype)


class DeviceAllocation:
    """
    Typo de dado de memória na GPU.
    """
