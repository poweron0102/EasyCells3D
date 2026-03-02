import math
import traceback
from typing import TYPE_CHECKING
from typing import Type

import raylibpy as rl

from ..Geometry import Vec3, Quaternion, Vec2
from ..NewGame import NewGame

if TYPE_CHECKING:
    from ..Game import Game


class Item:
    """
    Classe que representa um item que pode ter componentes e filhos.
    """
    transform: 'Transform'
    parent: 'Item | None'
    _global_transform: 'Transform'
    game: 'Game'

    @property
    def global_transform(self) -> 'Transform':
        """
        Cached version of the global_transform property.
        it is fast, but is only updated once per frame.
        """
        return self._global_transform

    def global_transform_get(self) -> 'Transform':
        """
        Calculates the global transform dynamically by traversing the parent hierarchy.
        This is not cached and performs calculations on every access.
        """
        if self.parent:
            return self.transform.ToGlobal(self.parent.global_transform)
        return self.transform.clone()

    def global_transform_set(self, value: 'Transform'):
        """
         Sets the local transform based on the given global transform by traversing the parent hierarchy.
         This is not cached and performs calculations on every access.
         Use it only if you really need to set the global transform directly.
         Otherwise, set the local transform and let the global transform be calculated automatically.
         Expensive operation.
         Use Transform.Global on `Component.loop` instead.
        """
        if self.parent:
            parent_global = self.parent.global_transform_get()

            # Calculate relative scale (S_local = S_global / S_parent)
            scale = Vec3(
                value.scale.x / parent_global.scale.x if parent_global.scale.x != 0 else 0,
                value.scale.y / parent_global.scale.y if parent_global.scale.y != 0 else 0,
                value.scale.z / parent_global.scale.z if parent_global.scale.z != 0 else 0
            )

            # Calculate relative rotation (R_local = R_parent^-1 * R_global)
            rotation = parent_global.rotation.inverse() * value.rotation

            # Calculate relative position
            # P_local = (R_parent^-1 * (P_global - P_parent)) / S_parent
            diff = value.position - parent_global.position
            inv_rotated_diff = parent_global.rotation.inverse().rotate_vector(diff)

            position = Vec3(
                inv_rotated_diff.x / parent_global.scale.x if parent_global.scale.x != 0 else 0,
                inv_rotated_diff.y / parent_global.scale.y if parent_global.scale.y != 0 else 0,
                inv_rotated_diff.z / parent_global.scale.z if parent_global.scale.z != 0 else 0
            )

            self.transform = Transform(position, rotation, scale)
        else:
            self.transform = value.clone()

    def global_position_set(self, value: Vec3):
        if self.parent:
            parent_global = self.parent.global_transform_get()
            diff = value - parent_global.position
            inv_rotated_diff = parent_global.rotation.inverse().rotate_vector(diff)

            self.transform.position = Vec3(
                inv_rotated_diff.x / parent_global.scale.x if parent_global.scale.x != 0 else 0,
                inv_rotated_diff.y / parent_global.scale.y if parent_global.scale.y != 0 else 0,
                inv_rotated_diff.z / parent_global.scale.z if parent_global.scale.z != 0 else 0
            )
        else:
            self.transform.position = value

    def global_rotation_set(self, value: Quaternion):
        if self.parent:
            parent_global = self.parent.global_transform_get()
            self.transform.rotation = parent_global.rotation.inverse() * value
        else:
            self.transform.rotation = value

    def global_scale_set(self, value: Vec3):
        if self.parent:
            parent_global = self.parent.global_transform_get()
            self.transform.scale = Vec3(
                value.x / parent_global.scale.x if parent_global.scale.x != 0 else 0,
                value.y / parent_global.scale.y if parent_global.scale.y != 0 else 0,
                value.z / parent_global.scale.z if parent_global.scale.z != 0 else 0
            )
        else:
            self.transform.scale = value

    def __init__(self, game: 'Game', parent=None):
        self.components: dict[Type, Component] = {}
        self.children: set[Item] = set()
        self.transform = Transform()
        self._global_transform = self.transform
        self.parent: 'Item | None' = parent
        self.game = game
        self.destroy_on_load = True
        if parent:
            parent.children.add(self)
        else:
            game.item_list.append(self)

    def CreateChild(self) -> 'Item':
        return Item(self.game, self)

    def AddChild(self, item: 'Item') -> None:
        self.children.add(item)
        if item.parent:
            item.parent.children.remove(item)
        else:
            self.game.item_list.remove(item)
        item.parent = self

    def Destroy(self):
        if self.parent:
            self.parent.children.remove(self)
        else:
            self.game.item_list.remove(self)

        for child in list(self.children):
            child.Destroy()

        for component in list(self.components.keys()):
            self.components[component].on_destroy()

    def update(self):
        if not self.parent:
            Transform.Global = Transform()
        self.transform.SetGlobal()
        self._global_transform = Transform.Global

        for component in list(self.components.keys()):
            if self.components[component].enable:
                try:
                    self.components[component].loop()
                except (KeyboardInterrupt, SystemExit, NewGame) as e:
                    raise e
                except Exception as e:
                    print(f"Error in {self.components[component]}:\n    {e}")
                    traceback.print_exc()

        for child in list(self.children):
            Transform.Global = self.global_transform
            child.update()

    def AddComponent[T: 'Component'](self, component: T) -> T:
        cls = component.__class__
        self.components[cls] = component
        while cls != Component:
            cls = cls.__bases__[0]
            self.components[cls] = component

        component._inicialize_(self)
        return component

    def GetComponent[T: Component](self, component: Type[T]) -> T | None:
        try:
            return self.components[component]
        except KeyError:
            for child in self.children:
                resp = child.GetComponent(component)
                if resp:
                    return resp
            return None

    def SetParent(self, parent):
        if self.parent:
            self.parent.children.remove(self)
        if parent:
            parent.children.add(self)
        self.parent = parent




class Component:
    item: Item
    enable: bool = True

    @property
    def transform(self) -> 'Transform':
        return self.item.transform

    @transform.setter
    def transform(self, value: 'Transform') -> None:
        self.item.transform = value

    @property
    def global_transform(self) -> 'Transform':
        return self.item.global_transform

    @property
    def game(self) -> 'Game':
        return self.item.game

    def _inicialize_(self, item: Item):
        self.item = item
        self.game.to_init.append(self.init)

    def init(self):
        pass

    def loop(self):
        pass

    def Destroy(self):
        self.on_destroy()
        self.item.components.pop(self.__class__)
        cls = self.__class__
        while cls != Component:
            cls = cls.__bases__[0]
            self.item.components.pop(cls)

    def on_destroy(self):
        pass

    def GetComponent[T: Component](self, component: Type[T]) -> T | None:
        return self.item.GetComponent(component)

    def CalculateGlobalTransform(self) -> 'Transform':
        """
        Calculate the global transform of the item.
        Expensive operation.
        Use Transform.Global on `Component.loop` instead.
        """
        result = Transform()
        parents: list[Item] = []
        current = self.item
        while current:
            parents.append(current)
            current = current.parent

        for i in range(len(parents) - 1, -1, -1):
            result = parents[i].transform.ToGlobal(result)

        return result


class Transform:
    """
    Classe que representa uma transformação 3D com posição, rotação (como um Quaternion) e escala.
    """
    Global: 'Transform'

    _position: Vec3

    @property
    def position(self):
        return self._position

    @property
    def positionVec2(self):
        return Vec2(self.position.x, self.position.y)

    @positionVec2.setter
    def positionVec2(self, value: Vec2):
        self.position = Vec3(value.x, value.y, self._position.z)

    @position.setter
    def position(self, value: Vec3 | Vec2):
        if isinstance(value, Vec2):
            value = Vec3(value.x, value.y, self._position.z)
        self._position = value

    @property
    def x(self):
        return self._position.x

    @x.setter
    def x(self, value):
        self._position.x = value

    @property
    def y(self):
        return self._position.y

    @y.setter
    def y(self, value):
        self._position.y = value

    @property
    def z(self):
        return self._position.z

    @z.setter
    def z(self, value):
        self._position.z = value

    @property
    def angle(self):
        return math.degrees(self.rotation.to_euler_angles().z)

    @angle.setter
    def angle(self, value):
        self.rotation = Quaternion.from_axis_angle(Vec3(0.0, 0.0, 1.0), math.radians(value))

    def __init__(self, position: Vec3 = None, rotation: Quaternion = None, scale: Vec3 = None):
        self.position = position if position is not None else Vec3(0.0, 0.0, 0.0)
        self.rotation = rotation if rotation is not None else Quaternion()
        self.scale = scale if scale is not None else Vec3(1.0, 1.0, 1.0)

    def __eq__(self, other):
        if not isinstance(other, Transform):
            return NotImplemented
        return self.position == other.position and self.rotation == other.rotation and self.scale == other.scale

    def __str__(self):
        pos_str = f"({self.position.x:.2f}, {self.position.y:.2f}, {self.position.z:.2f})"
        rot_str = f"({self.rotation.w:.2f}, {self.rotation.x:.2f}, {self.rotation.y:.2f}, {self.rotation.z:.2f})"
        scl_str = f"({self.scale.x:.2f}, {self.scale.y:.2f}, {self.scale.z:.2f})"
        return f"Transform(position={pos_str}, rotation={rot_str}, scale={scl_str})"

    def clone(self):
        return Transform(
            Vec3(self.position.x, self.position.y, self.position.z),
            Quaternion(self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z),
            Vec3(self.scale.x, self.scale.y, self.scale.z)
        )

    def ToGlobal(self, global_transform: 'Transform | None' = None) -> 'Transform':
        parent_global = global_transform if global_transform else Transform.Global

        # Combina rotações multiplicando os quaternions
        global_rotation = parent_global.rotation * self.rotation

        # Combina escalas com multiplicação por componente
        global_scale = Vec3(
            parent_global.scale.x * self.scale.x,
            parent_global.scale.y * self.scale.y,
            parent_global.scale.z * self.scale.z
        )

        # Calcula a posição global
        scaled_local_pos = Vec3(
            self.position.x * parent_global.scale.x,
            self.position.y * parent_global.scale.y,
            self.position.z * parent_global.scale.z
        )
        rotated_pos = parent_global.rotation.rotate_vector(scaled_local_pos)
        global_position = parent_global.position + rotated_pos

        return Transform(global_position, global_rotation, global_scale)

    def SetGlobal(self):
        Transform.Global = self.ToGlobal()

    def to_raylib(self) -> rl.Transform:
        return rl.Transform(
            self.position.to_raylib(),
            self.rotation.to_raylib(),
            self.scale.to_raylib()
        )

    @staticmethod
    def _get_rotation_from_direction(v_from: Vec3, value: Vec3, fallback_axis: Vec3) -> Quaternion:
        """Calcula a rotação necessária para alinhar v_from com value."""
        if value.magnitude() == 0:
            return Quaternion()

        v_to = value.normalize()
        dot = v_from.dot(v_to)

        if abs(dot - 1.0) < 1e-7:
            return Quaternion()

        if abs(dot + 1.0) < 1e-7:
            return Quaternion.from_axis_angle(fallback_axis, math.pi)

        axis = v_from.cross(v_to)
        angle = math.acos(dot)
        return Quaternion.from_axis_angle(axis, angle)

    @property
    def forward(self):
        """Vetor forward (frente) do transform. (-Z)"""
        return self.rotation.rotate_vector(Vec3(0.0, 0.0, -1.0)).normalize()

    @forward.setter
    def forward(self, value: Vec3):
        """
        Define a rotação do transform para que seu vetor 'forward' aponte na direção de 'value'.
        Isso recalcula toda a rotação.
        """
        self.rotation = Transform._get_rotation_from_direction(
            Vec3(0.0, 0.0, -1.0), value, Vec3(0.0, 1.0, 0.0)
        )

    @property
    def up(self):
        """Vetor up (para cima) do transform. (+Y)"""
        return self.rotation.rotate_vector(Vec3(0.0, 1.0, 0.0)).normalize()

    @up.setter
    def up(self, value: Vec3):
        """
        Define a rotação do transform para que seu vetor 'up' aponte na direção de 'value'.
        Isso recalcula toda a rotação.
        """
        self.rotation = Transform._get_rotation_from_direction(
            Vec3(0.0, 1.0, 0.0), value, Vec3(0.0, 0.0, 1.0)
        )

    @property
    def right(self):
        """Vetor right (direita) do transform. (+X)"""
        return self.rotation.rotate_vector(Vec3(1.0, 0.0, 0.0)).normalize()

    @right.setter
    def right(self, value: Vec3):
        """
        Define a rotação do transform para que seu vetor 'right' aponte na direção de 'value'.
        Isso recalcula toda a rotação.
        """
        self.rotation = Transform._get_rotation_from_direction(
            Vec3(1.0, 0.0, 0.0), value, Vec3(0.0, 1.0, 0.0)
        )
