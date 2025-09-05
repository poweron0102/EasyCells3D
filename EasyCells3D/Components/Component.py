import traceback
from typing import Type, Tuple
import math
from typing import TYPE_CHECKING

from ..Geometry import Vec2
from ..NewGame import NewGame

if TYPE_CHECKING:
    from ..Game import Game


class Item:
    """
    Class that represents an item that can have components and children.
    """
    transform: 'Transform'
    parent: 'Item | None'

    game: 'Game'

    def __init__(self, game: 'Game', parent=None):
        self.components: dict[Type, Component] = {}
        self.children: set[Item] = set()
        self.transform = Transform()
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
        current_global = Transform.Global

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
            Transform.Global = current_global
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


class Component:
    item: Item
    enable: bool = True

    # debug: bool = False  # Debug mode

    @property
    def transform(self) -> 'Transform':
        return self.item.transform

    @transform.setter
    def transform(self, value: 'Transform') -> None:
        self.item.transform = value

    @property
    def game(self) -> 'Game':
        return self.item.game

    def _inicialize_(self, item: Item):
        self.item = item
        self.game.to_init.append(self.init)

    # abstract method
    def init(self):
        pass

    # abstract method
    def loop(self):
        pass

    def Destroy(self):
        self.on_destroy()
        self.item.components.pop(self.__class__)
        cls = self.__class__
        while cls != Component:
            cls = cls.__bases__[0]
            self.item.components.pop(cls)

    # abstract method
    def on_destroy(self):
        """
        This can be called multiple times if the Component has multiple Inheritances.
        you can use:
        self.on_destroy = lambda: None
        to avoid this
        """
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
    Class that represents a transform with position, rotation and scale.
    """
    Global: 'Transform'

    x: float
    y: float
    z: float

    _scale: float

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value if value > 0.0001 else 0.0001

    _angle: float

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value % (2 * math.pi)

    @property
    def angle_deg(self):
        return math.degrees(self.angle)

    @angle_deg.setter
    def angle_deg(self, value):
        self.angle = math.radians(value)

    @property
    def position(self) -> Vec2[float]:
        return Vec2(self.x, self.y)

    @position.setter
    def position(self, value: Vec2[float]):
        self.x = value.x
        self.y = value.y

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, angle: float = 0, scale: float = 1):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle
        self.scale = scale

    def __add__(self, other):
        return Transform(self.x + other.x, self.y + other.y, self.z + other.z, self.angle + other.angle, self.scale)

    def __sub__(self, other):
        return Transform(self.x - other.x, self.y - other.y, self.z - other.z, self.angle - other.angle, self.scale)

    def __mul__(self, other: float):
        return Transform(self.x * other, self.y * other, self.z, self.angle, self.scale)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.angle == other.angle and self.scale == other.scale

    def __str__(self):
        return f"Transform(x={self.x}, y={self.y}, z={self.z}, angle={self.angle}, scale={self.scale})"

    def clone(self):
        return Transform(self.x, self.y, self.z, self.angle, self.scale)

    def ToGlobal(self, global_transform: 'Transform | None' = None) -> 'Transform':
        global_transform = global_transform if global_transform else Transform.Global
        # Rotate point by Global.angle
        new_x = self.x * math.cos(global_transform.angle) - self.y * math.sin(global_transform.angle)
        new_y = self.x * math.sin(global_transform.angle) + self.y * math.cos(global_transform.angle)

        # Scale point
        new_x *= global_transform.scale
        new_y *= global_transform.scale

        return Transform(
            new_x + global_transform.x,
            new_y + global_transform.y,
            self.z + global_transform.z,
            self.angle + global_transform.angle,
            self.scale * global_transform.scale
        )

    def apply_transform(self, point: Tuple[float, float]) -> Tuple[float, float]:
        # Rotate point by self.angle
        new_x = point[0] * math.cos(self.angle) - point[1] * math.sin(self.angle)
        new_y = point[0] * math.sin(self.angle) + point[1] * math.cos(self.angle)

        # Scale point
        new_x *= self.scale
        new_y *= self.scale

        return new_x + self.x, new_y + self.y

    def SetGlobal(self):
        Transform.Global = self.ToGlobal()
