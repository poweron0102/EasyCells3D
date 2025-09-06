import traceback
from typing import TYPE_CHECKING
from typing import Type

from ..Geometry import Vec3, Quaternion
from ..NewGame import NewGame

if TYPE_CHECKING:
    from ..Game import Game


class Item:
    """
    Classe que representa um item que pode ter componentes e filhos.
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

    position: Vec3[float]
    rotation: Quaternion
    scale: Vec3[float]

    def __init__(self, position: Vec3[float] = None, rotation: Quaternion = None, scale: Vec3[float] = None):
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
        return f"Transform3D(position={pos_str}, rotation={rot_str}, scale={scl_str})"

    def clone(self):
        return Transform(
            Vec3(self.position.x, self.position.y, self.position.z),
            Quaternion(self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z),
            Vec3(self.scale.x, self.scale.y, self.scale.z)
        )

    def ToGlobal(self, global_transform: 'Transform3D | None' = None) -> 'Transform':
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
