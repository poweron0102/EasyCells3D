"""Identity and lifecycle of an Item created by NetworkManager.spawn."""
from ..Components.Component import Component
from ..Geometry import Vec3, Quaternion
from .NetworkComponent import NetworkComponent, NetworkVariable


class NetworkObject(Component):
    def __init__(self, manager, identifier, prefab, args, kwargs, owner, items, components, state=None):
        self.manager = manager
        self.identifier = identifier
        self.prefab = prefab
        self.args, self.kwargs = args, kwargs
        self.owner = owner
        self.ready = False
        self._items = items
        self._components = components
        self._initial_state = state

    @staticmethod
    def collect(item):
        items, components = {}, {}

        def visit(current, path):
            items[path] = current
            for component in current._unique_components():
                if isinstance(component, NetworkComponent):
                    cls = type(component)
                    components[(path, f"{cls.__module__}.{cls.__qualname__}")] = component
            children = {}
            for child in current.children:
                key = str(getattr(child, "easycells_id", child.name))
                if key in children:
                    raise ValueError("Network prefab siblings need distinct names or scene IDs")
                children[key] = child
            for key, child in sorted(children.items()):
                visit(child, path + (key,))

        visit(item, ())
        return items, components

    def snapshot(self):
        transforms = []
        variables = []
        for path, item in self._items.items():
            t = item.transform
            transforms.append((path, (t.x, t.y, t.z),
                               (t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z),
                               (t.scale.x, t.scale.y, t.scale.z)))
        for component in self._components.values():
            for variable in vars(component).values():
                if isinstance(variable, NetworkVariable):
                    variables.append((variable.var_id, variable.value))
        identities = [(path, kind, component.identifier)
                      for (path, kind), component in self._components.items()]
        state = (transforms, variables, self.item.destroy_on_load)
        return (self.owner, self.args, self.kwargs, identities, state)

    def init(self):
        if self._initial_state is not None:
            transforms, variables, destroy_on_load = self._initial_state
            for path, position, rotation, scale in transforms:
                item = self._items[tuple(path)]
                item.transform.position = Vec3(*position)
                item.transform.rotation = Quaternion(*rotation)
                item.transform.scale = Vec3(*scale)
            for identifier, value in variables:
                variable = NetworkVariable._active_variables.get(identifier)
                if variable is not None:
                    variable._value = value
                    variable._initial_requested = True
            self.item.destroy_on_load = destroy_on_load
            self._initial_state = None
        self.ready = True
        for component in self._components.values():
            component.on_network_spawn()

    def on_destroy(self):
        self.manager._forget_spawn(self)
