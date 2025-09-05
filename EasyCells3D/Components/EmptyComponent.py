from ..Components.Component import Component


class EmptyComponent(Component):

    def __init__(self):
        pass

    # abstract method
    def init(self):
        pass

    # abstract method
    def loop(self):
        pass
