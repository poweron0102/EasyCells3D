import unittest
from unittest.mock import patch

from EasyCells3D.NetworkComponents import NetworkComponent, NetworkManager, NetworkVariable
from test_network_rpc import MemoryNetwork


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.manager = MemoryNetwork(0)
        for p in (patch.object(NetworkManager, 'instance', self.manager),
                  patch.dict(NetworkComponent._active_components, clear=True),
                  patch.dict(NetworkVariable._active_variables, clear=True)):
            p.start()
            self.addCleanup(p.stop)

    def test_binding_assigns_component_and_variable_identity(self):
        component = NetworkComponent()
        component.health = NetworkVariable(100)
        component._bind_identity(42, 2)
        component.init()
        self.assertEqual(component.identifier, 42)
        self.assertEqual(component.owner, 2)
        self.assertEqual(component.health.owner, 2)
        self.assertIs(NetworkComponent._active_components[42], component)
        self.assertIs(NetworkVariable._active_variables[component.health.var_id], component.health)
        component.on_destroy()
        self.assertNotIn(42, NetworkComponent._active_components)
        self.assertNotIn(component.health.var_id, NetworkVariable._active_variables)

    def test_duplicate_component_id_does_not_replace_original(self):
        first = NetworkComponent(42, 0)
        first.init()
        other = NetworkComponent(42, 0)
        with self.assertRaises(ValueError):
            other.init()
        other.on_destroy()
        self.assertIs(NetworkComponent._active_components[42], first)

    def test_unbound_component_has_clear_error(self):
        component = NetworkComponent()
        with self.assertRaisesRegex(RuntimeError, 'spawn'):
            component.init()


if __name__ == '__main__':
    unittest.main()
