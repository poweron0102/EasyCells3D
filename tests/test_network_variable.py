import unittest
from unittest.mock import patch

from EasyCells3D.NetworkComponents import NetworkManager, NetworkVariable
from EasyCells3D.NetworkComponents.NetworkComponent import VAR_SET
from test_network_rpc import MemoryNetwork


class VariableTests(unittest.TestCase):
    def setUp(self):
        self.network = MemoryNetwork(1)
        self.network._tcp_connected = True
        self.network._check_connection_complete = lambda _: None
        p = patch.object(NetworkManager, 'instance', self.network)
        p.start()
        self.addCleanup(p.stop)
        p = patch.dict(NetworkVariable._active_variables, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_non_owner_write_raises_before_mutation_or_send(self):
        value = NetworkVariable(100, 10, owner=2)
        self.network.sent.clear()
        with self.assertRaises(PermissionError):
            value.value = 999
        self.assertEqual(value.value, 100)
        self.assertEqual(self.network.sent, [])

    def test_owner_shared_and_server_writes_are_allowed(self):
        for owner, require_owner, server in ((1, True, False), (2, False, False), (2, True, True)):
            with self.subTest(owner=owner, shared=not require_owner, server=server):
                self.network.is_server = server
                value = NetworkVariable(100, 10, owner=owner, require_owner=require_owner)
                value.value = 80
                self.assertEqual(value.value, 80)

    def test_remote_unauthorized_write_is_rejected_and_corrected(self):
        self.network.is_server = True
        value = NetworkVariable(100, 10, owner=2)
        with self.assertRaises(PermissionError):
            value.handle_network_update(VAR_SET, (999,), sender_id=1)
        self.assertEqual(value.value, 100)
        self.assertEqual(self.network.sent[-1], (1, (2, 10, VAR_SET, (100,))))

    def test_initial_state_request_waits_for_connection(self):
        self.network._tcp_connected = False
        value = NetworkVariable(0, 10, owner=2)
        self.assertEqual(self.network.sent, [])
        NetworkManager.client_callback_tcp(self.network, 1)
        self.assertEqual(len(self.network.sent), 1)
        self.assertEqual(self.network.sent[0][1], (2, 10, 2, ()))


if __name__ == '__main__':
    unittest.main()
