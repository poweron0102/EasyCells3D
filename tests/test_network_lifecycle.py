import importlib
import threading
import time
import unittest
from unittest.mock import patch

from EasyCells3D.Components import Item
from EasyCells3D.NetworkComponents import NetworkManager
from EasyCells3D.NetworkTCP import NetworkClientTCP, NetworkServerTCP
from EasyCells3D.scheduler import Scheduler

network_module = importlib.import_module('EasyCells3D.NetworkComponents.NetworkComponent')


class DummyGame:
    def __init__(self):
        self.item_list = []
        self.to_init = []
        self.run_time = 0.0
        self.delta_time = 0.02
        self.scheduler = Scheduler(self)

    def CreateItem(self):
        return Item(self)

    def flush_init(self):
        pending = list(self.to_init)
        self.to_init.clear()
        for callback in pending:
            callback()


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for local connection')
        time.sleep(0.002)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        p = patch.object(NetworkManager, 'instance', None)
        p.start()
        self.addCleanup(p.stop)
        self.game = DummyGame()
        self.addCleanup(self.game.scheduler.clear)

    def test_manager_opens_transports_only_in_init(self):
        with patch.object(network_module, 'TcpTransport') as tcp:
            manager = NetworkManager('127.0.0.1', 5000, True, enable_udp=False)
            tcp.assert_not_called()
            self.assertIsNone(NetworkManager.instance)
            item = self.game.CreateItem()
            item.AddComponent(manager)
            tcp.assert_not_called()
            self.game.flush_init()
            tcp.assert_called_once()
            self.assertIs(NetworkManager.instance, manager)
            manager.on_destroy()
            tcp.return_value.close.assert_called_once()

    def test_tcp_only_ephemeral_port_is_exposed_after_init(self):
        manager = NetworkManager('127.0.0.1', 0, True, enable_udp=False)
        self.game.CreateItem().AddComponent(manager)
        self.addCleanup(manager.on_destroy)
        self.game.flush_init()
        self.assertGreater(manager.port, 0)

    def test_partial_startup_closes_tcp(self):
        with patch.object(network_module, 'TcpTransport') as tcp, patch.object(
                network_module, 'UdpTransport', side_effect=OSError('port busy')):
            manager = NetworkManager('127.0.0.1', 5000, True)
            self.game.CreateItem().AddComponent(manager)
            with self.assertRaises(OSError):
                self.game.flush_init()
            tcp.return_value.close.assert_called_once()
            self.assertIsNone(NetworkManager.instance)


    def test_client_callback_waits_for_main_thread_poll(self):
        calls = []
        server = NetworkServerTCP('127.0.0.1', 0)
        self.addCleanup(server.close)
        client = NetworkClientTCP('127.0.0.1', server.server_socket.getsockname()[1],
                                  connect_callback=lambda _: calls.append(threading.get_ident()))
        self.addCleanup(client.close)
        wait_until(lambda: client.connected or client.error)
        self.assertTrue(client.connected, client.error)
        client.connect_thread.join(1)
        self.assertEqual(calls, [])
        client.read()
        client.read()
        self.assertEqual(calls, [threading.get_ident()])


if __name__ == '__main__':
    unittest.main()
