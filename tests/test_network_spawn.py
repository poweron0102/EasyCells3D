from contextlib import contextmanager
from weakref import WeakValueDictionary
import unittest

from EasyCells3D.NetworkComponents import NetworkComponent, NetworkManager, NetworkTransform, NetworkVariable, Rpc, SendTo, Protocol
from test_network_lifecycle import DummyGame
from test_network_rpc import MemoryNetwork


class Player(NetworkComponent):
    def __init__(self):
        super().__init__()
        self.health = NetworkVariable(100)
        self.spawn_health = None

    def on_network_spawn(self):
        self.spawn_health = self.health.value

    @Rpc(send_to=SendTo.SERVER)
    def observe(self):
        self.rpc_health = self.spawn_health


def player_factory(game, *, x=0):
    item = game.CreateItem()
    item.transform.x = x
    item.AddComponent(Player())
    item.AddComponent(NetworkTransform())
    return item


class Peer:
    def __init__(self, peer_id):
        self.game = DummyGame()
        self.manager = NetworkManager('127.0.0.1', 0, peer_id == 0, enable_udp=False)
        self.game.CreateItem().AddComponent(self.manager)
        self.manager.id = peer_id
        self.manager._tcp_connected = peer_id != 0
        self.output = MemoryNetwork(peer_id)
        self.manager.transports = self.output.transports
        self.manager.send_to_client = self.output.send_to_client
        self.manager.send_to_server = self.output.send_to_server
        self.manager.broadcast = self.output.broadcast
        self.components = {}
        self.variables = WeakValueDictionary()

    @contextmanager
    def active(self):
        old = (NetworkManager.instance, NetworkComponent._active_components, NetworkVariable._active_variables)
        NetworkManager.instance = self.manager
        NetworkComponent._active_components = self.components
        NetworkVariable._active_variables = self.variables
        try:
            yield self.manager
        finally:
            NetworkManager.instance, NetworkComponent._active_components, NetworkVariable._active_variables = old


class SpawnTests(unittest.TestCase):
    def setUp(self):
        self.server, self.client = Peer(0), Peer(1)
        for peer in (self.server, self.client):
            with peer.active() as manager:
                manager.register_prefab('player', player_factory)

    def deliver(self):
        packets = list(self.server.output.sent)
        self.server.output.sent.clear()
        with self.client.active():
            for peer, packet in packets:
                if peer == 1:
                    NetworkManager.process_packet(packet, 0)
            self.client.game.flush_init()

    def test_spawn_assigns_matching_unique_ids_and_initial_state(self):
        with self.server.active() as manager:
            first = manager.spawn('player', x=3, owner=1)
            second = manager.spawn('player', x=5)
            self.server.game.flush_init()
        self.deliver()
        with self.client.active() as manager:
            copies = list(manager.spawned.values())
            self.assertEqual(len(copies), 2)
            copy = next(item for item in copies if item.transform.x == 3)
            self.assertEqual(copy.GetComponent(Player).identifier, first.GetComponent(Player).identifier)
            self.assertEqual(copy.GetComponent(Player).owner, 1)
            self.assertEqual(copy.GetComponent(Player).health.var_id, first.GetComponent(Player).health.var_id)
            self.assertEqual(copy.GetComponent(Player).health.owner, 1)
            self.assertEqual(copy.GetComponent(Player).spawn_health, 100)
            self.assertNotEqual(first.GetComponent(Player).identifier, second.GetComponent(Player).identifier)
            self.assertNotEqual(copy.GetComponent(Player).identifier, copy.GetComponent(NetworkTransform).identifier)

    def test_despawn_and_direct_destroy_remove_remote_objects(self):
        for direct in (False, True):
            with self.subTest(direct=direct):
                with self.server.active() as manager:
                    item = manager.spawn('player')
                    self.server.game.flush_init()
                self.deliver()
                with self.server.active() as manager:
                    item.Destroy() if direct else manager.despawn(item)
                self.deliver()
                self.assertEqual(self.client.manager.spawned, {})
                self.assertEqual(self.client.components, {})
                self.assertEqual(dict(self.client.variables), {})

    def test_client_cannot_spawn_or_despawn(self):
        with self.client.active() as manager:
            with self.assertRaises(PermissionError):
                manager.spawn('player')
            with self.assertRaises(PermissionError):
                manager.despawn(1)

    def test_failed_factory_does_not_leave_items_or_pending_init(self):
        def broken(game):
            root = player_factory(game)
            root.CreateChild().AddComponent(Player())
            raise ValueError('factory failed')

        with self.server.active() as manager:
            manager.register_prefab('broken', broken)
            previous_items = list(self.server.game.item_list)
            previous_init = list(self.server.game.to_init)
            with self.assertRaises(ValueError):
                manager.spawn('broken')
            self.assertEqual(self.server.game.item_list, previous_items)
            self.assertEqual(self.server.game.to_init, previous_init)
            self.assertEqual(manager.spawned, {})

    def test_destroy_before_init_does_not_register_dead_components(self):
        with self.server.active() as manager:
            item = manager.spawn('player')
            manager.despawn(item)
            self.server.game.flush_init()
            self.assertEqual(self.server.components, {})
            self.assertEqual(manager.spawned, {})

    def test_spawn_limit_rejects_extra_object(self):
        with self.server.active() as manager:
            manager.max_spawned_objects = 1
            manager.spawn('player')
            with self.assertRaises(ValueError):
                manager.spawn('player')
            self.assertEqual(len(manager.spawned), 1)

    def test_server_ignores_client_spawn_and_despawn_packets(self):
        with self.server.active() as manager:
            item = manager.spawn('player')
            identifier = next(iter(manager.spawned))
            NetworkManager.process_packet((4, identifier, None, ()), 1)
            NetworkManager.process_packet((3, 999, 'player', (1, (), {}, [], ([], [], True))), 1)
            self.assertEqual(manager.spawned, {identifier: item})

    def test_child_components_keep_matching_ids(self):
        def hierarchy(game):
            root = player_factory(game)
            child = root.CreateChild()
            child.name = 'weapon'
            child.transform.x = 7
            child.AddComponent(NetworkTransform())
            return root

        for peer in (self.server, self.client):
            with peer.active() as manager:
                manager.register_prefab('hierarchy', hierarchy)
        with self.server.active() as manager:
            item = manager.spawn('hierarchy')
            self.server.game.flush_init()
            original = next(iter(item.children)).GetComponent(NetworkTransform)
        self.deliver()
        with self.client.active() as manager:
            copy = next(iter(next(iter(manager.spawned.values())).children))
            self.assertEqual(copy.GetComponent(NetworkTransform).identifier, original.identifier)
            self.assertEqual(copy.transform.x, 7)

    def test_late_join_receives_current_state_not_factory_defaults(self):
        with self.server.active() as manager:
            item = manager.spawn('player', x=3, owner=1)
            self.server.game.flush_init()
            item.transform.x = 12
            item.GetComponent(Player).health.value = 75
            self.server.output.sent.clear()
            manager.server_callback_tcp(1)
        self.deliver()
        with self.client.active() as manager:
            self.assertEqual(len(manager.spawned), 1)
            copy = next(iter(manager.spawned.values()))
            self.assertEqual(copy.transform.x, 12)
            self.assertEqual(copy.GetComponent(Player).health.value, 75)
            self.assertEqual(copy.GetComponent(Player).spawn_health, 75)

    def test_rpc_waits_when_connection_callback_creates_its_target(self):
        manager = self.server.manager
        created = []

        class Transport:
            clients = [None, True]
            def read(inner, peer):
                if created:
                    return None
                item = manager.spawn('player', owner=1)
                created.append(item)
                return (1, item.GetComponent(Player).identifier, 'observe', (), {})

        with self.server.active():
            manager.transports = {Protocol.TCP: Transport()}
            manager._server_loop()
            player = created[0].GetComponent(Player)
            self.assertFalse(hasattr(player, 'rpc_health'))
            self.server.game.flush_init()
            manager._server_loop()
            self.assertEqual(player.rpc_health, 100)

    def test_unknown_factory_is_not_imported_or_executed(self):
        with self.server.active() as manager:
            with self.assertRaises(KeyError):
                manager.spawn('not_registered')


if __name__ == '__main__':
    unittest.main()
