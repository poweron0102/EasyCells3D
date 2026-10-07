import unittest
from types import SimpleNamespace
from unittest.mock import patch

from EasyCells3D.NetworkComponents import NetworkComponent, NetworkManager, Protocol, Rpc, SendTo


class MemoryNetwork:
    def __init__(self, peer_id):
        self.id = peer_id
        self.is_server = peer_id == 0
        self.sent = []
        self.transports = {p: SimpleNamespace(clients=[None, True, True]) for p in Protocol}

    def send_to_server(self, packet, protocol):
        self.sent.append((0, packet))

    def send_to_client(self, packet, peer_id, protocol):
        self.sent.append((peer_id, packet))

    def broadcast(self, packet, protocol):
        for peer_id in (1, 2):
            self.send_to_client(packet, peer_id, protocol)


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.network = MemoryNetwork(1)
        self.manager_patch = patch.object(NetworkManager, 'instance', self.network)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.registry_patch = patch.dict(NetworkComponent._active_components, clear=True)
        self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)
        self.static_patch = patch.dict(NetworkComponent._static_rpcs)
        self.static_patch.start()
        self.addCleanup(self.static_patch.stop)
        self.instance_patch = patch.object(NetworkComponent, '_static_instance', None)
        self.instance_patch.start()
        self.addCleanup(self.instance_patch.stop)

    def test_keyword_arguments_survive_round_trip(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            def hit(self, damage, *, source='unknown'):
                self.received = (damage, source)

        sender = Player(10, 1)
        receiver = Player(10, 1)
        receiver.init()
        sender.hit(damage=20, source='trap')
        self.network.is_server = True
        self.network.id = 0
        NetworkManager.process_packet(self.network.sent.pop()[1], 1)
        self.assertEqual(receiver.received, (20, 'trap'))

    def test_targeted_call_preserves_keywords(self):
        class Player(NetworkComponent):
            @Rpc()
            def hit(self, *, damage):
                self.received = damage

        player = Player(10, 1)
        player.init()
        self.network.is_server = True
        NetworkManager.call_rpc_on_client(self.network, 1, player.hit, damage=20)
        self.network.is_server = False
        NetworkManager.process_packet(self.network.sent.pop()[1], 0)
        self.assertEqual(player.received, 20)

    def test_destinations_for_every_origin_and_owner(self):
        for destination in SendTo:
            for origin in (0, 1, 2):
                for owner in (0, 1, 2):
                    with self.subTest(destination=destination, origin=origin, owner=owner):
                        executed = []

                        class Player(NetworkComponent):
                            @Rpc(send_to=destination, require_owner=False)
                            def event(self):
                                executed.append(NetworkManager.instance.id)

                        peers = {i: MemoryNetwork(i) for i in (0, 1, 2)}
                        objects = {i: Player(10, owner) for i in peers}
                        NetworkManager.instance = peers[origin]
                        objects[origin].event()
                        if origin:
                            packet = peers[origin].sent.pop()[1]
                            NetworkManager.instance = peers[0]
                            NetworkComponent._active_components[10] = objects[0]
                            NetworkManager.process_packet(packet, origin)
                        for recipient, packet in peers[0].sent:
                            NetworkManager.instance = peers[recipient]
                            NetworkComponent._active_components[10] = objects[recipient]
                            NetworkManager.process_packet(packet, 0)
                        expected = {
                            SendTo.ALL: [0, 1, 2],
                            SendTo.SERVER: [0],
                            SendTo.CLIENTS: [1, 2],
                            SendTo.OWNER: [owner],
                            SendTo.NOT_ME: [i for i in peers if i != origin],
                        }[destination]
                        self.assertEqual(sorted(executed), expected)

    def test_global_rpc_is_received_without_previous_local_call(self):
        received = []

        @Rpc(send_to=SendTo.SERVER, require_owner=False)
        def announce(*, message):
            received.append(message)

        self.network.id = 0
        self.network.is_server = True
        name = announce.__module__ + '.' + announce.__qualname__
        NetworkManager.process_packet((1, 0, name, (), {'message': 'hello'}), 1)
        self.assertEqual(received, ['hello'])

    def test_static_names_are_qualified_and_instance_methods_are_not_global(self):
        received = []

        class First(NetworkComponent):
            @staticmethod
            @Rpc(require_owner=False)
            def announce():
                received.append('first')

            @Rpc()
            def hit(self):
                pass

        class Second(NetworkComponent):
            @Rpc(require_owner=False)
            @staticmethod
            def announce():
                received.append('second')

        self.assertNotIn(First.hit, NetworkComponent._static_rpcs.values())
        for method in (First.announce, Second.announce):
            method()
        packets = list(self.network.sent)
        self.network.is_server = True
        self.network.id = 0
        for _, packet in packets:
            NetworkManager.process_packet(packet, 1)
        self.assertEqual(received, ['first', 'second'])

    def test_nested_rpc_keeps_its_own_destination(self):
        received = []

        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            def outer(self):
                self.inner()

            @Rpc(send_to=SendTo.CLIENTS)
            def inner(self):
                received.append(NetworkManager.instance.id)

        player = Player(10, 1)
        player.init()
        player.outer()
        self.network.is_server = True
        self.network.id = 0
        NetworkManager.process_packet(self.network.sent.pop()[1], 1)
        self.assertEqual(received, [])
        self.assertEqual([peer for peer, _ in self.network.sent], [1, 2])


if __name__ == '__main__':
    unittest.main()
