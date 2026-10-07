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


if __name__ == '__main__':
    unittest.main()
