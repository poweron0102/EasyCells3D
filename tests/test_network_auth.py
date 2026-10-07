import pickle
import socket
import unittest

from EasyCells3D.NetworkUDP import _DatagramSession
from EasyCells3D.Transport import TcpTransport, UdpTransport
from test_network_lifecycle import DummyGame, wait_until


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.game = DummyGame()
        self.addCleanup(self.game.scheduler.clear)
        self.tcp = TcpTransport('127.0.0.1', 0, 4, True, lambda _: None)
        self.addCleanup(self.tcp.close)
        self.port = self.tcp._impl.server_socket.getsockname()[1]
        self.udp = UdpTransport('127.0.0.1', self.port, 4, True, lambda _: None, self.tcp)
        self.addCleanup(self.udp.close)
        self.client = TcpTransport('127.0.0.1', self.port, 4, False, lambda _: None)
        self.addCleanup(self.client.close)
        wait_until(lambda: self.client._impl.connected)

    def test_id_only_handshake_cannot_claim_tcp_peer(self):
        attacker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(attacker.close)
        attacker.settimeout(0.1)
        attacker.sendto(pickle.dumps(('HANDSHAKE', self.client._impl.id)), ('127.0.0.1', self.port))
        with self.assertRaises(TimeoutError):
            attacker.recvfrom(65535)
        self.assertEqual(self.udp._impl.client_map, {})

    def test_authenticated_udp_connects_and_exchanges_data(self):
        client = UdpTransport('127.0.0.1', self.port, 4, False, lambda _: None, self.client)
        self.addCleanup(client.close)
        wait_until(lambda: client._impl.id is not None)
        client.send(('hello',))
        wait_until(lambda: bool(self.udp._impl.msg_queues.get(client._impl.id)))
        self.assertEqual(self.udp.read(client._impl.id), ('hello',))
        self.udp.send(('world',), client._impl.id)
        messages = []
        def received():
            value = client.read()
            if value is not None:
                messages.append(value)
            return bool(messages)
        wait_until(received)
        self.assertEqual(messages, [('world',)])


class DatagramTests(unittest.TestCase):
    def test_tampering_wrong_key_and_reflection_are_rejected(self):
        client = _DatagramSession(1, b'a' * 32, False)
        server = _DatagramSession(1, b'a' * 32, True)
        packet = client.encode(('hello',))
        for receiver, data in (
                (server, packet[:-1] + bytes([packet[-1] ^ 1])),
                (_DatagramSession(1, b'b' * 32, True), packet),
                (client, packet)):
            with self.assertRaises(ValueError):
                receiver.decode(data)
        self.assertEqual(server.decode(packet), ('hello',))

    def test_reordered_packets_are_allowed_once_within_window(self):
        client = _DatagramSession(1, b'a' * 32, False)
        server = _DatagramSession(1, b'a' * 32, True)
        packets = [client.encode(i) for i in range(70)]
        self.assertEqual(server.decode(packets[1]), 1)
        self.assertEqual(server.decode(packets[0]), 0)
        with self.assertRaises(ValueError):
            server.decode(packets[0])
        self.assertEqual(server.decode(packets[-1]), 69)
        with self.assertRaises(ValueError):
            server.decode(packets[2])


if __name__ == '__main__':
    unittest.main()
