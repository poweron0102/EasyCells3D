import pickle
import socket
import unittest
from unittest.mock import patch
from collections import deque

from EasyCells3D.NetworkComponents import NetworkManager, Protocol

from EasyCells3D.NetworkUDP import NetworkServerUDP, _DatagramSession

from EasyCells3D.NetworkTCP import NetworkClientTCP, NetworkServerTCP, _read
from test_network_lifecycle import wait_until


class TcpLimitsTests(unittest.TestCase):
    def test_maximum_clients_rejects_excess_connection(self):
        server = NetworkServerTCP('127.0.0.1', 0, max_clients=1)
        self.addCleanup(server.close)
        port = server.server_socket.getsockname()[1]
        first = NetworkClientTCP('127.0.0.1', port)
        self.addCleanup(first.close)
        wait_until(lambda: first.connected)
        second = NetworkClientTCP('127.0.0.1', port)
        self.addCleanup(second.close)
        wait_until(lambda: bool(second.error))
        self.assertFalse(second.connected)
        self.assertEqual(len(server._buffers), 1)

    def test_buffered_messages_are_consumed_before_reading_more_bytes(self):
        sender, receiver = socket.socketpair()
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        payload = pickle.dumps(('buffered',), protocol=4)
        frame = len(payload).to_bytes(4, 'big') + payload
        buffer = bytearray(frame * 100)
        sender.sendall(frame * 100)
        previous = len(buffer)
        self.assertEqual(_read(receiver, buffer), ('buffered',))
        self.assertEqual(len(buffer), previous - len(frame))

    def test_shutdown_stops_accept_thread(self):
        server = NetworkServerTCP('127.0.0.1', 0)
        server.close()
        self.assertFalse(server.accept_thread.is_alive())


class UdpLimitsTests(unittest.TestCase):
    def test_oversized_datagram_is_rejected_before_sending(self):
        session = _DatagramSession(1, b'a' * 32, False)
        with self.assertRaises(ValueError):
            session.encode(b'x' * 1200)

    def test_authenticated_receive_rate_is_bounded(self):
        from EasyCells3D.NetworkUDP import MAX_PACKETS_PER_SECOND
        sender = _DatagramSession(1, b'a' * 32, False)
        receiver = _DatagramSession(1, b'a' * 32, True)
        with patch('EasyCells3D.NetworkUDP.time.monotonic', return_value=receiver.receive_window):
            for _ in range(MAX_PACKETS_PER_SECOND):
                receiver.decode(sender.encode('message'))
            with self.assertRaisesRegex(ValueError, 'rate'):
                receiver.decode(sender.encode('excess'))
        with patch('EasyCells3D.NetworkUDP.time.monotonic', return_value=receiver.receive_window + 2):
            self.assertEqual(receiver.decode(sender.encode('resumed')), 'resumed')

    def test_queue_retains_only_latest_messages(self):
        key = b'a' * 32
        client = _DatagramSession(1, key, False)
        packets = iter(client.encode(msg) for msg in ('HANDSHAKE', 1, 2, 3, 4))

        class Socket:
            def bind(self, address): pass
            def settimeout(self, timeout): pass
            def sendto(self, data, address): pass
            def close(self): pass
            def recvfrom(self, size):
                try:
                    return next(packets), ('127.0.0.1', 5000)
                except StopIteration:
                    raise OSError('end')

        with patch('EasyCells3D.NetworkUDP.socket.socket', return_value=Socket()), patch(
                'EasyCells3D.NetworkUDP.threading.Thread'):
            server = NetworkServerUDP('127.0.0.1', 0, peer_token=lambda cid: key if cid == 1 else None,
                                      max_queue=2)
            server.receive_loop()
            self.assertEqual(list(server.msg_queues[1]), [3, 4])
            server.close()


class FrameLimitsTests(unittest.TestCase):
    def test_budget_is_shared_and_busy_peers_do_not_starve_others(self):
        class Transport:
            clients = [None, True, True]
            def __init__(self):
                self.queues = {1: deque(range(10)), 2: deque(range(10))}
            def read(self, peer):
                return self.queues[peer].popleft() if self.queues[peer] else None

        manager = NetworkManager('127.0.0.1', 0, True, max_packets_per_frame=5,
                                 max_packets_per_peer=3, max_frame_ms=100)
        manager.transports = {Protocol.TCP: Transport(), Protocol.UDP: Transport()}
        received = []
        manager.process_packet = lambda packet, peer: received.append(peer)
        manager._server_loop()
        self.assertEqual(len(received), 5)
        self.assertLessEqual(received.count(1), 3)
        self.assertLessEqual(received.count(2), 3)
        self.assertEqual(set(received), {1, 2})

    def test_time_budget_stops_before_next_packet(self):
        class Transport:
            clients = [None, True]
            def read(self, peer): return ('packet',)

        manager = NetworkManager('127.0.0.1', 0, True, max_frame_ms=1)
        manager.transports = {Protocol.TCP: Transport()}
        received = []
        manager.process_packet = lambda *args: received.append(args)
        with patch('EasyCells3D.NetworkComponents.NetworkComponent.perf_counter',
                   side_effect=[0, 0, .002]):
            manager._server_loop()
        self.assertEqual(len(received), 1)


if __name__ == '__main__':
    unittest.main()
