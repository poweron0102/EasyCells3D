import pickle
import socket
import unittest

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


if __name__ == '__main__':
    unittest.main()
