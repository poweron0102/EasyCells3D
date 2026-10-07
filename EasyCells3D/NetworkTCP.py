"""Framed TCP transport with buffered reads and stable peer identifiers."""
import io
import pickle
import select
import socket
import threading
from typing import Callable
from .scheduler import Scheduler

SIZE_SIZE = 4
MAX_PACKET = 1_048_576

class _DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Objects are not allowed in network packets")

def _decode(data):
    return _DataUnpickler(io.BytesIO(data)).load()

def _send(sock, data):
    packet = pickle.dumps(data, protocol=4)
    if len(packet) > MAX_PACKET:
        raise ValueError("Network packet too large")
    sock.sendall(len(packet).to_bytes(SIZE_SIZE, "big") + packet)

def _read(sock, buffer):
    ready, _, _ = select.select([sock], [], [], 0)
    if ready:
        chunk = sock.recv(65536)
        if not chunk:
            raise ConnectionError("Peer disconnected")
        buffer.extend(chunk)
    if len(buffer) < SIZE_SIZE:
        return None
    size = int.from_bytes(buffer[:SIZE_SIZE], "big")
    if not 0 < size <= MAX_PACKET:
        raise ConnectionError("Invalid packet size")
    if len(buffer) < SIZE_SIZE + size:
        return None
    packet = bytes(buffer[SIZE_SIZE:SIZE_SIZE + size])
    del buffer[:SIZE_SIZE + size]
    return _decode(packet)

def _block_read(sock):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Peer disconnected during handshake")
            data.extend(chunk)
        return data
    size = int.from_bytes(exact(SIZE_SIZE), "big")
    if not 0 < size <= MAX_PACKET:
        raise ConnectionError("Invalid packet size")
    return _decode(exact(size))

class NetworkServerTCP:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable = lambda _: None):
        self.ip, self.port = ip, port
        self.clients = [None]
        self._buffers = {}
        self.running = True
        self.connect_callback = connect_callback
        self.scheduler = Scheduler.instance
        self.server_socket = socket.socket(socket.AF_INET6 if ip_version == 6 else socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((ip, port))
        self.server_socket.listen(8)
        self.accept_thread = threading.Thread(target=self.accept_clients, daemon=True)
        self.accept_thread.start()

    def accept_clients(self):
        while self.running:
            try:
                peer, _ = self.server_socket.accept()
                peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                peer.settimeout(.1)
                cid = len(self.clients)
                self.clients.append(peer)
                self._buffers[cid] = bytearray()
                self.send(cid, cid)
                self.scheduler.create_task(self._run_connect_callback(cid))
            except OSError:
                if not self.running:
                    break

    async def _run_connect_callback(self, client_id):
        self.connect_callback(client_id)

    def send(self, data, client_id):
        if client_id >= len(self.clients) or self.clients[client_id] is None:
            return
        try:
            _send(self.clients[client_id], data)
        except OSError:
            self.close_client(client_id)

    def read(self, client_id):
        if client_id >= len(self.clients) or self.clients[client_id] is None:
            return None
        try:
            data = _read(self.clients[client_id], self._buffers.setdefault(client_id, bytearray()))
            if data == "close":
                self.close_client(client_id)
                return None
            return data
        except (OSError, ValueError, pickle.UnpicklingError, EOFError):
            self.close_client(client_id)
            return None

    def block_read(self, client_id):
        return _block_read(self.clients[client_id])

    def broadcast(self, data):
        for cid in range(1, len(self.clients)):
            self.send(data, cid)

    def close_client(self, client_id):
        peer = self.clients[client_id]
        if peer is not None:
            self.clients[client_id] = None
            peer.close()
        self._buffers.pop(client_id, None)

    def close(self):
        self.running = False
        for cid in range(1, len(self.clients)):
            self.close_client(cid)
        self.server_socket.close()

class NetworkClientTCP:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable = lambda _: None):
        self.ip, self.port = ip, port
        self.connect_callback = connect_callback
        self.id = None
        self.error = ""
        self.connected = False
        self._buffer = bytearray()
        self.server_socket = socket.socket(socket.AF_INET6 if ip_version == 6 else socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.settimeout(5)
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.connect_thread = threading.Thread(target=self.connect, daemon=True)
        self.connect_thread.start()

    def connect(self):
        try:
            self.server_socket.connect((self.ip, self.port))
            self.id = int(self.block_read())
            self.server_socket.settimeout(.1)
            self.connected = True
            self.connect_callback(self.id)
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            self.error = str(exc)
            self.server_socket.close()

    def send(self, data):
        if not self.connected:
            return
        try:
            _send(self.server_socket, data)
        except OSError as exc:
            self.error = str(exc)
            self.close()

    def read(self):
        if not self.connected:
            return None
        try:
            data = _read(self.server_socket, self._buffer)
            if data == "close":
                self.close()
                return None
            return data
        except (OSError, ValueError, pickle.UnpicklingError, EOFError) as exc:
            self.error = str(exc)
            self.close()
            return None

    def block_read(self):
        return _block_read(self.server_socket)

    def close(self):
        self.connected = False
        self.server_socket.close()
