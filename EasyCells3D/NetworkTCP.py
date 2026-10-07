"""Framed TCP transport with buffered reads and stable peer identifiers."""
import io
import pickle
import select
import socket
import secrets
import threading
from typing import Callable
from collections import deque

SIZE_SIZE = 4
MAX_PACKET = 1_048_576
MAX_PEER_ID = 65535

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
    size = int.from_bytes(buffer[:SIZE_SIZE], "big") if len(buffer) >= SIZE_SIZE else None
    if size is not None and not 0 < size <= MAX_PACKET:
        raise ConnectionError("Invalid packet size")
    if size is None or len(buffer) < SIZE_SIZE + size:
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

class _ConnectionEvents:
    """Deliver worker-thread notifications only when the caller polls on its game thread."""

    def __init__(self, callback):
        self.connect_callback = callback
        self._connect_events = deque()

    def poll_events(self):
        for _ in range(len(self._connect_events)):
            try:
                client_id = self._connect_events.popleft()
            except IndexError:
                break
            self.connect_callback(client_id)


class NetworkServerTCP(_ConnectionEvents):
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable = lambda _: None, *, max_clients: int = 64):
        self.ip, self.port = ip, port
        if not 1 <= max_clients <= MAX_PEER_ID:
            raise ValueError("max_clients must be between 1 and 65535")
        self.max_clients = max_clients
        self.clients = [None]
        self._buffers = {}
        self.peer_tokens = {}
        self.running = True
        super().__init__(connect_callback)
        self.server_socket = socket.socket(socket.AF_INET6 if ip_version == 6 else socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.server_socket.bind((ip, port))
            self.server_socket.listen(min(max_clients, 128))
            self.server_socket.settimeout(.1)
        except OSError:
            self.server_socket.close()
            raise
        self.accept_thread = threading.Thread(target=self.accept_clients, daemon=True)
        self.accept_thread.start()

    def accept_clients(self):
        while self.running:
            try:
                peer, _ = self.server_socket.accept()
                peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                peer.settimeout(.1)
                if len(self._buffers) >= self.max_clients or len(self.clients) > MAX_PEER_ID:
                    peer.close()
                    continue
                cid = len(self.clients)
                token = secrets.token_bytes(32)
                try:
                    _send(peer, (cid, token))
                except OSError:
                    peer.close()
                    continue
                if not self.running:
                    peer.close()
                    break
                self._buffers[cid] = bytearray()
                self.peer_tokens[cid] = token
                self.clients.append(peer)
                self._connect_events.append(cid)

            except OSError:
                if not self.running:
                    break

    def send(self, data, client_id):
        if not 0 < client_id < len(self.clients) or self.clients[client_id] is None:
            return
        try:
            _send(self.clients[client_id], data)
        except OSError:
            self.close_client(client_id)

    def read(self, client_id):
        self.poll_events()
        if not 0 < client_id < len(self.clients) or self.clients[client_id] is None:
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
        if not 0 < client_id < len(self.clients):
            return
        try:
            self._connect_events.remove(client_id)
        except ValueError:
            pass
        peer = self.clients[client_id]
        if peer is not None:
            self.clients[client_id] = None
            peer.close()
        self._buffers.pop(client_id, None)
        self.peer_tokens.pop(client_id, None)

    def close(self):
        self._connect_events.clear()
        self.running = False
        self.server_socket.close()
        self.accept_thread.join(timeout=1)
        for cid in range(1, len(self.clients)):
            self.close_client(cid)
        self._connect_events.clear()

class NetworkClientTCP(_ConnectionEvents):
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable = lambda _: None):
        self.ip, self.port = ip, port
        super().__init__(connect_callback)
        self.id = None
        self.session_token = None
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
            client_id, token = self.block_read()
            if type(client_id) is not int or client_id <= 0 or not isinstance(token, bytes) or len(token) != 32:
                raise ValueError("Invalid TCP session handshake")
            self.session_token = token
            self.id = client_id
            self.server_socket.settimeout(.1)
            self.connected = True
            self._connect_events.append(self.id)
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
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
        self.poll_events()
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
        self._connect_events.clear()
        self.connected = False
        self.session_token = None
        self.server_socket.close()
