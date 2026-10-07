import socket
import hmac
import struct
from typing import Callable, Any
import pickle
import threading
import select
import time
from collections import deque
from EasyCells3D.NetworkTCP import _decode, _ConnectionEvents, MAX_PEER_ID

MAX_DATAGRAM = 1200
MAX_PACKETS_PER_SECOND = 1000


class _DatagramSession:
    """Authenticate direction, peer and sequence before decoding the payload."""
    header = struct.Struct("!IQ")

    def __init__(self, peer_id, token, is_server):
        self.peer_id, self.token = peer_id, token
        self.outgoing = b"S" if is_server else b"C"
        self.incoming = b"C" if is_server else b"S"
        self.sequence = 0
        self.latest = 0
        self.seen = 0
        self.receive_window = time.monotonic()
        self.received = 0
        self.lock = threading.Lock()

    def encode(self, message):
        with self.lock:
            self.sequence += 1
            body = self.header.pack(self.peer_id, self.sequence) + pickle.dumps(message, protocol=4)
            if len(body) + 32 > MAX_DATAGRAM:
                raise ValueError("UDP packet exceeds 1200 bytes; use TCP for larger messages")
            return body + hmac.digest(self.token, self.outgoing + body, "sha256")

    def decode(self, packet):
        if not self.header.size + 32 <= len(packet) <= MAX_DATAGRAM:
            raise ValueError("Invalid UDP envelope")
        body, tag = packet[:-32], packet[-32:]
        if not hmac.compare_digest(tag, hmac.digest(self.token, self.incoming + body, "sha256")):
            raise ValueError("Invalid UDP authentication")
        now = time.monotonic()
        if now - self.receive_window >= 1:
            self.receive_window, self.received = now, 0
        if self.received >= MAX_PACKETS_PER_SECOND:
            raise ValueError("UDP receive rate exceeded")
        self.received += 1
        peer_id, sequence = self.header.unpack_from(body)
        if peer_id != self.peer_id or sequence == 0:
            raise ValueError("Invalid UDP peer or sequence")
        age = self.latest - sequence
        if age >= 64 or (age >= 0 and self.seen & (1 << age)):
            raise ValueError("Replayed UDP packet")
        message = _decode(body[self.header.size:])
        if sequence > self.latest:
            self.seen = (self.seen << min(sequence - self.latest, 64)) & ((1 << 64) - 1)
            self.latest = sequence
            age = 0
        self.seen |= 1 << age
        return message


class NetworkServerUDP(_ConnectionEvents):
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None,
                 peer_token: Callable[[int], bytes | None] | None = None, *, max_queue: int = 128):
        self.ip = ip
        self.port = port
        self.ip_version = ip_version
        if peer_token is None:
            raise ValueError("UDP requires a TCP session token provider")
        self.peer_token = peer_token
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        self.max_queue = max_queue
        self._sessions = {}

        # Clients list stores tuples of (ip, port)
        # Index 0 is reserved/None to match original 1-based logic
        self.clients: list[tuple[str, int] | None] = [None]

        # Maps (ip, port) -> client_id for fast lookup
        self.client_map: dict[tuple[str, int], int] = {}

        # UDP requires us to buffer messages per client manually
        self.msg_queues: dict[int, deque] = {}

        if ip_version == 6:
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        elif ip_version == 4:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError("Invalid IP version")

        self.server_socket.bind((self.ip, self.port))
        self.server_socket.settimeout(.1)
        super().__init__(connect_callback)
        print(f"UDP Server running on {(self.ip, self.port)}")

        self.running = True
        self.recv_thread = threading.Thread(target=self.receive_loop)
        # ends the thread when the main program ends
        self.recv_thread.daemon = True
        self.recv_thread.start()

    def receive_loop(self):
        """
        Background thread that constantly reads UDP packets from the single server socket
        and routes them to the correct client's message queue.
        """
        while self.running:
            try:
                # 65535 is the theoretical max UDP packet size.
                # This blocks until data is received.
                data, addr = self.server_socket.recvfrom(MAX_DATAGRAM + 1)

                if not _DatagramSession.header.size + 32 <= len(data) <= MAX_DATAGRAM:
                    continue
                client_id, _ = _DatagramSession.header.unpack_from(data)
                if not 0 < client_id <= MAX_PEER_ID:
                    continue
                token = self.peer_token(client_id)
                if token is None:
                    continue
                session = self._sessions.get(client_id)
                if session is None or session.token != token:
                    session = _DatagramSession(client_id, token, True)
                msg = session.decode(data)
                if msg == "HANDSHAKE":
                    previous = self._sessions.get(client_id)
                    if previous is not None and previous.token != token:
                        self.close_client(client_id)
                    if addr in self.client_map and self.client_map[addr] != client_id:
                        continue
                    if client_id < len(self.clients) and self.clients[client_id] not in (None, addr):
                        continue
                    if client_id not in self._sessions:
                        self.clients.extend([None] * max(0, client_id + 1 - len(self.clients)))
                        self.clients[client_id] = addr
                        self.client_map[addr] = client_id
                        self.msg_queues[client_id] = deque(maxlen=self.max_queue)
                        self._sessions[client_id] = session
                        self._connect_events.append(client_id)
                    self.send(client_id, client_id)
                elif self.client_map.get(addr) == client_id and self._sessions.get(client_id) is session:
                    self.msg_queues[client_id].append(msg)

            except (pickle.UnpicklingError, EOFError, ValueError, TypeError):
                continue
            except TimeoutError:
                continue
            except ConnectionResetError:
                continue  # Windows may report a departed UDP peer on the shared socket.
            except OSError:
                # Socket likely closed
                break

    def send(self, data: object, client_id: int):
        if not 0 < client_id < len(self.clients) or self.clients[client_id] is None:
            return

        addr = self.clients[client_id]
        serialized = self._sessions[client_id].encode(data)
        try:
            # UDP preserves boundaries, so we don't need a size header.
            # However, data must fit in one packet (approx 64k).
            self.server_socket.sendto(serialized, addr)
        except Exception as e:
            print(f"Send error to {client_id}: {e}")

    def read(self, client_id: int) -> Any:
        self.poll_events()
        # Check if we have buffered messages for this client
        queue = self.msg_queues.get(client_id)
        if queue:
            return queue.popleft()
        return None

    def block_read(self, client_id: int, timeout: float = 5.0) -> Any:
        # Simple polling wait since we rely on the background thread
        deadline = time.monotonic() + timeout
        while self.running and client_id in self.msg_queues:
            if time.monotonic() >= deadline:
                raise TimeoutError("UDP read timed out")
            val = self.read(client_id)
            if val is not None:
                return val
            time.sleep(0.01)
        raise ConnectionError("UDP peer is closed")

    def broadcast(self, data: object):
        for i in range(1, len(self.clients)):
            if self.clients[i] is not None:
                self.send(data, i)

    def close(self):
        self._connect_events.clear()
        self.running = False
        for i in range(1, len(self.clients)):
            if self.clients[i] is not None:
                self.send("close", i)

        # Send a dummy packet to self to unblock the recv loop?
        # Or just close socket (causes OSError in thread, which we catch)
        self.server_socket.close()
        self.recv_thread.join(timeout=1)
        self._connect_events.clear()
        self._sessions.clear()
        self.msg_queues.clear()
        self.client_map.clear()

    def close_client(self, client_id: int):
        if 0 < client_id < len(self.clients) and self.clients[client_id]:
            self.send("close", client_id)
            addr = self.clients[client_id]
            if addr in self.client_map:
                del self.client_map[addr]
            if client_id in self.msg_queues:
                del self.msg_queues[client_id]
            self.clients[client_id] = None
            self._sessions.pop(client_id, None)


class NetworkClientUDP(_ConnectionEvents):
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None,
                 peer_id: Callable[[], int | None] | None = None,
                 peer_token: Callable[[], bytes | None] | None = None):
        self.ip = ip
        self.port = port
        super().__init__(connect_callback)
        if peer_id is None or peer_token is None:
            raise ValueError("UDP requires TCP peer and session token providers")
        self.peer_id = peer_id
        self.peer_token = peer_token
        self._session = None

        if ip_version == 6:
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        elif ip_version == 4:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError("Invalid IP version")

        self.id: int | None = None
        self.running = True
        self.error = ""

        self.connect_thread = threading.Thread(target=self.connect)
        self.connect_thread.daemon = True
        self.connect_thread.start()

    def connect(self):
        # UDP connect() just filters incoming packets to this address
        try:
            self.server_socket.connect((self.ip, self.port))
            self.server_socket.settimeout(.5)
            deadline = time.monotonic() + 5
            while self.peer_id() is None or self.peer_token() is None:
                if not self.running:
                    return
                if time.monotonic() >= deadline:
                    self.error = "TCP handshake timed out"
                    return
                time.sleep(.01)
            self._session = _DatagramSession(self.peer_id(), self.peer_token(), False)
            handshake = "HANDSHAKE"
            for _ in range(10):
                if not self.running:
                    return
                self.send(handshake)
                try:
                    deadline = time.monotonic() + .5
                    while time.monotonic() < deadline:
                        self.server_socket.settimeout(max(.001, deadline-time.monotonic()))
                        try:
                            reply = self.block_read()
                        except (ValueError, pickle.UnpicklingError, EOFError):
                            continue
                        # A lost ACK can leave gameplay datagrams ahead of the retry's ACK.
                        if type(reply) is int and reply == self.peer_id():
                            self.server_socket.settimeout(None)
                            self.id = reply
                            self._connect_events.append(self.id)
                            return
                except (TimeoutError, ConnectionResetError):
                    continue
            self.error = "UDP handshake timed out"
        except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            self.error = str(exc)

    def send(self, data: object):
        if not self.running:
            return
        if self.id is None and data != "HANDSHAKE" and not (
                isinstance(data, tuple) and len(data) == 2 and data[0] == "HANDSHAKE"):
            return
        if self._session is None:
            return
        serialized = self._session.encode(data)
        try:
            self.server_socket.sendall(serialized)
        except OSError as exc:
            self.error = str(exc)

    def read(self) -> Any:
        self.poll_events()
        # Only the handshake thread reads until it has received the peer ID.
        if self.id is None or not self.running:
            return None
        # Use select to check if data is available (non-blocking check)
        ready_to_read, _, _ = select.select([self.server_socket], [], [], 0)
        if not ready_to_read:
            return None  # No data available yet

        try:
            data, _ = self.server_socket.recvfrom(65535)
            message = self._session.decode(data)
            if isinstance(message, int):
                return None  # A repeated handshake acknowledgement.
            if message == "close":
                self.running = False
                return None
            return message
        except (OSError, ValueError, pickle.UnpicklingError, EOFError):
            # UDP ICMP Port Unreachable can sometimes trigger this on Windows
            return None

    def block_read(self) -> Any:
        # Blocking read
        data, _ = self.server_socket.recvfrom(65535)
        return self._session.decode(data)

    def close(self):
        self._connect_events.clear()
        self.running = False
        self.server_socket.close()


# Test
if __name__ == "__main__":
    IP = "localhost"
    PORT = 25765

    is_server = bool(int(input("Server(1) or Client(0): ")))

    if is_server:
        server = NetworkServerUDP(IP, PORT)

        print("Waiting for clients...")
        while len(server.clients) == 1:
            time.sleep(0.1)

        while True:
            # We just read from client 1 for the test
            data = server.read(1)
            if data:
                print(f"Received: {data}")
                response = input("Response: ")
                server.send(response, 1)
            time.sleep(0.1)

    else:
        client = NetworkClientUDP(IP, PORT)

        # Wait for connection to complete
        while client.id is None:
            time.sleep(0.1)

        while True:
            response = input("Data: ")
            client.send(response)
            data = client.block_read()
            print(f"Received: {data}")
