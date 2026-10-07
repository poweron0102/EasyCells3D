import socket
from typing import Callable, Any
import pickle
import threading
import select
import time
from collections import deque
from EasyCells3D.scheduler import Scheduler
from EasyCells3D.NetworkTCP import _decode


class NetworkServerUDP:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None,
                 peer_exists: Callable[[int], bool] | None = None):
        self.ip = ip
        self.port = port
        self.ip_version = ip_version
        self.peer_exists = peer_exists

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
        self.connect_callback = connect_callback
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
                data, addr = self.server_socket.recvfrom(65535)

                msg = _decode(data)
                handshake = msg == "HANDSHAKE" or (
                    isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "HANDSHAKE")
                if handshake:
                    client_id = msg[1] if isinstance(msg, tuple) else self.client_map.get(addr, len(self.clients))
                    if not isinstance(client_id, int) or client_id <= 0:
                        continue
                    if self.peer_exists is not None and (
                            msg == "HANDSHAKE" or not self.peer_exists(client_id)):
                        continue
                    if addr in self.client_map:
                        if self.client_map[addr] == client_id:
                            self.send(client_id, client_id)
                        continue
                    # --- New Client Handling ---
                    if client_id < len(self.clients) and self.clients[client_id] is not None:
                        continue
                    self.clients.extend([None] * max(0, client_id + 1 - len(self.clients)))
                    self.clients[client_id] = addr
                    self.client_map[addr] = client_id
                    self.msg_queues[client_id] = deque()

                    print(f"Connection (UDP) established with {addr}, id: {client_id}")

                    # Send the client their ID
                    self.send(client_id, client_id)

                    Scheduler.instance.create_task(self._run_connect_callback(client_id))

                else:
                    queue = self.msg_queues.get(self.client_map.get(addr))
                    if queue is not None:
                        queue.append(msg)

            except (pickle.UnpicklingError, EOFError, ValueError, TypeError):
                continue
            except ConnectionResetError:
                continue  # Windows may report a departed UDP peer on the shared socket.
            except OSError:
                # Socket likely closed
                break

    async def _run_connect_callback(self, client_id: int):
        self.connect_callback(client_id)

    def send(self, data: object, client_id: int):
        if client_id >= len(self.clients) or self.clients[client_id] is None:
            return

        addr = self.clients[client_id]
        try:
            # UDP preserves boundaries, so we don't need a size header.
            # However, data must fit in one packet (approx 64k).
            serialized = pickle.dumps(data)
            self.server_socket.sendto(serialized, addr)
        except Exception as e:
            print(f"Send error to {client_id}: {e}")

    def read(self, client_id: int) -> Any:
        # Check if we have buffered messages for this client
        if client_id in self.msg_queues and self.msg_queues[client_id]:
            return self.msg_queues[client_id].popleft()
        return None

    def block_read(self, client_id: int) -> Any:
        # Simple polling wait since we rely on the background thread
        while True:
            val = self.read(client_id)
            if val is not None:
                return val
            time.sleep(0.01)

    def broadcast(self, data: object):
        for i in range(1, len(self.clients)):
            if self.clients[i] is not None:
                self.send(data, i)

    def close(self):
        self.running = False
        for i in range(1, len(self.clients)):
            if self.clients[i] is not None:
                self.send("close", i)

        # Send a dummy packet to self to unblock the recv loop?
        # Or just close socket (causes OSError in thread, which we catch)
        self.server_socket.close()
        print("Server closed")

    def close_client(self, client_id: int):
        if client_id < len(self.clients) and self.clients[client_id]:
            self.send("close", client_id)
            addr = self.clients[client_id]
            if addr in self.client_map:
                del self.client_map[addr]
            if client_id in self.msg_queues:
                del self.msg_queues[client_id]
            self.clients[client_id] = None


class NetworkClientUDP:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None,
                 peer_id: Callable[[], int | None] | None = None):
        self.ip = ip
        self.port = port
        self.connect_callback = connect_callback
        self.peer_id = peer_id

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
            while self.peer_id is not None and self.peer_id() is None:
                if not self.running:
                    return
                if time.monotonic() >= deadline:
                    self.error = "TCP handshake timed out"
                    return
                time.sleep(.01)
            handshake = ("HANDSHAKE", self.peer_id()) if self.peer_id is not None else "HANDSHAKE"
            for _ in range(10):
                if not self.running:
                    return
                self.send(handshake)
                try:
                    deadline = time.monotonic() + .5
                    while time.monotonic() < deadline:
                        self.server_socket.settimeout(max(.001, deadline-time.monotonic()))
                        reply = self.block_read()
                        # A lost ACK can leave gameplay datagrams ahead of the retry's ACK.
                        if isinstance(reply, int) and (self.peer_id is None or reply == self.peer_id()):
                            self.id = reply
                            self.server_socket.settimeout(None)
                            self.connect_callback(self.id)
                            return
                except (TimeoutError, ConnectionResetError):
                    continue
            self.error = "UDP handshake timed out"
        except (OSError, ValueError, pickle.UnpicklingError) as exc:
            self.error = str(exc)

    def send(self, data: object):
        if not self.running:
            return
        if self.id is None and data != "HANDSHAKE" and not (
                isinstance(data, tuple) and len(data) == 2 and data[0] == "HANDSHAKE"):
            return
        serialized = pickle.dumps(data)
        try:
            self.server_socket.sendall(serialized)
        except OSError as exc:
            self.error = str(exc)

    def read(self) -> Any:
        # Only the handshake thread reads until it has received the peer ID.
        if self.id is None or not self.running:
            return None
        # Use select to check if data is available (non-blocking check)
        ready_to_read, _, _ = select.select([self.server_socket], [], [], 0)
        if not ready_to_read:
            return None  # No data available yet

        try:
            data, _ = self.server_socket.recvfrom(65535)
            message = _decode(data)
            if isinstance(message, int):
                return None  # A repeated handshake acknowledgement.
            if message == "close":
                self.running = False
                return None
            return message
        except (OSError, pickle.UnpicklingError, EOFError):
            # UDP ICMP Port Unreachable can sometimes trigger this on Windows
            return None

    def block_read(self) -> Any:
        # Blocking read
        data, _ = self.server_socket.recvfrom(65535)
        return _decode(data)

    def close(self):
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
