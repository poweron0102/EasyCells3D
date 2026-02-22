import socket
from typing import Callable, Any
import pickle
import threading
import select
import time
from collections import deque
from EasyCells.scheduler import Scheduler


class NetworkServerUDP:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None):
        self.ip = ip
        self.port = port
        self.ip_version = ip_version

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

                if addr not in self.client_map:
                    # --- New Client Handling ---
                    self.clients.append(addr)
                    client_id = len(self.clients) - 1
                    self.client_map[addr] = client_id
                    self.msg_queues[client_id] = deque()

                    print(f"Connection (UDP) established with {addr}, id: {client_id}")

                    # Send the client their ID
                    self.send(client_id, client_id)

                    Scheduler.instance.add(0, lambda: self.connect_callback(client_id))

                    # If the packet contains real data (not just a handshake), queue it
                    try:
                        msg = pickle.loads(data)
                        if msg != "HANDSHAKE":
                            self.msg_queues[client_id].append(msg)
                    except:
                        pass  # Ignore decode errors on handshake

                else:
                    # --- Existing Client Handling ---
                    client_id = self.client_map[addr]
                    try:
                        msg = pickle.loads(data)
                        # Filter out repeated handshakes if client retries
                        if msg != "HANDSHAKE":
                            self.msg_queues[client_id].append(msg)
                    except Exception as e:
                        print(f"Error decoding data from {client_id}: {e}")

            except OSError:
                # Socket likely closed
                break

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
                 connect_callback: Callable[[int], None] = lambda x: None):
        self.ip = ip
        self.port = port
        self.connect_callback = connect_callback

        if ip_version == 6:
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        elif ip_version == 4:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise ValueError("Invalid IP version")

        self.id: int | None = None

        self.connect_thread = threading.Thread(target=self.connect)
        self.connect_thread.daemon = True
        self.connect_thread.start()

    def connect(self):
        # UDP connect() just filters incoming packets to this address
        self.server_socket.connect((self.ip, self.port))

        # UDP is stateless, so the server doesn't know we exist yet.
        # We must send a Handshake packet to trigger registration.
        self.send("HANDSHAKE")

        # Get the client ID from the server
        self.id = int(self.block_read())
        print(f"Connected to server with id {self.id}")
        self.connect_callback(self.id)

    def send(self, data: object):
        serialized = pickle.dumps(data)
        self.server_socket.sendall(serialized)

    def read(self) -> Any:
        # Use select to check if data is available (non-blocking check)
        ready_to_read, _, _ = select.select([self.server_socket], [], [], 0)
        if not ready_to_read:
            return None  # No data available yet

        try:
            data, _ = self.server_socket.recvfrom(65535)
            return pickle.loads(data)
        except ConnectionResetError:
            # UDP ICMP Port Unreachable can sometimes trigger this on Windows
            return None

    def block_read(self) -> Any:
        # Blocking read
        data, _ = self.server_socket.recvfrom(65535)
        return pickle.loads(data)

    def close(self):
        self.send("close")
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