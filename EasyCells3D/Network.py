import socket
from typing import Callable, Any

import select
import pickle
import threading

from .scheduler import Scheduler

SIZE_SIZE = 4


class NetworkServer:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None):
        self.ip = ip
        self.port = port
        self.clients: list[socket.socket | None] = [None]
        if ip_version == 6:
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        elif ip_version == 4:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            raise ValueError("Invalid IP version")
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen()
        self.connect_callback = connect_callback
        print(f"Server running on {(self.ip, self.port)}")

        self.accept_thread = threading.Thread(target=self.accept_clients)
        # ends the thread when the main program ends
        self.accept_thread.daemon = True
        self.accept_thread.start()

    def accept_clients(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection established with {addr}, id: {len(self.clients)}")
            self.clients.append(client_socket)

            client_id = len(self.clients) - 1
            # Send the client their ID
            self.send(client_id, client_id)

            Scheduler.instance.add(0, lambda: self.connect_callback(client_id))

    def send(self, data: object, client_id: int):
        data = pickle.dumps(data)
        size = len(data).to_bytes(SIZE_SIZE, "big")
        # print(f"Data size: {len(data)}")  # Debug

        self.clients[client_id].sendall(size)
        self.clients[client_id].sendall(data)

    def read(self, client_id: int) -> Any:
        client_socket: socket.socket = self.clients[client_id]

        # Use select to check if data is available
        ready_to_read, _, _ = select.select([client_socket], [], [], 0)
        if not ready_to_read:
            return None  # No data available yet

        # Peek into the buffer to check the available data
        try:
            available_data = client_socket.recv(SIZE_SIZE, socket.MSG_PEEK)
        except ConnectionResetError:
            exit(1)
        if len(available_data) < SIZE_SIZE:
            return None  # Header size not fully available

        # Read the size from the available data
        size = int.from_bytes(available_data[:SIZE_SIZE], "big")

        # Check if the full data is available
        if len(client_socket.recv(size + SIZE_SIZE, socket.MSG_PEEK)) < size + SIZE_SIZE:
            return None  # Data not fully available yet

        # Read the full message from the buffer
        client_socket.recv(SIZE_SIZE)  # Consume the size header
        data = client_socket.recv(size)
        return pickle.loads(data)

    def block_read(self, client_id: int) -> Any:
        client_socket: socket.socket = self.clients[client_id]

        # Read the size from the available data
        size = int.from_bytes(client_socket.recv(SIZE_SIZE), "big")

        # Read the full message from the buffer
        data = client_socket.recv(size)
        return pickle.loads(data)

    def broadcast(self, data: object):
        for i in range(1, len(self.clients)):
            self.send(data, i)

    def close(self):
        for i in range(1, len(self.clients)):
            self.send("close", i)
            self.clients[i].close()
        self.clients = [None]
        self.server_socket.close()
        print("Server closed")

    def close_client(self, client_id: int):
        if self.clients[client_id]:
            self.send("close", client_id)
            self.clients[client_id].close()
            self.clients.pop(client_id)


class NetworkClient:
    def __init__(self, ip: str, port: int, ip_version: int = 4,
                 connect_callback: Callable[[int], None] = lambda x: None):
        self.ip = ip
        self.port = port
        self.connect_callback = connect_callback

        if ip_version == 6:
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        elif ip_version == 4:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            raise ValueError("Invalid IP version")

        self.id: int | None = None

        self.connect_thread = threading.Thread(target=self.connect)
        self.connect_thread.daemon = True
        self.connect_thread.start()

    def connect(self):
        self.server_socket.connect((self.ip, self.port))

        # Get the client ID from the server
        self.id = int(self.block_read())
        print(f"Connected to server with id {self.id}")
        self.connect_callback(self.id)

    def send(self, data: object):
        data = pickle.dumps(data)
        size = len(data).to_bytes(SIZE_SIZE, "big")

        self.server_socket.sendall(size)
        self.server_socket.sendall(data)

    def read(self) -> Any:
        # Peek into the buffer to check the available data

        # Use select to check if data is available
        ready_to_read, _, _ = select.select([self.server_socket], [], [], 0)
        if not ready_to_read:
            return None  # No data available yet

        available_data = self.server_socket.recv(SIZE_SIZE, socket.MSG_PEEK)
        if len(available_data) < SIZE_SIZE:
            return None  # Header size not fully available

        # Read the size from the available data
        size = int.from_bytes(available_data[:SIZE_SIZE], "big")

        # Check if the full data is available
        if len(self.server_socket.recv(size + SIZE_SIZE, socket.MSG_PEEK)) < size + SIZE_SIZE:
            return None  # Data not fully available yet

        # Read the full message from the buffer
        self.server_socket.recv(SIZE_SIZE)  # Consume the size header
        data = self.server_socket.recv(size)
        return pickle.loads(data)

    def block_read(self) -> Any:
        # Read the size from the available data
        size = int.from_bytes(self.server_socket.recv(SIZE_SIZE), "big")

        # Read the full message from the buffer
        data = self.server_socket.recv(size)
        return pickle.loads(data)

    def close(self):
        self.send("close")
        self.server_socket.close()

# Test
# IP = "localhost"
# PORT = 25765
#
# is_server = bool(int(input("Server(1) or Client(0): ")))
#
# if is_server:
#     server = NetworkServer(IP, PORT)
#
#     while len(server.clients) == 1:
#         pass
#
#     while True:
#         data = server.read(1)
#         print(data)
#         response = input("Response: ")
#         server.send(response, 1)
#
# else:
#     client = NetworkClient(IP, PORT)
#
#     while True:
#         response = input("Data: ")
#         client.send(response)
#         data = client.read()
#         print(data)
