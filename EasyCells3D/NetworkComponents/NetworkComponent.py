import ipaddress
from enum import Enum
from functools import wraps
from typing import Callable, Any

from numpy.compat import unicode

from EasyCells3D.Components.Component import Component
from EasyCells3D.Network import NetworkServer, NetworkClient


class SendTo(Enum):
    ALL = 0
    SERVER = 1
    CLIENTS = 2
    OWNER = 3
    NOT_ME = 4


def Rpc(send_to: SendTo = SendTo.ALL, require_owner: bool = True):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata to the function
        wrapper._rpc_metadata = {
            "send_to": send_to,
            "require_owner": require_owner,
            "is_static": isinstance(func, staticmethod),
        }
        if isinstance(func, staticmethod):
            if func.__name__ in NetworkComponent.Rpcs:
                print(
                    f"RPC function '{func.__name__}' already registered.\n"
                    "Two RPC functions cannot have the same name.\n"
                    "Evem if they are in different classes."
                )
            NetworkComponent.Rpcs[func.__name__] = wrapper

            def new_func(*args, attr_name=func.__name__):
                return NetworkComponent.instance.invoke_rpc(attr_name, *args)

            return new_func

        return wrapper

    return decorator


class NetworkComponent(Component):
    instance: 'NetworkComponent'
    Rpcs: dict[str, Callable] = {}
    NetworkComponents: dict[int, "NetworkComponent"] = {}

    @staticmethod
    def GetFunction(name: str, identifier: int = None):
        if f"{name}:{identifier}" in NetworkComponent.Rpcs:
            return NetworkComponent.Rpcs[f"{name}:{identifier}"]
        elif name in NetworkComponent.Rpcs:
            return NetworkComponent.Rpcs[name]
        else:
            return None

    def __init__(self, identifier: int, owner: int):
        self.identifier = identifier
        self.owner = owner

        if identifier is None or owner is None:
            return

        NetworkComponent.NetworkComponents[identifier] = self

    def init(self):
        self.register_rpcs()

    def register_rpcs(self):
        """
        Automatically register methods decorated with @Rpc.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_rpc_metadata"):

                def new_func(*args, attr_name=attr_name):
                    return self.invoke_rpc(attr_name, *args)

                if attr._rpc_metadata["is_static"]:
                    continue
                else:
                    NetworkComponent.Rpcs[f"{attr_name}:{self.identifier}"] = attr
                    setattr(self, attr_name, new_func)

    def invoke_rpc(self, func_name: str, *args):
        """
        Call an RPC function locally or send the call to the network manager.
        """
        method = f"{func_name}:{self.identifier}" in NetworkComponent.Rpcs
        static_method = func_name in NetworkComponent.Rpcs
        if method or static_method:
            if method:
                func = NetworkComponent.Rpcs[f"{func_name}:{self.identifier}"]
            else:
                func = NetworkComponent.Rpcs[func_name]
            metadata = getattr(func, "_rpc_metadata", {})
            send_to = metadata.get("send_to", SendTo.ALL)
            require_owner = metadata.get("require_owner", True)

            if require_owner and not NetworkManager.instance.is_server and (
                    self.owner != NetworkManager.instance.id and self.owner is not None):
                raise PermissionError("This RPC requires ownership.")

            # Determine where to send the RPC
            if send_to == SendTo.ALL:
                self.send_rpc_to_all(func_name, *args)
            elif send_to == SendTo.SERVER:
                self.send_rpc_to_server(func_name, *args)
            elif send_to == SendTo.CLIENTS:
                self.send_rpc_to_clients(func_name, *args)
            elif send_to == SendTo.OWNER and self.owner:
                self.send_rpc_to_client(self.owner, func_name, *args)
            elif send_to == SendTo.NOT_ME:
                self.send_rpc_to_not_me(func_name, *args)
        else:
            raise ValueError(f"RPC function '{func_name}:{self.identifier}' or '{func_name}' not registered.")

    @staticmethod
    def CallRpc_on_client_id(func_name: str, identifier: int | None, client_id: int, *args):
        """
        Call an RPC function on a specific client.
        Only works on the server.
        if the identifier is None, this RPC needs to be a static method.
        """
        # if not NetworkManager.instance.is_server:
        #     return
        NetworkManager.instance.network_server.send(
            ("Rpc", (func_name, identifier, args)),
            client_id
        )

    def send_rpc_to_server(self, func_name: str, *args):
        if NetworkManager.instance.is_server:
            NetworkComponent.GetFunction(func_name, self.identifier)(*args)
        else:
            data = ("Rpc", (func_name, self.identifier, args))
            NetworkManager.instance.network_client.send(data)

    def send_rpc_to_clients(self, func_name: str, *args):
        if NetworkManager.instance.is_server:
            NetworkManager.instance.network_server.broadcast(("Rpc", (func_name, self.identifier, args)))
        else:
            data = ("RpcT", (func_name, self.identifier, args))
            NetworkManager.instance.network_client.send(data)

    def send_rpc_to_client(self, client_id: int, func_name: str, *args):
        if NetworkManager.instance.is_server:
            data = ("Rpc", (func_name, self.identifier, args))
            NetworkManager.instance.network_server.send(data, client_id)
        else:
            data = ("RpcT", (func_name, self.identifier, args))
            NetworkManager.instance.network_client.send(data)

    def send_rpc_to_all(self, func_name: str, *args):
        self.send_rpc_to_clients(func_name, *args)
        if NetworkManager.instance.is_server:
            NetworkComponent.GetFunction(func_name, self.identifier)(*args)

    def send_rpc_to_not_me(self, func_name: str, *args):
        if NetworkManager.instance.is_server:
            self.send_rpc_to_clients(func_name, *args)
        else:
            data = ("RpcT", (func_name, self.identifier, args))
            NetworkManager.instance.network_client.send(data)

    @staticmethod
    def rpc_handler_server(client_id: int, func_name: str, obj_identifier: int, args: tuple):
        func = NetworkComponent.GetFunction(func_name, obj_identifier)
        requer_owner = func._rpc_metadata.get("require_owner", True)

        if requer_owner and NetworkComponent.NetworkComponents[obj_identifier].owner != client_id:
            print("This RPC requires ownership.")
            return

        func(*args)

    @staticmethod
    def rpc_handler_client(func_name: str, obj_identifier: int, args: tuple):
        func = NetworkComponent.GetFunction(func_name, obj_identifier)
        if func is not None:
            func(*args)
        else:
            pass
            # print((func_name, obj_identifier, args))

    @staticmethod
    def rpct_handler_server(client_id: int, func_name: str, obj_identifier: int, args: tuple):
        func = NetworkComponent.GetFunction(func_name, obj_identifier)
        requer_owner = func._rpc_metadata.get("require_owner", True)
        is_static = func._rpc_metadata.get("is_static", False)

        if requer_owner and NetworkComponent.NetworkComponents[obj_identifier].owner != client_id:
            print("This RPC requires ownership.")
            return

        send_to = func._rpc_metadata.get("send_to", SendTo.ALL)

        if send_to == SendTo.ALL:
            NetworkManager.instance.network_server.broadcast(("Rpc", (func_name, obj_identifier, args)))
            func(*args)
        elif send_to == SendTo.CLIENTS:
            NetworkManager.instance.network_server.broadcast(("Rpc", (func_name, obj_identifier, args)))
        elif send_to == SendTo.OWNER:
            NetworkManager.instance.network_server.send(
                ("Rpc", (func_name, obj_identifier, args)),
                NetworkComponent.NetworkComponents[obj_identifier].owner
            )
        elif send_to == SendTo.NOT_ME:
            for i in range(1, len(NetworkManager.instance.network_server.clients)):
                if i != client_id:
                    NetworkManager.instance.network_server.send(("Rpc", (func_name, obj_identifier, args)), i)
            func(*args)

    @staticmethod
    def rpct_handler_client(func_name: str, obj_identifier: int, args: tuple):
        print("Received RPCT and I'm a client, this should not happen.")

    ping_time: float


class NetworkManager(Component):
    instance: 'NetworkManager' = None
    on_data_received: dict[str, Callable] = {}

    def __init__(
            self,
            ip: str,
            port: int,
            is_server: bool,
            ip_version: int = None,
            connect_callback: Callable[[int], None] = None,
    ):
        self.ip = ip
        self.port = port
        self.is_server = is_server
        NetworkManager.instance = self

        self.connect_callbacks: list[Callable[[int], None]] = []
        if connect_callback is not None:
            self.connect_callbacks.append(connect_callback)

        if ip_version is None:
            if ip == "localhost":
                ip_version = 4
            else:
                ip_version = ipaddress.ip_address(unicode(ip)).version

        if is_server:
            self.network_server = NetworkServer(ip, port, ip_version, self.server_callback)
            self.id = 0
            self.loop = self.server_loop
            NetworkManager.on_data_received["Rpc"] = NetworkComponent.rpc_handler_server
            NetworkManager.on_data_received["RpcT"] = NetworkComponent.rpct_handler_server
            NetworkManager.on_data_received["VarS"] = NetworkVariable.handle_variable_set_server
            NetworkManager.on_data_received["VarG"] = NetworkVariable.handle_variable_get_server
        else:
            self.network_client = NetworkClient(ip, port, ip_version, self.client_callback)
            self.loop = self.client_loop
            NetworkManager.on_data_received["Rpc"] = NetworkComponent.rpc_handler_client
            NetworkManager.on_data_received["RpcT"] = NetworkComponent.rpct_handler_client
            NetworkManager.on_data_received["VarS"] = NetworkVariable.handle_variable_set_client

    def client_callback(self, client_id: int):
        self.id = client_id
        for call in self.connect_callbacks:
            call(client_id)

    def server_callback(self, client_id: int):
        for call in self.connect_callbacks:
            call(client_id)

    def init(self):
        self.item.destroy_on_load = False

    def server_loop(self):
        for i in range(1, len(self.network_server.clients)):
            data = self.network_server.read(i)
            if data:
                self.handle_data(data, i)

    def client_loop(self):
        data = self.network_client.read()
        if data:
            self.handle_client(data)

    def handle_data(self, data: Any, client_id: int):
        operation, data = data

        if operation in self.on_data_received:
            self.on_data_received[operation](client_id, *data)

    def handle_client(self, data: object):
        operation, data = data

        if operation in NetworkManager.on_data_received:
            self.on_data_received[operation](*data)


class NetworkVariable[T]:
    variables: dict[int, 'NetworkVariable'] = {}

    _value: T

    def __init__(self, value: T, identifier: int, owner: int, require_owner: bool = True):
        """
        A variable that can be synchronized over the network.
        """

        self.identifier = identifier
        self.owner = owner

        self.require_owner = require_owner

        NetworkVariable.variables[identifier] = self

        self.value = value

        if not NetworkManager.instance.is_server:
            NetworkManager.instance.network_client.send(
                ("VarG", (identifier,))
            )

    @property
    def value(self) -> T:
        return self._value

    def set_Server(self, value: T):
        self._value = value
        for i in range(1, len(NetworkManager.instance.network_server.clients)):
            NetworkManager.instance.network_server.send(("VarS", (self.identifier, value)), i)

    def set_Client(self, value: T):
        self._value = value
        NetworkManager.instance.network_client.send(("VarS", (self.identifier, value)))

    @value.setter
    def value(self, value):
        if NetworkManager.instance.is_server:
            self.set_Server(value)
        else:
            self.set_Client(value)

    @staticmethod
    def handle_variable_set_server(client_id: int, identifier: int, value: T):
        if NetworkVariable.variables[identifier].require_owner:
            if NetworkVariable.variables[identifier].owner != client_id:
                print("This variable requires ownership.")
                return
        NetworkVariable.variables[identifier]._value = value

        for i in range(1, len(NetworkManager.instance.network_server.clients)):
            if i != client_id:
                NetworkManager.instance.network_server.send(("VarS", (identifier, value)), i)

    @staticmethod
    def handle_variable_set_client(identifier: int, value: T):
        NetworkVariable.variables[identifier]._value = value

    @staticmethod
    def handle_variable_get_server(client_id: int, identifier: int):
        NetworkManager.instance.network_server.send(
            ("VarS", (identifier, NetworkVariable.variables[identifier].value)
             ),
            client_id
        )


network_component_instance = NetworkComponent(None, None)
NetworkComponent.instance = network_component_instance
