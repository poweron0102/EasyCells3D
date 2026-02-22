import ipaddress
from enum import Enum, auto
from functools import wraps
from typing import Callable
from weakref import WeakValueDictionary

from EasyCells3D.Components.Component import Component
from EasyCells3D.NetworkTCP import NetworkServerTCP, NetworkClientTCP
from EasyCells3D.NetworkUDP import NetworkServerUDP, NetworkClientUDP

# --- Constantes de Protocolo ---
OP_RPC = 1
OP_VAR = 2

# Sub-operações para Variáveis
VAR_SET = 1
VAR_GET = 2

# ID reservado para RPCs estáticos/globais
STATIC_NET_ID = 0


class Protocol(Enum):
    TCP = auto()
    UDP = auto()


class SendTo(Enum):
    ALL = 0  # Envia para todos (incluindo eu, se for Server)
    SERVER = 1  # Envia para o Server
    CLIENTS = 2  # Server envia para todos os clientes
    OWNER = 3  # Envia apenas para o dono do objeto
    NOT_ME = 4  # Envia para todos, exceto quem enviou


def Rpc(send_to: SendTo = SendTo.ALL, require_owner: bool = True, protocol: Protocol = Protocol.TCP):
    """
    Decorador que suporta métodos de instância (NetworkComponent),
    métodos estáticos (@staticmethod) e funções livres.
    Agora suporta escolha de protocolo (TCP ou UDP).
    """

    def decorator(func: Callable):
        if func.__qualname__ != func.__name__:
            pass

        @wraps(func)
        def wrapper(*args, **kwargs):
            instance = None
            if args and isinstance(args[0], NetworkComponent):
                instance = args[0]

            # --- Caminho de Instância ---
            if instance:
                if not getattr(instance, "_is_executing_rpc", False):
                    # Passa o protocolo definido para o send_rpc
                    instance.send_rpc(func.__name__, args[1:], send_to, protocol)

                    if NetworkManager.instance.is_server and send_to == SendTo.ALL:
                        return func(instance, *args[1:], **kwargs)
                    return None

                return func(instance, *args[1:], **kwargs)

            # --- Caminho Estático / Função Livre ---
            else:
                static_comp = NetworkComponent.get_static_instance()

                if func.__name__ not in NetworkComponent._static_rpcs:
                    NetworkComponent._static_rpcs[func.__name__] = func

                if not getattr(static_comp, "_is_executing_rpc", False):
                    static_comp.send_rpc(func.__name__, args, send_to, protocol)

                    if NetworkManager.instance.is_server and send_to == SendTo.ALL:
                        return func(*args, **kwargs)
                    return None

                return func(*args, **kwargs)

        wrapper._rpc_config = {
            "send_to": send_to,
            "require_owner": require_owner,
            "protocol": protocol  # Salva a config do protocolo
        }

        NetworkComponent._static_rpcs[func.__name__] = wrapper

        return wrapper

    return decorator


class NetworkComponent(Component):
    _active_components: dict[int, "NetworkComponent"] = {}
    _static_instance: "NetworkComponent" = None
    _static_rpcs: dict[str, Callable] = {}

    def __init__(self, identifier: int, owner: int):
        self.identifier = identifier
        self.owner = owner
        self._is_executing_rpc = False

        if identifier == STATIC_NET_ID:
            NetworkComponent._static_instance = self

    def init(self):
        if self.identifier is not None:
            NetworkComponent._active_components[self.identifier] = self

    @classmethod
    def get_static_instance(cls):
        if cls._static_instance is None:
            cls._static_instance = NetworkComponent(STATIC_NET_ID, 0)
            cls._active_components[STATIC_NET_ID] = cls._static_instance
        return cls._static_instance

    def on_destroy(self):
        if self.identifier in NetworkComponent._active_components:
            del NetworkComponent._active_components[self.identifier]
        self.on_destroy = lambda: None

    def send_rpc(self, method_name: str, args: tuple, send_to: SendTo, protocol: Protocol):
        """Encapsula e envia o pacote usando o protocolo especificado."""
        packet = (OP_RPC, self.identifier, method_name, args)
        nm = NetworkManager.instance

        if nm.is_server:
            if send_to == SendTo.ALL or send_to == SendTo.CLIENTS:
                nm.broadcast(packet, protocol)
            elif send_to == SendTo.NOT_ME:
                nm.broadcast(packet, protocol)
            elif send_to == SendTo.OWNER and self.owner != 0:
                nm.send_to_client(packet, self.owner, protocol)
        else:
            nm.send_to_server(packet, protocol)

    def handle_incoming_rpc(self, method_name: str, args: tuple, sender_id: int):
        """Recebe o pacote da rede, verifica segurança e executa."""
        method = None
        config = None

        if self.identifier == STATIC_NET_ID:
            if method_name in NetworkComponent._static_rpcs:
                original_func = NetworkComponent._static_rpcs[method_name]
                method = original_func
                config = method._rpc_config
            else:
                print(f"Erro: RPC Estático '{method_name}' não registrado.")
                return
        else:
            if not hasattr(self, method_name):
                print(f"Erro: RPC '{method_name}' não encontrado no objeto {self.identifier}")
                return
            method = getattr(self, method_name)
            if hasattr(method, "_rpc_config"):
                config = method._rpc_config

        if method is None or config is None:
            return

        if NetworkManager.instance.is_server and config["require_owner"]:
            if sender_id != self.owner:
                print(f"Negado: RPC {method_name} no objeto {self.identifier} exige permissão de dono.")
                return

        try:
            self._is_executing_rpc = True
            if self.identifier == STATIC_NET_ID:
                method(*args)
            else:
                method(*args)
        finally:
            self._is_executing_rpc = False

        # Se for Server, retransmite se necessário, respeitando o protocolo original
        if NetworkManager.instance.is_server:
            # Usa o protocolo definido na config do RPC para retransmitir
            protocol = config.get("protocol", Protocol.TCP)
            self._server_relay_rpc(method_name, args, config["send_to"], sender_id, protocol)

    def _server_relay_rpc(self, method_name: str, args: tuple, send_to: SendTo, sender_id: int, protocol: Protocol):
        """Lógica de retransmissão do servidor (Relay)."""
        packet = (OP_RPC, self.identifier, method_name, args)
        nm = NetworkManager.instance

        if send_to == SendTo.ALL or send_to == SendTo.CLIENTS:
            nm.broadcast(packet, protocol)
        elif send_to == SendTo.NOT_ME:
            # Broadcast manual excluindo o sender
            clients = nm.tcp_server.clients if protocol == Protocol.TCP else nm.udp_server.clients
            for cid in range(1, len(clients)):
                if cid != sender_id:
                    nm.send_to_client(packet, cid, protocol)
        elif send_to == SendTo.OWNER and self.owner != sender_id:
            nm.send_to_client(packet, self.owner, protocol)


class NetworkVariable[T]:
    """
    Variável sincronizada automaticamente pela rede.
    Por padrão, variáveis sempre usam TCP para garantir integridade.
    """
    _active_variables: WeakValueDictionary[int, 'NetworkVariable'] = WeakValueDictionary()

    def __init__(self, value: T, identifier: int, owner: int, require_owner: bool = True):
        self.var_id = identifier
        self.owner = owner
        self.require_owner = require_owner
        self._value = value

        NetworkVariable._active_variables[identifier] = self

        if not NetworkManager.instance.is_server:
            packet = (OP_VAR, self.var_id, VAR_GET, ())
            # Força TCP para variáveis
            NetworkManager.instance.send_to_server(packet, Protocol.TCP)

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, new_value: T):
        self._value = new_value
        packet = (OP_VAR, self.var_id, VAR_SET, (new_value,))
        nm = NetworkManager.instance

        if nm.is_server:
            nm.broadcast(packet, Protocol.TCP)
        else:
            nm.send_to_server(packet, Protocol.TCP)

    def handle_network_update(self, sub_op: int, args: tuple, sender_id: int):
        nm = NetworkManager.instance

        if sub_op == VAR_SET:
            new_val = args[0]
            if nm.is_server:
                if self.require_owner and sender_id != self.owner:
                    return

                self._value = new_val
                packet = (OP_VAR, self.var_id, VAR_SET, (new_val,))

                # Relay via TCP para garantir entrega
                clients = nm.tcp_server.clients
                for cid in range(1, len(clients)):
                    if cid != sender_id:
                        nm.send_to_client(packet, cid, Protocol.TCP)
            else:
                self._value = new_val

        elif sub_op == VAR_GET:
            if nm.is_server:
                packet = (OP_VAR, self.var_id, VAR_SET, (self._value,))
                nm.send_to_client(packet, sender_id, Protocol.TCP)


class NetworkManager(Component):
    instance: 'NetworkManager' = None

    def __init__(
            self,
            ip: str,
            port: int,
            is_server: bool,
            connect_callback: Callable[[int], None] = None
    ):
        NetworkManager.instance = self
        self.is_server = is_server
        self.ip = ip
        self.port = port
        self.id = 0 if is_server else -1

        self.connect_callbacks: list[Callable[[int], None]] = []
        if connect_callback is not None:
            self.connect_callbacks.append(connect_callback)

        if ip == "localhost":
            ip_version = 4
        else:
            try:
                ip_version = ipaddress.ip_address(ip).version
            except ValueError:
                ip_version = 4

        # Inicializa AMBOS os protocolos
        if self.is_server:
            self.tcp_server = NetworkServerTCP(ip, port, ip_version, self.server_callback_tcp)
            self.udp_server = NetworkServerUDP(ip, port, ip_version, self.server_callback_udp)
        else:
            self.tcp_client = NetworkClientTCP(ip, port, ip_version, self.client_callback_tcp)
            self.udp_client = NetworkClientUDP(ip, port, ip_version, self.client_callback_udp)

        self._tcp_connected = False
        self._udp_connected = False

    # --- Callbacks ---
    # Nota: Assumimos que o TCP é a conexão "Mestre" para definir o ID e disparar o callback do usuário

    def client_callback_tcp(self, client_id: int):
        self.id = client_id
        self._tcp_connected = True
        self._check_connection_complete(client_id)

    def client_callback_udp(self, client_id: int):
        self._udp_connected = True
        # Idealmente o ID do UDP deve bater com o do TCP.
        # Como as classes são separadas, esperamos que a ordem de conexão seja consistente.
        pass

    def _check_connection_complete(self, client_id):
        # Dispara o callback do usuário quando o TCP conecta (UDP pode vir depois ou falhar silenciosamente)
        if self._tcp_connected:
            for call in self.connect_callbacks:
                call(client_id)

    def server_callback_tcp(self, client_id: int):
        # Notifica nova conexão TCP
        for call in self.connect_callbacks:
            call(client_id)

    def server_callback_udp(self, client_id: int):
        pass

    # --- Métodos de Envio Abstraídos ---

    def send_to_server(self, packet: object, protocol: Protocol):
        """Envia pacote para o servidor usando o protocolo escolhido."""
        if self.is_server:
            return  # Servidor não envia para servidor

        if protocol == Protocol.TCP:
            self.tcp_client.send(packet)
        else:
            self.udp_client.send(packet)

    def send_to_client(self, packet: object, client_id: int, protocol: Protocol):
        """Envia pacote para um cliente específico."""
        if not self.is_server:
            return

        if protocol == Protocol.TCP:
            self.tcp_server.send(packet, client_id)
        else:
            self.udp_server.send(packet, client_id)

    def broadcast(self, packet: object, protocol: Protocol):
        """Envia para todos os clientes."""
        if not self.is_server:
            return

        if protocol == Protocol.TCP:
            self.tcp_server.broadcast(packet)
        else:
            self.udp_server.broadcast(packet)

    # --- Loop ---

    def init(self):
        if hasattr(self.item, "destroy_on_load"):
            self.item.destroy_on_load = False

    def loop(self):
        if self.is_server:
            self._server_loop()
        else:
            self._client_loop()

    def _server_loop(self):
        # Ler TCP
        clients_tcp = self.tcp_server.clients
        for client_id in range(1, len(clients_tcp)):
            while data := self.tcp_server.read(client_id):
                self.process_packet(data, client_id)

        # Ler UDP
        clients_udp = self.udp_server.clients
        for client_id in range(1, len(clients_udp)):
            while data := self.udp_server.read(client_id):
                self.process_packet(data, client_id)

    def _client_loop(self):
        # Ler TCP
        while data := self.tcp_client.read():
            self.process_packet(data, 0)

        # Ler UDP
        while data := self.udp_client.read():
            self.process_packet(data, 0)

    def call_rpc_on_client(self, client_id: int, rpc_method: Callable, *args):
        """Invoca um RPC num cliente específico. Tenta detectar protocolo do método."""
        if not self.is_server:
            return

        target_id = STATIC_NET_ID
        method_name = rpc_method.__name__
        protocol = Protocol.TCP  # Default seguro

        # Tenta pegar a config do wrapper
        if hasattr(rpc_method, "_rpc_config"):
            protocol = rpc_method._rpc_config.get("protocol", Protocol.TCP)

        if hasattr(rpc_method, "__self__") and isinstance(rpc_method.__self__, NetworkComponent):
            target_id = rpc_method.__self__.identifier

        packet = (OP_RPC, target_id, method_name, args)
        self.send_to_client(packet, client_id, protocol)

    @staticmethod
    def process_packet(data: tuple, sender_id: int):
        try:
            op_code, target_id, payload, args = data

            if op_code == OP_RPC:
                component = NetworkComponent._active_components.get(target_id)
                if component:
                    component.handle_incoming_rpc(payload, args, sender_id)

            elif op_code == OP_VAR:
                variable = NetworkVariable._active_variables.get(target_id)
                if variable:
                    variable.handle_network_update(payload, args, sender_id)

        except ValueError:
            print("Pacote malformado recebido.")
        except Exception as e:
            print(f"Erro processando pacote: {e}")

    def on_destroy(self):
        if self.is_server:
            self.tcp_server.close()
            self.udp_server.close()
        else:
            self.tcp_client.close()
            self.udp_client.close()