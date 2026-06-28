from abc import ABC, abstractmethod
from typing import Any, Callable

from EasyCells3D.NetworkTCP import NetworkServerTCP, NetworkClientTCP
from EasyCells3D.NetworkUDP import NetworkServerUDP, NetworkClientUDP


class Transport(ABC):
    """Uniform interface over a TCP/UDP backend, server or client role."""

    @abstractmethod
    def send(self, data: object, client_id: int = 0) -> None: ...

    @abstractmethod
    def read(self, client_id: int = 0) -> Any: ...

    @abstractmethod
    def broadcast(self, data: object) -> None: ...

    @property
    @abstractmethod
    def clients(self) -> list: ...

    @abstractmethod
    def close(self) -> None: ...


class _BaseTransport(Transport):
    """Shared role-normalization; subclasses set self._impl + self.is_server."""

    is_server: bool
    _impl: Any

    def send(self, data: object, client_id: int = 0) -> None:
        if self.is_server:
            self._impl.send(data, client_id)
        else:
            self._impl.send(data)

    def read(self, client_id: int = 0) -> Any:
        return self._impl.read(client_id) if self.is_server else self._impl.read()

    def broadcast(self, data: object) -> None:
        if self.is_server:
            self._impl.broadcast(data)

    @property
    def clients(self) -> list:
        # Only meaningful server-side; client backends have no peer list.
        return getattr(self._impl, "clients", [None])

    def close(self) -> None:
        self._impl.close()


class TcpTransport(_BaseTransport):
    def __init__(self, ip: str, port: int, ip_version: int, is_server: bool,
                 callback: Callable[[int], None]):
        self.is_server = is_server
        self._impl = (NetworkServerTCP if is_server else NetworkClientTCP)(
            ip, port, ip_version, callback)


class UdpTransport(_BaseTransport):
    def __init__(self, ip: str, port: int, ip_version: int, is_server: bool,
                 callback: Callable[[int], None]):
        self.is_server = is_server
        self._impl = (NetworkServerUDP if is_server else NetworkClientUDP)(
            ip, port, ip_version, callback)
