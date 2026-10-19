"""Request/reply RPCs. All state transitions run on the game's polling thread."""
from dataclasses import dataclass
import logging
import math
import pickle
from time import monotonic

from EasyCells3D.NetworkTCP import MAX_PACKET, _decode
from EasyCells3D.NewGame import NewGame
from EasyCells3D.scheduler import SchedulerFuture


OP_ASYNC_RPC = 5
_logger = logging.getLogger(__name__)


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("RPC timeout must be a positive finite number of seconds")
    return float(value)


def _packet_copy(packet):
    encoded = pickle.dumps(packet, protocol=4)
    if len(encoded) > MAX_PACKET:
        raise ValueError("Network RPC packet too large")
    return _decode(encoded)


def _error_data(error):
    # Exceptions themselves must never be deserialized on another peer.
    return type(error).__name__, str(error)[:4096]


class RpcRemoteError(Exception):
    def __init__(self, peer_id, remote_type, message):
        self.peer_id = peer_id
        self.remote_type = remote_type
        self.message = message
        super().__init__(f"Peer {peer_id}: {remote_type}: {message}")


class RpcTimeoutError(RpcRemoteError, TimeoutError):
    pass


class RpcConnectionError(RpcRemoteError, ConnectionError):
    pass


class RpcGroupError(Exception):
    """A collective call failed; successful results remain available."""

    def __init__(self, results, errors):
        self.results = dict(results)
        self.errors = dict(errors)
        super().__init__("RPC failed: " + "; ".join(str(error) for error in errors.values()))


def _remote_error(peer, data):
    kind, message = data
    cls = {"TimeoutError": RpcTimeoutError, "ConnectionError": RpcConnectionError}.get(kind, RpcRemoteError)
    return cls(peer, kind, message)


class RpcCall(SchedulerFuture):
    """An already-sent RPC. Await it in this game's Scheduler, or call result()."""

    def __init__(self, scheduler, name, timeout):
        super().__init__(scheduler)
        self.name = name
        self.deadline = monotonic() + timeout
        self.results = {}
        self.errors = {}
        self._targets = None
        self._collective = False
        self._observed = False

    def __await__(self):
        self._observed = True
        return (yield from super().__await__())

    def result(self):
        if self.done:
            self._observed = True
        return super().result()

    def __del__(self):
        if self.exception is not None and not self._observed and not self.cancelled:
            _logger.error("Unobserved RPC failure in %s: %s", self.name, self.exception)

    def _accept(self, targets, collective):
        if self.done or self._targets is not None:
            return
        self._targets = set(targets)
        self._collective = collective
        self._complete_if_ready()

    def _receive(self, peer, success, value):
        if self.done or self._targets is None or peer not in self._targets:
            return
        if peer in self.results or peer in self.errors:
            return
        if success:
            self.results[peer] = value
        else:
            self.errors[peer] = _remote_error(peer, value)
        self._complete_if_ready()

    def _complete_if_ready(self):
        if self._targets is None or self._targets - self.results.keys() - self.errors.keys():
            return
        if self.errors:
            error = RpcGroupError(self.results, self.errors) if self._collective else next(iter(self.errors.values()))
            self.set_exception(error)
        else:
            self.set_result(dict(self.results) if self._collective else next(iter(self.results.values())))

    def _fail_pending(self, kind, message):
        if self.done:
            return
        if self._targets is None:
            self.set_exception(_remote_error(0, (kind, message)))
            return
        for peer in self._targets - self.results.keys() - self.errors.keys():
            self.errors[peer] = _remote_error(peer, (kind, message))
        self._complete_if_ready()


@dataclass
class _Route:
    origin: int
    call_id: int
    remaining: set
    deadline: float


class _AsyncRpc:
    """Owns local calls, server routes and local executions for one manager."""

    def __init__(self, manager):
        self.manager = manager
        self.calls = {}
        self.routes = {}
        self.executions = {}
        self._next_call = 0
        self._next_route = 0

    def _connected(self, peer):
        from .NetworkComponent import Protocol
        tcp = self.manager.transports.get(Protocol.TCP)
        if tcp is None or self.manager._closing:
            return False
        if self.manager.is_server:
            return peer == 0 or 0 < peer < len(tcp.clients) and tcp.clients[peer] is not None
        return peer == 0 and self.manager._tcp_connected and tcp._impl.connected

    def _send(self, peer, call_id, kind, payload):
        from .NetworkComponent import Protocol
        if not self._connected(peer):
            return
        packet = (OP_ASYNC_RPC, call_id, kind, payload)
        if self.manager.is_server:
            if peer == 0:
                self._deliver(call_id, kind, payload)
            else:
                self.manager.send_to_client(packet, peer, Protocol.TCP)
        else:
            self.manager.send_to_server(packet, Protocol.TCP)

    def start(self, target, name, args, kwargs, config, recipient=None):
        configured = config.get("timeout")
        duration = _timeout(self.manager.rpc_timeout if configured is None else configured)
        call = RpcCall(self.manager.game.scheduler, name, duration)
        self._next_call += 1
        call_id = self._next_call
        self.calls[call_id] = call
        call.add_done_callback(lambda _: self.calls.pop(call_id, None))
        try:
            if not self._connected(0):
                raise ConnectionError("NetworkManager is not connected")
            payload = (target, name, args, kwargs, duration, recipient)
            packet = _packet_copy((OP_ASYNC_RPC, call_id, "request", payload))
            if self.manager.is_server:
                self._request(call_id, packet[3], 0)
            else:
                self._send(0, call_id, "request", packet[3])
        except Exception as error:
            call.set_exception(_remote_error(self.manager.id, _error_data(error)))
        return call

    @staticmethod
    def _method(target, name):
        from .NetworkComponent import NetworkComponent
        component = (NetworkComponent.get_static_instance() if target == 0
                     else NetworkComponent._active_components.get(target))
        if component is None:
            raise LookupError(f"Network component {target} does not exist")
        method = NetworkComponent._static_rpcs.get(name) if target == 0 else getattr(component, name, None)
        if not getattr(method, "_rpc_config", {}).get("async"):
            raise LookupError(f"Async RPC {name!r} is not registered on component {target}")
        return component, method

    def _request(self, call_id, payload, origin):
        from .NetworkComponent import Protocol, SendTo, _rpc_recipient
        try:
            target, name, args, kwargs, duration, recipient = payload
            duration = _timeout(duration)
            component, method = self._method(target, name)
            config = method._rpc_config
            if config["protocol"] != Protocol.TCP:
                raise ValueError("Async RPCs require TCP")
            if origin != 0 and config["require_owner"] and component.owner != origin:
                raise PermissionError("Only the owner or server can call this RPC")
            if recipient is not None and origin != 0:
                raise PermissionError("Only the server can select an explicit RPC recipient")
            clients = self.manager.transports[Protocol.TCP].clients
            peers = {0} | {i for i in range(1, len(clients)) if clients[i] is not None}
            destination = config["send_to"]
            collective = recipient is None and destination in (SendTo.ALL, SendTo.CLIENTS, SendTo.NOT_ME)
            if recipient is not None:
                if type(recipient) is not int or recipient <= 0:
                    raise ValueError("A targeted RPC requires a client ID")
                targets = {recipient} & peers
            else:
                targets = {peer for peer in peers if _rpc_recipient(destination, peer, origin, component.owner)}
            if not collective and not targets:
                raise ConnectionError("RPC recipient is not connected")
            self._next_route += 1
            route_id = self._next_route
            self.routes[route_id] = _Route(origin, call_id, set(targets), monotonic() + duration)
            self._send(origin, call_id, "accept", (sorted(targets), collective))
            if not targets:
                self.routes.pop(route_id, None)
            # Relay before scheduling the server's own handler.
            invocation = (target, name, args, kwargs)
            for peer in sorted(targets - {0}):
                self._send(peer, route_id, "invoke", invocation)
            if 0 in targets:
                self._invoke(route_id, invocation)
        except Exception as error:
            self._send(origin, call_id, "reject", _error_data(error))

    def _invoke(self, route_id, payload):
        if route_id in self.executions:
            return
        try:
            target, name, args, kwargs = payload
            component, method = self._method(target, name)

            async def execute():
                try:
                    original = method.__wrapped__
                    value = await (original(*args, **kwargs) if target == 0 else original(component, *args, **kwargs))
                    # Validate the actual reply envelope, including its size, even for local execution.
                    return True, _packet_copy((OP_ASYNC_RPC, route_id, "reply", (True, value)))[3][1]
                except NewGame:
                    raise
                except Exception as error:
                    return False, _error_data(error)

            task = self.manager.game.scheduler.create_task(execute())
            self.executions[route_id] = (component, task)

            def finished(task):
                self.executions.pop(route_id, None)
                result = (False, _error_data(task.exception)) if task.exception is not None else task.result_value
                self._reply(route_id, *result)

            task.add_done_callback(finished)
        except Exception as error:
            self._reply(route_id, False, _error_data(error))

    def _reply(self, route_id, success, value):
        if self.manager.is_server:
            self._receive_reply(route_id, 0, success, value)
        else:
            self._send(0, route_id, "reply", (success, value))

    def _receive_reply(self, route_id, peer, success, value):
        route = self.routes.get(route_id)
        if route is None or peer not in route.remaining:
            return  # Unsolicited, duplicate and late replies cannot settle a call.
        if type(success) is not bool:
            return
        if not success and (not isinstance(value, (tuple, list)) or len(value) != 2 or
                            not all(isinstance(part, str) for part in value)):
            return
        try:
            _packet_copy((OP_ASYNC_RPC, route.call_id, "result", (peer, success, value)))
        except Exception as error:
            success, value = False, _error_data(error)
        route.remaining.remove(peer)
        self._send(route.origin, route.call_id, "result", (peer, success, value))
        if not route.remaining:
            self.routes.pop(route_id, None)

    def _deliver(self, call_id, kind, payload):
        call = self.calls.get(call_id)
        if call is None:
            return
        if monotonic() >= call.deadline:
            call._fail_pending("TimeoutError", "RPC deadline exceeded")
            return
        if kind == "accept":
            call._accept(*payload)
        elif kind == "result":
            call._receive(*payload)
        elif kind == "reject":
            call.set_exception(_remote_error(0, payload))

    def process(self, packet, sender):
        _, call_id, kind, payload = packet
        if type(call_id) is not int or call_id <= 0:
            raise ValueError("Invalid RPC call ID")
        if self.manager.is_server:
            if kind == "request":
                self._request(call_id, payload, sender)
            elif kind == "reply":
                self._receive_reply(call_id, sender, *payload)
            else:
                raise PermissionError("Clients may only send RPC requests and replies")
        elif sender == 0:
            if kind == "invoke":
                self._invoke(call_id, payload)
            elif kind in ("accept", "result", "reject"):
                self._deliver(call_id, kind, payload)
            else:
                raise PermissionError("Unexpected RPC packet from server")
        else:
            raise PermissionError("RPC traffic must pass through the server")

    def update(self):
        now = monotonic()
        for call in list(self.calls.values()):
            if not self._connected(0):
                call._fail_pending("ConnectionError", "Server disconnected")
            elif now >= call.deadline:
                call._fail_pending("TimeoutError", "RPC deadline exceeded")
        for route_id, route in list(self.routes.items()):
            if not self._connected(route.origin):
                self.routes.pop(route_id, None)
                continue
            for peer in list(route.remaining):
                if not self._connected(peer):
                    self._receive_reply(route_id, peer, False, ("ConnectionError", "RPC recipient disconnected"))
                elif now >= route.deadline:
                    self._receive_reply(route_id, peer, False, ("TimeoutError", "RPC deadline exceeded"))

    def cancel_component(self, component):
        for owner, task in list(self.executions.values()):
            if owner is component:
                task.cancel()

    def close(self):
        for _, task in list(self.executions.values()):
            task.cancel()
        for call in list(self.calls.values()):
            call._fail_pending("ConnectionError", "NetworkManager closed")
        self.routes.clear()
