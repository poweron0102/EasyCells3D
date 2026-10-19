"""Deterministic protocol tests with three separate game/scheduler contexts."""
from collections import deque
import gc
import importlib
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from EasyCells3D.NetworkComponents import (
    NetworkComponent, NetworkManager, Protocol, Rpc, SendTo,
    RpcCall, RpcConnectionError, RpcGroupError, RpcRemoteError, RpcTimeoutError,
)
from EasyCells3D.scheduler import Scheduler, SchedulerTaskCancelled
from test_network_lifecycle import DummyGame

async_module = importlib.import_module('EasyCells3D.NetworkComponents.AsyncRpc')


class MemoryTransport:
    def __init__(self, peer, queue):
        self.peer = peer
        self.queue = queue
        self.clients = [None, True, True]
        self._impl = SimpleNamespace(connected=True)

    def send(self, packet, client_id=0):
        self.queue.append((client_id, self.peer, pickle.loads(pickle.dumps(packet))))

    def read(self, client_id=0):
        for entry in self.queue:
            if entry[:2] == (self.peer, client_id):
                self.queue.remove(entry)
                return entry[2]

    def close(self):
        self._impl.connected = False


class AsyncRpcTests(unittest.TestCase):
    def setUp(self):
        for p in (patch.object(NetworkManager, 'instance', None),
                  patch.object(Scheduler, 'instance', None),
                  patch.object(NetworkComponent, '_static_instance', None),
                  patch.dict(NetworkComponent._active_components, clear=True),
                  patch.dict(NetworkComponent._static_rpcs)):
            p.start()
            self.addCleanup(p.stop)
        self.now = 100.0
        clock = patch.object(async_module, 'monotonic', lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.queue = deque()
        self.games = {i: DummyGame() for i in range(3)}
        self.managers = {}
        self.objects = {i: {} for i in range(3)}
        self.calls = []
        for peer, game in self.games.items():
            manager = NetworkManager('127.0.0.1', 0, peer == 0, enable_udp=False, max_frame_ms=1000)
            game.CreateItem().AddComponent(manager)
            manager.transports = {Protocol.TCP: MemoryTransport(peer, self.queue)}
            manager.id = peer
            manager._tcp_connected = True
            manager._connected_peers = {1, 2} if peer == 0 else set()
            self.managers[peer] = manager
        self.addCleanup(self.close)

    def close(self):
        for call in self.calls:
            call._observed = True
        for peer, manager in self.managers.items():
            self.activate(peer)
            self.games[peer].scheduler.clear()
            manager.on_destroy()

    def activate(self, peer):
        NetworkManager.instance = self.managers[peer]
        Scheduler.instance = self.games[peer].scheduler
        NetworkComponent._active_components.clear()
        NetworkComponent._active_components.update(self.objects[peer])

    def install(self, cls, owner=1):
        for peer, game in self.games.items():
            player = cls(10, owner)
            game.CreateItem().AddComponent(player)
            self.objects[peer][10] = player

    def call(self, origin, method='event', *args, **kwargs):
        self.activate(origin)
        call = getattr(self.objects[origin][10], method)(*args, **kwargs)
        self.calls.append(call)
        return call

    def pump(self, frames=8, peers=(0, 1, 2)):
        for _ in range(frames):
            for peer in peers:
                self.activate(peer)
                self.managers[peer].loop()
                self.games[peer].run_time += .02
                self.games[peer].scheduler.update()

    def test_every_destination_origin_and_owner_returns_correct_values(self):
        for destination in SendTo:
            for origin in range(3):
                for owner in range(3):
                    with self.subTest(destination=destination, origin=origin, owner=owner):
                        class Player(NetworkComponent):
                            @Rpc(send_to=destination, require_owner=False)
                            async def event(self, value, *, extra):
                                await self.game.scheduler.next_frame()
                                return NetworkManager.instance.id, value + extra

                        self.install(Player, owner)
                        call = self.call(origin, 'event', 3, extra=4)
                        self.assertIsInstance(call, RpcCall)
                        self.assertFalse(call.done)
                        self.pump()
                        targets = {
                            SendTo.ALL: [0, 1, 2], SendTo.SERVER: [0],
                            SendTo.CLIENTS: [1, 2], SendTo.OWNER: [owner],
                            SendTo.NOT_ME: [i for i in range(3) if i != origin],
                        }[destination]
                        expected = {peer: (peer, 7) for peer in targets}
                        self.assertEqual(call.result(), expected if destination in
                                         (SendTo.ALL, SendTo.CLIENTS, SendTo.NOT_ME) else expected[targets[0]])
                        self.assertFalse(self.managers[0]._async_rpc.routes)
                        self.assertTrue(all(not m._async_rpc.executions for m in self.managers.values()))

    def test_scheduler_await_nested_rpc_and_concurrent_calls(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER, require_owner=False)
            async def event(self, value):
                return 1 + await self.inner(value)

            @Rpc(send_to=SendTo.OWNER, require_owner=False)
            async def inner(self, value):
                await self.game.scheduler.sleep(.04)
                return value * 2

        self.install(Player, owner=2)
        calls = [self.call(1, 'event', value) for value in (3, 10)]
        received = []

        async def consumer():
            for call in calls:
                received.append(await call)

        self.games[1].scheduler.create_task(consumer())
        self.pump(20)
        self.assertEqual(received, [7, 21])

    def test_collective_failure_waits_for_other_results(self):
        class Player(NetworkComponent):
            @Rpc(require_owner=False)
            async def event(self):
                if NetworkManager.instance.id == 1:
                    raise ValueError('denied')
                await self.game.scheduler.sleep(.06)
                return NetworkManager.instance.id

        self.install(Player)
        call = self.call(2)
        self.pump(2)
        self.assertFalse(call.done)
        self.pump()
        with self.assertRaises(RpcGroupError) as caught:
            call.result()
        self.assertEqual(caught.exception.results, {0: 0, 2: 2})
        self.assertEqual(caught.exception.errors[1].remote_type, 'ValueError')
        self.assertEqual(caught.exception.errors[1].message, 'denied')

    def test_timeout_uses_real_clock_preserves_partial_results_and_ignores_late_reply(self):
        class Player(NetworkComponent):
            @Rpc(require_owner=False, timeout=2)
            async def event(self):
                if NetworkManager.instance.id == 2:
                    await self.game.scheduler.sleep(100)
                return NetworkManager.instance.id

        self.install(Player)
        call = self.call(1)
        self.pump()
        self.assertFalse(call.done)
        self.now += 3
        self.pump(1)
        with self.assertRaises(RpcGroupError) as caught:
            call.result()
        self.assertEqual(caught.exception.results, {0: 0, 1: 1})
        self.assertIsInstance(caught.exception.errors[2], RpcTimeoutError)
        self.games[2].run_time += 101
        self.pump()
        self.assertIs(call.exception, caught.exception)
        self.assertFalse(self.managers[0]._async_rpc.routes)
        self.assertFalse(self.managers[1]._async_rpc.calls)

    def test_manager_default_timeout_starts_before_server_receives_request(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                return 4

        self.install(Player)
        self.managers[1].rpc_timeout = 3
        call = self.call(1)
        self.now += 4
        self.pump(1, peers=(1,))
        with self.assertRaises(RpcTimeoutError):
            call.result()

    def test_snapshot_empty_group_missing_owner_and_disconnect(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.CLIENTS, require_owner=False)
            async def event(self):
                await self.game.scheduler.sleep(1)
                return 7

            @Rpc(send_to=SendTo.OWNER, require_owner=False)
            async def owner_result(self):
                return 9

        self.install(Player, owner=2)
        tcp = self.managers[0].transports[Protocol.TCP]
        tcp.clients[2] = None
        missing = self.call(0, 'owner_result')
        with self.assertRaises(RpcConnectionError):
            missing.result()
        call = self.call(0)
        tcp.clients[2] = True  # A late connection must not join the snapshot.
        self.pump(2)
        tcp.clients[1] = None
        self.pump(1)
        with self.assertRaises(RpcGroupError) as caught:
            call.result()
        self.assertEqual(set(caught.exception.errors), {1})
        self.assertIsInstance(caught.exception.errors[1], RpcConnectionError)
        tcp.clients[2] = None
        self.assertEqual(self.call(0).result(), {})

    def test_owner_permissions_and_server_authority(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.OWNER)
            async def event(self):
                return 7

        self.install(Player, owner=2)
        denied = self.call(1)
        allowed = self.call(0)
        self.pump()
        with self.assertRaises(RpcRemoteError) as caught:
            denied.result()
        self.assertEqual(caught.exception.remote_type, 'PermissionError')
        self.assertEqual(allowed.result(), 7)

    def test_targeted_global_and_both_static_decorator_orders(self):
        @Rpc(send_to=SendTo.SERVER, require_owner=False)
        async def global_rpc(*, value):
            await NetworkManager.instance.game.scheduler.next_frame()
            return NetworkManager.instance.id, value

        class Player(NetworkComponent):
            @staticmethod
            @Rpc(send_to=SendTo.SERVER, require_owner=False)
            async def first():
                return 'first'

            @Rpc(send_to=SendTo.SERVER, require_owner=False)
            @staticmethod
            async def second():
                return 'second'

            @Rpc(send_to=SendTo.SERVER)
            async def event(self, *, value):
                return NetworkManager.instance.id, value

        self.install(Player)
        self.activate(0)
        targeted = self.managers[0].call_rpc_on_client(2, self.objects[0][10].event, value=12)
        targeted_global = self.managers[0].call_rpc_on_client(2, global_rpc, value=13)
        self.activate(1)
        calls = [global_rpc(value=11), Player.first(), Player.second(), targeted, targeted_global]
        self.calls.extend(calls)
        self.pump()
        self.assertEqual([call.result() for call in calls], [(0, 11), 'first', 'second', (2, 12), (2, 13)])

    def test_invalid_return_is_an_error_without_losing_connection(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self, large=False):
                return b'x' * (1024 * 1024) if large else self

        self.install(Player)
        calls = [self.call(1), self.call(1, 'event', True)]
        self.pump()
        for call in calls:
            with self.assertRaises(RpcRemoteError):
                call.result()
        self.assertTrue(self.managers[1].transports[Protocol.TCP]._impl.connected)

    def test_errors_are_thrown_into_awaiting_coroutine(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                raise ValueError('failed')

        self.install(Player)
        self.activate(1)
        received = []

        async def consumer():
            try:
                await self.objects[1][10].event()
            except RpcRemoteError as error:
                received.append(error.remote_type)

        task = self.games[1].scheduler.create_task(consumer())
        self.pump()
        self.assertEqual(received, ['ValueError'])
        self.assertIsNone(task.exception)

    def test_cancelling_waiter_or_handle_does_not_cancel_remote_effects(self):
        effects = []

        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                await self.game.scheduler.sleep(.04)
                effects.append('executed')

        self.install(Player)
        call = self.call(1)

        async def consumer():
            await call

        task = self.games[1].scheduler.create_task(consumer())
        self.pump(1)
        task.cancel()
        self.assertEqual(call._waiters, [])
        call.cancel()
        self.pump()
        self.assertEqual(effects, ['executed'])
        with self.assertRaises(SchedulerTaskCancelled):
            call.result()

    def test_scene_clear_cancels_even_unstarted_global_execution(self):
        effects = []

        @Rpc(send_to=SendTo.SERVER, require_owner=False)
        async def global_rpc():
            effects.append('executed')

        self.activate(1)
        call = global_rpc()
        self.calls.append(call)
        self.activate(0)
        self.managers[0].loop()  # Receive without starting its coroutine.
        self.games[0].scheduler.clear()
        self.pump()
        self.assertEqual(effects, [])
        with self.assertRaises(RpcRemoteError) as caught:
            call.result()
        self.assertEqual(caught.exception.remote_type, 'SchedulerTaskCancelled')

    def test_destruction_cancels_running_handler_and_runs_finally(self):
        finalized = []

        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.OWNER)
            async def event(self):
                try:
                    await self.game.scheduler.sleep(100)
                finally:
                    finalized.append(True)

        self.install(Player)
        call = self.call(0)
        self.pump(2)
        self.activate(1)
        self.objects[1][10].on_destroy()
        self.pump()
        self.assertEqual(finalized, [True])
        with self.assertRaises(RpcRemoteError):
            call.result()

    def test_handler_can_destroy_its_own_component_without_losing_cancellation_reply(self):
        finalized = []

        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                try:
                    self.on_destroy()
                    await self.game.scheduler.next_frame()
                    finalized.append('resumed')
                finally:
                    finalized.append('closed')

        self.install(Player)
        call = self.call(1)
        self.pump()
        self.assertEqual(finalized, ['closed'])
        with self.assertRaises(RpcRemoteError) as caught:
            call.result()
        self.assertEqual(caught.exception.remote_type, 'SchedulerTaskCancelled')
        self.assertFalse(self.managers[0]._async_rpc.executions)

    def test_client_scene_clear_and_manager_close_settle_local_calls(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                await self.game.scheduler.sleep(100)

        self.install(Player)
        call = self.call(1)
        self.games[1].scheduler.clear()
        self.assertTrue(call.cancelled)
        call = self.call(1)
        self.activate(1)
        self.managers[1].on_destroy()
        with self.assertRaises(RpcConnectionError):
            call.result()

    def test_server_disconnection_settles_waiter(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER)
            async def event(self):
                await self.game.scheduler.sleep(100)

        self.install(Player)
        call = self.call(1)
        self.pump(2)
        self.managers[1].transports[Protocol.TCP]._impl.connected = False
        self.pump(1, peers=(1,))
        with self.assertRaises(RpcConnectionError):
            call.result()

    def test_missing_method_and_component_produce_explicit_errors(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.OWNER, require_owner=False)
            async def event(self):
                return 1

        self.install(Player, owner=2)
        self.objects[2].clear()
        call = self.call(1)
        self.pump()
        with self.assertRaises(RpcRemoteError) as caught:
            call.result()
        self.assertEqual(caught.exception.peer_id, 2)
        self.assertEqual(caught.exception.remote_type, 'LookupError')
        self.objects[0][10] = NetworkComponent(10, 2)
        call = self.call(1)
        self.pump()
        with self.assertRaises(RpcRemoteError) as caught:
            call.result()
        self.assertEqual(caught.exception.remote_type, 'LookupError')

    def test_forged_and_duplicate_replies_cannot_complete_other_peers(self):
        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.OWNER, require_owner=False)
            async def event(self):
                return 7

        self.install(Player, owner=2)
        call = self.call(1)
        self.activate(0)
        self.managers[0].loop()
        route_id = next(iter(self.managers[0]._async_rpc.routes))
        NetworkManager.process_packet((5, route_id, 'reply', (True, 999)), 1)
        self.pump()
        self.assertEqual(call.result(), 7)
        self.activate(0)
        NetworkManager.process_packet((5, route_id, 'reply', (True, 888)), 2)
        self.assertEqual(call.result(), 7)

    def test_udp_and_invalid_timeout_are_rejected(self):
        async def handler():
            pass

        with self.assertRaises(ValueError):
            Rpc(protocol=Protocol.UDP)(handler)
        for timeout in (0, -1, float('inf'), float('nan'), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                Rpc(timeout=timeout)(handler)

    def test_udp_request_and_client_selected_recipient_cannot_execute(self):
        effects = []

        class Player(NetworkComponent):
            @Rpc(send_to=SendTo.SERVER, require_owner=False)
            async def event(self):
                effects.append(True)

        self.install(Player)
        self.activate(0)
        udp_queue = deque([(0, 1, (5, 1, 'request', (10, 'event', (), {}, 10, None)))])
        udp = MemoryTransport(0, udp_queue)
        self.managers[0].transports[Protocol.UDP] = udp
        self.managers[0]._server_loop()
        self.assertFalse(self.managers[0]._async_rpc.executions)
        self.assertFalse(udp_queue)
        call = self.call(1)
        recipient, sender, packet = self.queue.pop()
        target, name, args, kwargs, duration, _ = packet[3]
        self.queue.append((recipient, sender, (5, packet[1], 'request',
                                               (target, name, args, kwargs, duration, 2))))
        self.pump()
        self.assertEqual(effects, [])
        with self.assertRaises(RpcRemoteError) as caught:
            call.result()
        self.assertEqual(caught.exception.remote_type, 'PermissionError')

    def test_unobserved_error_is_logged_when_handle_is_discarded(self):
        self.activate(1)
        call = RpcCall(self.games[1].scheduler, 'broken', 10)
        call.set_exception(RpcRemoteError(2, 'ValueError', 'broken'))
        with self.assertLogs(async_module.__name__, level='ERROR') as logs:
            del call
            gc.collect()
        self.assertIn('broken', logs.output[0])
        self.assertIn('Peer 2', logs.output[0])


if __name__ == '__main__':
    unittest.main()
