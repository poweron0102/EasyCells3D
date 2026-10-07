"""End-to-end networking with independent runtimes and real loopback sockets."""
import multiprocessing
import threading
import time
import traceback
import unittest

from EasyCells3D.NetworkComponents import (
    NetworkComponent, NetworkManager, NetworkObject, NetworkTransform, NetworkVariable, Rpc, SendTo,
)
from test_network_lifecycle import DummyGame


class Avatar(NetworkComponent):
    def __init__(self):
        super().__init__()
        self.health = NetworkVariable(100)
        self.message = ''
        self.initial_health = None

    def on_network_spawn(self):
        self.initial_health = self.health.value

    @Rpc(send_to=SendTo.ALL)
    def announce(self, *, message):
        self.message = message


def avatar_factory(game, *, x=0):
    item = game.CreateItem()
    item.transform.x = x
    item.AddComponent(Avatar())
    item.AddComponent(NetworkTransform())
    return item


def peer_worker(connection, is_server, port):
    game = DummyGame()
    callback_threads = []
    manager = NetworkManager('127.0.0.1', port, is_server,
                             connect_callback=lambda _: callback_threads.append(threading.get_ident()))
    manager.register_prefab('avatar', avatar_factory)
    game.CreateItem().AddComponent(manager)
    try:
        game.flush_init()
        connection.send({'port': manager.port})
        running = True
        while running:
            game.run_time += game.delta_time
            game.flush_init()
            for item in list(game.item_list):
                item.update()
            game.scheduler.update()
            if connection.poll():
                command, args = connection.recv()
                try:
                    if command == 'stop':
                        running = False
                        result = True
                    elif command == 'spawn':
                        item = manager.spawn('avatar', owner=args['owner'], x=args['x'])
                        result = item.GetComponent(NetworkObject).identifier
                    elif command == 'despawn':
                        manager.despawn(args['identifier'])
                        result = True
                    elif command == 'change':
                        item = next(iter(manager.spawned.values()))
                        avatar = item.GetComponent(Avatar)
                        if 'health' in args:
                            avatar.health.value = args['health']
                        if 'x' in args:
                            item.transform.x = args['x']
                        if 'message' in args:
                            avatar.announce(message=args['message'])
                        result = True
                    elif command == 'state':
                        objects = []
                        for identifier, item in manager.spawned.items():
                            avatar = item.GetComponent(Avatar)
                            objects.append(dict(identifier=identifier, component=avatar.identifier,
                                                transform=item.GetComponent(NetworkTransform).identifier,
                                                owner=avatar.owner, health=avatar.health.value,
                                                initial_health=avatar.initial_health,
                                                message=avatar.message, x=item.transform.x))
                        result = dict(id=manager.id, tcp=manager._tcp_connected, udp=manager._udp_connected,
                                      error=manager.error, objects=objects,
                                      callbacks_on_main=all(t == threading.get_ident() for t in callback_threads))
                    else:
                        raise ValueError(command)
                    connection.send({'result': result})
                except Exception as exc:
                    connection.send({'error': type(exc).__name__, 'detail': str(exc)})
            time.sleep(.002)
    except Exception:
        connection.send({'fatal': traceback.format_exc()})
    finally:
        for item in list(game.item_list):
            item.Destroy()
        game.scheduler.clear()
        connection.close()


class NetworkIntegrationTests(unittest.TestCase):
    def start_peer(self, is_server, port=0):
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        process = context.Process(target=peer_worker, args=(child, is_server, port))
        process.start()
        child.close()

        def stop():
            if process.is_alive():
                try:
                    parent.send(('stop', {}))
                except OSError:
                    pass
                process.join(3)
            if process.is_alive():
                process.terminate()
                process.join(3)
            parent.close()

        self.addCleanup(stop)
        self.assertTrue(parent.poll(10), 'peer startup timed out')
        ready = parent.recv()
        self.assertIn('port', ready, ready)
        return parent, ready['port']

    def call(self, peer, command, **args):
        peer.send((command, args))
        self.assertTrue(peer.poll(5), 'peer response timed out')
        result = peer.recv()
        self.assertNotIn('fatal', result, result)
        return result

    def state_when(self, peer, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = self.call(peer, 'state')
            self.assertIn('result', response, response)
            state = response['result']
            if predicate(state):
                return state
            time.sleep(.01)
        self.fail(f'Expected network state was not reached: {state}')

    def test_spawn_rpc_variable_udp_late_join_and_despawn(self):
        server, port = self.start_peer(True)
        owner, _ = self.start_peer(False, port)
        state = self.state_when(owner, lambda s: s['tcp'] and s['udp'])
        self.assertEqual(state['id'], 1)
        self.assertTrue(state['callbacks_on_main'])
        spawned = self.call(server, 'spawn', owner=1, x=3)
        self.assertIn('result', spawned, spawned)
        identifier = spawned['result']
        self.state_when(owner, lambda s: len(s['objects']) == 1 and s['objects'][0]['initial_health'] == 100)
        self.assertEqual(self.call(owner, 'change', health=75, x=12, message='hello'), {'result': True})
        current = self.state_when(server, lambda s: bool(s['objects']) and
                                  s['objects'][0]['health'] == 75 and s['objects'][0]['x'] == 12 and
                                  s['objects'][0]['message'] == 'hello')
        self.assertTrue(current['callbacks_on_main'])
        self.state_when(owner, lambda s: s['objects'][0]['message'] == 'hello')
        late, _ = self.start_peer(False, port)
        late_state = self.state_when(late, lambda s: s['udp'] and bool(s['objects']) and
                                     s['objects'][0]['initial_health'] == 75)
        replica = late_state['objects'][0]
        self.assertEqual(replica['identifier'], identifier)
        self.assertEqual(replica['component'], current['objects'][0]['component'])
        self.assertEqual(replica['transform'], current['objects'][0]['transform'])
        self.assertEqual(replica['x'], 12)
        self.assertEqual(self.call(late, 'change', health=999)['error'], 'PermissionError')
        self.assertEqual(self.call(late, 'state')['result']['objects'][0]['health'], 75)
        self.assertEqual(self.call(server, 'despawn', identifier=identifier), {'result': True})
        self.state_when(owner, lambda s: not s['objects'])
        self.state_when(late, lambda s: not s['objects'])


if __name__ == '__main__':
    unittest.main()
