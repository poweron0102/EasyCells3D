import unittest
from types import ModuleType
from unittest.mock import patch

from EasyCells3D.Game import Game
from EasyCells3D.NetworkComponents import NetworkComponent, NetworkManager, NetworkTransform
from test_network_lifecycle import DummyGame
from test_network_rpc import MemoryNetwork


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.game = DummyGame()
        self.game.physics_world = None
        self.game.update = lambda: None
        self.network = MemoryNetwork(0)
        for p in (patch.object(NetworkManager, 'instance', self.network),
                  patch.dict(NetworkComponent._active_components, clear=True)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.game.scheduler.clear)
        self.item = self.game.CreateItem()
        self.item.destroy_on_load = False
        self.transform = self.item.AddComponent(NetworkTransform(10, 0))
        self.game.flush_init()

    def tick(self):
        self.game.run_time += 0.02
        self.item.update()
        self.game.scheduler.update()

    def test_disabled_transform_stops_sending_and_resumes(self):
        self.tick()
        self.network.sent.clear()
        self.transform.enable = False
        self.item.transform.x = 5
        for _ in range(30):
            self.tick()
        self.assertEqual(self.network.sent, [])
        self.transform.enable = True
        self.tick()
        self.assertEqual(len(self.network.sent), 2)

    def test_persistent_transform_keeps_sending_after_scene_change(self):
        self.tick()
        self.network.sent.clear()
        level = ModuleType('next_level')
        level.init = lambda game: None
        Game.new_game(self.game, level, supress=True)
        self.item.transform.x = 5
        self.tick()
        self.assertEqual(len(self.network.sent), 2)


if __name__ == '__main__':
    unittest.main()
