import unittest

import pyray as rl

from EasyCells3D.Components import Item
from EasyCells3D.Geometry import Vec2, Vec3
from EasyCells3D.PhysicsComponents import RectCollider, Rigidbody, SATPhysicsWorld


class DummyGame:
    def __init__(self, world=None):
        self.item_list = []
        self.to_init = []
        self.physics_world = world

    def CreateItem(self):
        return Item(self)

    def flush_init(self):
        for init in list(self.to_init):
            init()
        self.to_init.clear()
        for item in list(self.item_list):
            item.update()


class SATPhysicsWorldTests(unittest.TestCase):
    def test_dynamic_body_integrates_gravity_through_world_step(self):
        world = SATPhysicsWorld()
        game = DummyGame(world)
        body_item = _body(game, Vec2(0, 0), dynamic=True)
        rb = body_item.GetComponent(Rigidbody)
        game.flush_init()

        world.step(1.0 / 60.0)

        self.assertGreater(body_item.transform.y, 0)
        self.assertGreater(rb.velocity.y, 0)

    def test_dynamic_body_resolves_against_kinematic_ground(self):
        world = SATPhysicsWorld()
        game = DummyGame(world)
        body_item = _body(game, Vec2(0, 0), dynamic=True)
        _body(game, Vec2(0, 20), dynamic=False, size=Vec2(100, 10))
        game.flush_init()

        for _ in range(90):
            world.step(1.0 / 60.0)

        self.assertLessEqual(body_item.transform.y, 11.0)

    def test_queries_obey_masks(self):
        world = SATPhysicsWorld(gravity=Vec2.zero())
        game = DummyGame(world)
        _body(game, Vec2(0, 0), dynamic=False, mask=1)
        game.flush_init()

        ray_hit = world.ray_cast(Vec2(-20, 0), Vec2(1, 0), 50, mask=2)
        overlaps = world.overlap_sphere(Vec3(0, 0, 0), 8, mask=2)

        self.assertIsNone(ray_hit)
        self.assertEqual(overlaps, [])

    def test_destroy_unregisters_body_and_collider(self):
        world = SATPhysicsWorld()
        game = DummyGame(world)
        body_item = _body(game, Vec2(0, 0), dynamic=True)
        game.flush_init()

        body_item.Destroy()

        self.assertEqual(world.rigidbodies, [])
        self.assertEqual(world.colliders, [])

    def test_raycast_and_overlap_use_world_registries(self):
        world = SATPhysicsWorld()
        game = DummyGame(world)
        body_item = _body(game, Vec2(0, 0), dynamic=False, mask=4)
        rb = body_item.GetComponent(Rigidbody)
        collider = body_item.GetComponent(RectCollider)
        game.flush_init()

        ray_hit = world.ray_cast(Vec2(-20, 0), Vec2(1, 0), 50, mask=4)
        common_hit = world.raycast(Vec3(-20, 0, 0), Vec3(1, 0, 0), 50, mask=4)
        overlaps = world.overlap_sphere(Vec3(0, 0, 0), 8, mask=4)

        self.assertIsNotNone(ray_hit)
        self.assertIs(ray_hit[0], collider)
        self.assertIsNotNone(common_hit)
        self.assertIs(common_hit.body, rb)
        self.assertIn(rb, overlaps)

    def test_2d_rigidbody_requires_sat_world(self):
        game = DummyGame()
        item = game.CreateItem()
        item.AddComponent(Rigidbody())

        with self.assertRaisesRegex(RuntimeError, "SATPhysicsWorld"):
            game.flush_init()

    def test_import_smoke(self):
        from EasyCells3D.PhysicsComponents import SATPhysicsWorld as Exported

        self.assertIs(Exported, SATPhysicsWorld)


def _body(
        game: DummyGame,
        position: Vec2,
        dynamic: bool,
        size: Vec2 = Vec2(10, 10),
        mask: int = 1,
) -> Item:
    item = game.CreateItem()
    item.transform.positionVec2 = position
    item.AddComponent(RectCollider(rl.Rectangle(0, 0, size.x, size.y), mask=mask))
    item.AddComponent(Rigidbody(
        use_gravity=dynamic,
        is_kinematic=not dynamic,
        restitution=0,
        drag=0,
    ))
    return item


if __name__ == "__main__":
    unittest.main()
