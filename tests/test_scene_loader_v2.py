import json
import sys
import tempfile
import unittest
from pathlib import Path

from EasyCells3D.Assets import Asset, export
from EasyCells3D.ComponentDiscovery import discover_assets_ast, discover_assets_runtime
from EasyCells3D.ComponentRegistry import ComponentRegistry
from EasyCells3D.Components import Component, Item
from EasyCells3D.SceneLoader import SceneLoader


class DummyGame:
    def __init__(self):
        self.item_list = []
        self.to_init = []

    def CreateItem(self):
        return Item(self)


class TestRegistry(ComponentRegistry):
    _components = {}
    _assets = {}
    _discovered = True


class Recorder(Component):
    def __init__(self, asset=None, tile=None, inline=None):
        self.asset = asset
        self.tile = tile
        self.inline = inline
        self.target = None
        self.target_component = None


class TileMapAsset(Asset):
    def __init__(self, img_path: str, tile_size_w: int, tile_size_h: int):
        self.img_path = img_path
        self.tile_size = (tile_size_w, tile_size_h)

    @export
    def get_tile(self, x: int = 0, y: int = 0):
        return {
            "img_path": self.img_path,
            "tile_size": self.tile_size,
            "x": x,
            "y": y,
        }

    def hidden(self):
        return "nope"


TestRegistry.register("Recorder", Recorder)
TestRegistry.register_asset("TileMapAsset", TileMapAsset)


class SceneLoaderV2Tests(unittest.TestCase):
    def test_loads_assets_selectors_and_references(self):
        scene = {
            "format": "easycells3d.scene",
            "version": 2,
            "assets": {
                "terrain": {
                    "type": "TileMapAsset",
                    "args": {
                        "img_path": "Tilesets/terrain.png",
                        "tile_size_w": 16,
                        "tile_size_h": 16,
                    },
                }
            },
            "Item": [
                {
                    "id": "1",
                    "name": "Root",
                    "parent": None,
                    "components": [
                        {
                            "id": "2",
                            "type": "Recorder",
                            "args": {
                                "asset": {"$assetRef": "terrain"},
                                "tile": {
                                    "$assetRef": "terrain",
                                    "selector": {
                                        "method": "get_tile",
                                        "args": {"x": 3, "y": 4},
                                    },
                                },
                                "inline": {
                                    "type": "TileMapAsset",
                                    "args": {
                                        "img_path": "Tilesets/inline.png",
                                        "tile_size_w": 8,
                                        "tile_size_h": 8,
                                    },
                                    "selector": {"method": "get_tile"},
                                },
                            },
                            "fields": {
                                "target": {"$ref": "3"},
                                "target_component": {"$componentRef": "4"},
                            },
                        }
                    ],
                },
                {
                    "id": "3",
                    "name": "Child",
                    "parent": "1",
                    "components": [
                        {
                            "id": "4",
                            "type": "Recorder",
                            "args": {},
                            "fields": {},
                        }
                    ],
                },
            ],
        }

        roots = _load_scene(scene)

        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertEqual(root.easycells_id, "1")
        self.assertEqual(len(root.children), 1)
        recorder = root.components[Recorder]
        child = next(iter(root.children))
        child_recorder = child.components[Recorder]

        self.assertIsInstance(recorder.asset, TileMapAsset)
        self.assertEqual(recorder.tile["x"], 3)
        self.assertEqual(recorder.tile["y"], 4)
        self.assertEqual(recorder.inline["img_path"], "Tilesets/inline.png")
        self.assertIs(recorder.target, child)
        self.assertIs(recorder.target_component, child_recorder)

    def test_selector_requires_exported_method(self):
        scene = _minimal_scene({
            "asset": {
                "type": "TileMapAsset",
                "args": {
                    "img_path": "Tilesets/terrain.png",
                    "tile_size_w": 16,
                    "tile_size_h": 16,
                },
                "selector": {"method": "hidden"},
            }
        })

        with self.assertRaises(KeyError):
            _load_scene(scene)

    def test_rejects_duplicate_ids(self):
        scene = _minimal_scene({})
        scene["Item"][0]["components"][0]["id"] = "1"

        with self.assertRaises(ValueError):
            _load_scene(scene)

    def test_rejects_missing_asset_ref(self):
        scene = _minimal_scene({"asset": {"$assetRef": "missing"}})

        with self.assertRaises(KeyError):
            _load_scene(scene)

    def test_rejects_item_ref_by_name(self):
        scene = _scene_with_child_ref({"$ref": "Child"})

        with self.assertRaises(KeyError):
            _load_scene(scene)

    def test_rejects_old_id_ref_alias(self):
        scene = _scene_with_child_ref({"$id": "3"})

        with self.assertRaises(ValueError):
            _load_scene(scene)

    def test_discovers_asset_metadata_ast_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "UserAssets"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "test_asset.py").write_text(
                "\n".join([
                    "from EasyCells3D.Assets import Asset, export",
                    "",
                    "class DemoAsset(Asset):",
                    "    def __init__(self, path: str, scale: int = 1):",
                    "        self.path = path",
                    "        self.scale = scale",
                    "",
                    "    @export",
                    "    def pick(self, index: int = 0) -> str:",
                    "        return self.path",
                    "",
                ]),
                encoding="utf-8",
            )

            try:
                ast_assets = discover_assets_ast(root)
                runtime_assets = discover_assets_runtime(root)
            finally:
                for name in list(sys.modules):
                    if name == "UserAssets" or name.startswith("UserAssets."):
                        sys.modules.pop(name, None)

        ast_demo = _only_asset(ast_assets)
        runtime_demo = _only_asset(runtime_assets)
        self.assertEqual(ast_demo.name, "DemoAsset")
        self.assertEqual(ast_demo.required_args[0].name, "path")
        self.assertEqual(ast_demo.optional_args[0].name, "scale")
        self.assertIn("pick", ast_demo.methods)
        self.assertIn("pick", runtime_demo.methods)


def _minimal_scene(args):
    return {
        "format": "easycells3d.scene",
        "version": 2,
        "Item": [
            {
                "id": "1",
                "name": "Root",
                "parent": None,
                "components": [
                    {
                        "id": "2",
                        "type": "Recorder",
                        "args": args,
                        "fields": {},
                    }
                ],
            }
        ],
    }


def _scene_with_child_ref(ref):
    scene = _minimal_scene({})
    scene["Item"][0]["components"][0]["fields"] = {"target": ref}
    scene["Item"].append({
        "id": "3",
        "name": "Child",
        "parent": "1",
        "components": [],
    })
    return scene


def _load_scene(scene):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.ecscene"
        path.write_text(json.dumps(scene), encoding="utf-8")
        return SceneLoader(DummyGame(), TestRegistry).load(path)


def _only_asset(assets):
    demos = [asset for asset in assets if asset.name == "DemoAsset"]
    assert len(demos) == 1
    return demos[0]


if __name__ == "__main__":
    unittest.main()
