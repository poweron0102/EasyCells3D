import importlib.util
import json
import sys
from pathlib import Path


def load_discovery(project_root: Path):
    discovery_path = project_root / "EasyCells3D" / "ComponentDiscovery.py"
    spec = importlib.util.spec_from_file_location("easycells3d_component_discovery", discovery_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: discover_components.py <project_root>")

    project_root = Path(sys.argv[1]).resolve()
    discovery = load_discovery(project_root)
    components = discovery.discover_components(project_root=project_root, mode="ast")
    print(json.dumps({"components": [component.to_dict() for component in components]}))


if __name__ == "__main__":
    main()
