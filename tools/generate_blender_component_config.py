import json
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DISCOVERY_PATH = REPO_ROOT / "EasyCells3D" / "ComponentDiscovery.py"
spec = importlib.util.spec_from_file_location("easycells3d_component_discovery", DISCOVERY_PATH)
discovery = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = discovery
spec.loader.exec_module(discovery)


def main():
    components = discovery.discover_components(project_root=REPO_ROOT, mode="ast")
    output = {
        "components": [component.to_dict() for component in components]
    }

    output_path = REPO_ROOT / "tools/blender/easycells3d_components.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(f"{len(components)} components written to {output_path}")


if __name__ == "__main__":
    main()
