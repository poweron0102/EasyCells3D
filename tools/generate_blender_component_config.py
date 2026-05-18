import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

COMPONENTS = [
    "UserComponents.ratating_obj.RotatingObj",
]


def load_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    output = {
        "components": []
    }
    for class_path in COMPONENTS:
        component_cls = load_class(class_path)
        fields = getattr(component_cls, "blender_fields", None)
        if not fields:
            continue
        output["components"].append({
            "name": component_cls.__name__,
            "class": class_path,
            "fields": fields,
        })

    output_path = REPO_ROOT / "tools/blender/easycells3d_components.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
