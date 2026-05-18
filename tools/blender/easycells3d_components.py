bl_info = {
    "name": "EasyCells3D Components",
    "author": "EasyCells3D",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "category": "Object",
}

import json
from pathlib import Path

import bpy


CONFIG_PATH = Path(__file__).with_name("easycells3d_components.json")


def load_component_config():
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {
        "components": [
            {
                "name": "RotatingObj",
                "fields": {
                    "speed": {"type": "float", "default": 1.0}
                },
            }
        ]
    }


def object_components(obj):
    raw = obj.get("components", "[]")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return raw if isinstance(raw, list) else []


def save_object_components(obj, components):
    obj["components"] = json.dumps(components)


class EC3D_ComponentName(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()


class EC3D_OT_add_component(bpy.types.Operator):
    bl_idname = "ec3d.add_component"
    bl_label = "Add EasyCells3D Component"

    component_name: bpy.props.StringProperty()

    def execute(self, context):
        obj = context.object
        if not obj:
            return {"CANCELLED"}
        config = load_component_config()
        component_info = next((c for c in config["components"] if c["name"] == self.component_name), None)
        fields = component_info.get("fields", {}) if component_info else {}
        components = object_components(obj)
        components.append({
            "type": self.component_name,
            "config": {name: field.get("default") for name, field in fields.items()},
        })
        save_object_components(obj, components)
        return {"FINISHED"}


class EC3D_OT_remove_component(bpy.types.Operator):
    bl_idname = "ec3d.remove_component"
    bl_label = "Remove EasyCells3D Component"

    index: bpy.props.IntProperty()

    def execute(self, context):
        obj = context.object
        components = object_components(obj)
        if 0 <= self.index < len(components):
            components.pop(self.index)
            save_object_components(obj, components)
        return {"FINISHED"}


class EC3D_PT_components(bpy.types.Panel):
    bl_label = "EasyCells3D Components"
    bl_idname = "EC3D_PT_components"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if not obj:
            layout.label(text="No object selected")
            return

        components = object_components(obj)
        for index, component in enumerate(components):
            row = layout.row()
            row.label(text=component.get("type", "Unknown"))
            op = row.operator("ec3d.remove_component", text="", icon="X")
            op.index = index
            layout.label(text=json.dumps(component.get("config", {})))

        config = load_component_config()
        for component in config.get("components", []):
            op = layout.operator("ec3d.add_component", text=f"Add {component['name']}")
            op.component_name = component["name"]


classes = (
    EC3D_ComponentName,
    EC3D_OT_add_component,
    EC3D_OT_remove_component,
    EC3D_PT_components,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
