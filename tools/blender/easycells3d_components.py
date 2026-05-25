bl_info = {
    "name": "EasyCells3D Components",
    "author": "PowerON0102",
    "version": (0, 2, 0),
    "blender": (3, 6, 0),
    "category": "Object",
}

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

import bpy


COMPONENTS_PROP = "components"
EASYCELLS_ID_PROP = "easycells_id"
ANIMATED_MODEL_PROP = "easycells_animated_model"


def addon_prefs(context=None):
    context = context or bpy.context
    return context.preferences.addons[__name__].preferences


def load_component_cache(context=None):
    raw = addon_prefs(context).component_cache
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return {"components": []}


def component_items(self, context):
    components = load_component_cache(context).get("components", [])
    if not components:
        return [("__none__", "No components found", "Use Refresh Components first")]
    return [
        (component["name"], component["name"], component.get("class_path", component["name"]))
        for component in components
    ]


def object_components(obj):
    raw = obj.get(COMPONENTS_PROP, "[]")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        data = raw

    if isinstance(data, dict):
        data = data.get("components", [])
    if not isinstance(data, list):
        return []
    return [_migrate_component(component) for component in data if isinstance(component, dict)]


def save_object_components(obj, components):
    obj[COMPONENTS_PROP] = json.dumps(components)


def component_metadata(name, context=None):
    for component in load_component_cache(context).get("components", []):
        if component.get("name") == name or component.get("class_path") == name:
            return component
    return None


def ensure_easycells_id(obj):
    if not obj.get(EASYCELLS_ID_PROP):
        obj[EASYCELLS_ID_PROP] = uuid.uuid4().hex
    return obj[EASYCELLS_ID_PROP]


def ensure_component_id_properties(obj, components):
    for index, component in enumerate(components):
        metadata = component_metadata(component.get("type"))
        args = component.setdefault("args", {})
        fields = component.setdefault("fields", {})

        for arg in (metadata or {}).get("required_args", []) + (metadata or {}).get("optional_args", []):
            key = _component_prop_key(index, "args", arg["name"])
            if key not in obj:
                obj[key] = args.get(arg["name"], _default_for(arg))

        for name, field in (metadata or {}).get("fields", {}).items():
            key = _component_prop_key(index, "fields", name)
            if key not in obj:
                obj[key] = _ui_value(fields.get(name, field.get("default")), field)
            if _field_ref(field) == "component":
                component_key = f"{key}_component"
                if component_key not in obj:
                    obj[component_key] = _component_ref_type(fields.get(name))


def sync_object_components(obj):
    components = object_components(obj)
    ensure_component_id_properties(obj, components)
    for index, component in enumerate(components):
        metadata = component_metadata(component.get("type"))
        component["args"] = {}
        component["fields"] = {}

        for arg in (metadata or {}).get("required_args", []) + (metadata or {}).get("optional_args", []):
            key = _component_prop_key(index, "args", arg["name"])
            component["args"][arg["name"]] = _typed_value(obj.get(key, _default_for(arg)), arg)

        for name, field in (metadata or {}).get("fields", {}).items():
            key = _component_prop_key(index, "fields", name)
            value = obj.get(key, _default_for(field))
            component_type = obj.get(f"{key}_component", "")
            component["fields"][name] = _export_field_value(value, field, component_type)

    save_object_components(obj, components)
    return components


def sync_scene_components(scene):
    for obj in scene.objects:
        ensure_easycells_id(obj)
        if COMPONENTS_PROP in obj:
            sync_object_components(obj)


def cleanup_editor_properties(scene):
    for obj in scene.objects:
        for key in list(obj.keys()):
            if str(key).startswith("ec3d_"):
                del obj[key]


class EC3D_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    project_root: bpy.props.StringProperty(
        name="Project Root",
        subtype="DIR_PATH",
        default="",
    )
    main_script: bpy.props.StringProperty(
        name="Main Script",
        default="main.py",
    )
    export_path: bpy.props.StringProperty(
        name="Default Export Path",
        subtype="FILE_PATH",
        default="Assets/Blender/scene.glb",
    )
    python_command: bpy.props.StringProperty(
        name="Python Command",
        default="python",
        description="Manual Python command/path. Leave as python to let the add-on use a project venv when found",
    )
    auto_detect_venv: bpy.props.BoolProperty(
        name="Auto-detect Project Venv",
        default=True,
        description="Use .venv, venv, or env from the project root when Python Command is not manually overridden",
    )
    run_args: bpy.props.StringProperty(
        name="Run Arguments",
        default="",
    )
    component_cache: bpy.props.StringProperty(default="")
    status_message: bpy.props.StringProperty(default="Not configured")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "project_root")
        layout.prop(self, "main_script")
        layout.prop(self, "export_path")
        layout.prop(self, "auto_detect_venv")
        layout.prop(self, "python_command")
        layout.prop(self, "run_args")
        layout.label(text=f"Runtime: {_runtime_python_label(self)}")
        layout.operator("ec3d.refresh_components", icon="FILE_REFRESH")
        layout.label(text=self.status_message)


class EC3D_OT_refresh_components(bpy.types.Operator):
    bl_idname = "ec3d.refresh_components"
    bl_label = "Refresh Components"

    def execute(self, context):
        prefs = addon_prefs(context)
        project_root = Path(bpy.path.abspath(prefs.project_root)).resolve() if prefs.project_root else Path.cwd()
        try:
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            discovery = _load_discovery_module(project_root)
            components = discovery.discover_components(project_root=project_root, mode="ast")
            payload = {"components": [component.to_dict() for component in components]}
            prefs.component_cache = json.dumps(payload)
            prefs.status_message = f"{len(components)} components found"
        except Exception as exc:
            prefs.status_message = f"Refresh failed: {exc}"
            self.report({"ERROR"}, prefs.status_message)
            return {"CANCELLED"}
        return {"FINISHED"}


class EC3D_OT_add_component(bpy.types.Operator):
    bl_idname = "ec3d.add_component"
    bl_label = "Add EasyCells3D Component"

    def execute(self, context):
        obj = context.object
        if not obj:
            return {"CANCELLED"}
        component_name = context.scene.ec3d_component_to_add
        if component_name == "__none__":
            return {"CANCELLED"}

        metadata = component_metadata(component_name, context)
        component = {
            "type": component_name,
            "args": {
                arg["name"]: _default_for(arg)
                for arg in (metadata or {}).get("required_args", []) + (metadata or {}).get("optional_args", [])
            },
            "fields": {
                name: field.get("default")
                for name, field in (metadata or {}).get("fields", {}).items()
            },
        }
        components = object_components(obj)
        components.append(component)
        save_object_components(obj, components)
        ensure_component_id_properties(obj, components)
        context.scene.ec3d_selected_component_index = len(components) - 1
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
            context.scene.ec3d_selected_component_index = min(self.index, max(0, len(components) - 1))
        return {"FINISHED"}


class EC3D_OT_select_component(bpy.types.Operator):
    bl_idname = "ec3d.select_component"
    bl_label = "Select Component"

    index: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.ec3d_selected_component_index = self.index
        return {"FINISHED"}


class EC3D_OT_sync_components(bpy.types.Operator):
    bl_idname = "ec3d.sync_components"
    bl_label = "Force Sync Components"
    bl_description = "Diagnostic action: export already syncs components automatically"

    def execute(self, context):
        sync_scene_components(context.scene)
        addon_prefs(context).status_message = "Components synced"
        return {"FINISHED"}


class EC3D_OT_export_scene(bpy.types.Operator):
    bl_idname = "ec3d.export_scene"
    bl_label = "Export"

    run_after_export: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        prefs = addon_prefs(context)
        project_root = Path(bpy.path.abspath(prefs.project_root)).resolve() if prefs.project_root else Path.cwd()
        export_path = Path(bpy.path.abspath(prefs.export_path))
        if not export_path.is_absolute():
            export_path = project_root / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        sync_scene_components(context.scene)
        animated_count = export_animated_models(context.scene, export_path, project_root)
        cleanup_editor_properties(context.scene)
        kwargs = _gltf_export_kwargs(export_path)
        bpy.ops.export_scene.gltf(**kwargs)
        if animated_count:
            prefs.status_message = f"Exported {export_path} and {animated_count} animated model(s)"
        else:
            prefs.status_message = f"Exported {export_path}"

        if self.run_after_export:
            try:
                command, runtime_source, main_script = _run_command(prefs, project_root)
                launch_command = _launch_command(command)
                subprocess.Popen(launch_command, cwd=str(project_root))
                prefs.status_message = f"Exported and launched {main_script.name} using {runtime_source}"
            except Exception as exc:
                prefs.status_message = f"Exported, but run failed: {exc}"
                self.report({"ERROR"}, prefs.status_message)
                return {"CANCELLED"}

        return {"FINISHED"}


class EC3D_OT_ensure_ids(bpy.types.Operator):
    bl_idname = "ec3d.ensure_ids"
    bl_label = "Ensure Stable IDs"

    def execute(self, context):
        for obj in context.scene.objects:
            ensure_easycells_id(obj)
        addon_prefs(context).status_message = "Stable IDs ready"
        return {"FINISHED"}


class EC3D_PT_viewport(bpy.types.Panel):
    bl_label = "EasyCells3D"
    bl_idname = "EC3D_PT_viewport"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "EasyCells3D"

    def draw(self, context):
        layout = self.layout
        prefs = addon_prefs(context)
        layout.operator("ec3d.refresh_components", icon="FILE_REFRESH")
        layout.operator("ec3d.ensure_ids", icon="KEYINGSET")
        layout.operator("ec3d.sync_components", text="Force Sync Components", icon="CHECKMARK")
        layout.operator("ec3d.export_scene", icon="EXPORT")
        op = layout.operator("ec3d.export_scene", text="Export & Run", icon="PLAY")
        op.run_after_export = True
        layout.separator()
        layout.label(text=f"Project: {prefs.project_root or '<not set>'}")
        layout.label(text=f"Export: {prefs.export_path or '<not set>'}")
        layout.label(text=f"Runtime: {_runtime_python_label(prefs)}")
        layout.label(text=prefs.status_message)


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

        ensure_easycells_id(obj)
        components = object_components(obj)
        ensure_component_id_properties(obj, components)

        row = layout.row(align=True)
        row.prop(context.scene, "ec3d_component_to_add", text="")
        row.operator("ec3d.add_component", text="Add", icon="ADD")

        layout.separator()
        if not components:
            layout.label(text="No components")
            return

        for index, component in enumerate(components):
            row = layout.row(align=True)
            icon = "DISCLOSURE_TRI_DOWN" if index == context.scene.ec3d_selected_component_index else "DISCLOSURE_TRI_RIGHT"
            op = row.operator("ec3d.select_component", text=component.get("type", "Unknown"), icon=icon)
            op.index = index
            remove = row.operator("ec3d.remove_component", text="", icon="X")
            remove.index = index

        selected = context.scene.ec3d_selected_component_index
        if selected < 0 or selected >= len(components):
            return

        component = components[selected]
        metadata = component_metadata(component.get("type"), context) or {}
        box = layout.box()
        box.label(text=component.get("type", "Unknown"))
        _draw_parameters(box, obj, selected, "args", metadata.get("required_args", []) + metadata.get("optional_args", []))
        _draw_fields(box, obj, selected, metadata.get("fields", {}), context)


def _draw_parameters(layout, obj, index, section, params):
    for param in params:
        key = _component_prop_key(index, section, param["name"])
        _ensure_prop_default(obj, key, _default_for(param))
        layout.prop(obj, f'["{key}"]', text=param["name"])


def _draw_fields(layout, obj, index, fields, context):
    for name, field in fields.items():
        key = _component_prop_key(index, "fields", name)
        _ensure_prop_default(obj, key, _default_for(field))
        ref = _field_ref(field)
        if ref == "item":
            layout.prop_search(obj, f'["{key}"]', context.scene, "objects", text=name)
        elif ref == "component":
            layout.prop_search(obj, f'["{key}"]', context.scene, "objects", text=f"{name} object")
            component_key = f"{key}_component"
            _ensure_prop_default(obj, component_key, "")
            layout.prop(obj, f'["{component_key}"]', text="component")
        else:
            layout.prop(obj, f'["{key}"]', text=name)


def _component_prop_key(index, section, name):
    return f"ec3d_{index}_{section}_{name}"


def _gltf_export_kwargs(export_path, selected=False):
    kwargs = {
        "filepath": str(export_path),
    }
    properties = _operator_property_ids(bpy.ops.export_scene.gltf)
    if export_path.suffix.lower() == ".glb" and "export_format" in properties:
        kwargs["export_format"] = "GLB"
    if selected:
        if "use_selection" in properties:
            kwargs["use_selection"] = True
        if "export_selected" in properties:
            kwargs["export_selected"] = True
    if "export_cameras" in properties:
        kwargs["export_cameras"] = True
    if "export_lights" in properties:
        kwargs["export_lights"] = True
    for animation_flag in ("export_animations", "export_animation"):
        if animation_flag in properties:
            kwargs[animation_flag] = True
    for animation_option in ("export_nla_strips", "export_force_sampling"):
        if animation_option in properties:
            kwargs[animation_option] = True
    if "export_extras" in properties:
        kwargs["export_extras"] = True
    elif "export_custom_properties" in properties:
        kwargs["export_custom_properties"] = True
    return kwargs


def _operator_property_ids(operator):
    try:
        return {prop.identifier for prop in operator.get_rna_type().properties}
    except Exception:
        return set()


def export_animated_models(scene, export_path, project_root):
    animated_roots = _animated_roots(scene)
    for root in animated_roots:
        asset_path = _animated_asset_path(root, export_path)
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        root[ANIMATED_MODEL_PROP] = json.dumps({
            "path": _path_for_metadata(asset_path, project_root),
            "clip_names": _animation_clip_names(root),
            "autoplay": True,
        })
        _export_selected_objects(_animated_export_objects(root), asset_path, root)
    return len(animated_roots)


def _animated_roots(scene):
    roots = []
    for obj in scene.objects:
        if not _is_animated_root(obj):
            continue
        if any(_is_descendant_of(obj, root) for root in roots):
            continue
        roots = [root for root in roots if not _is_descendant_of(root, obj)]
        roots.append(obj)
    return roots


def _is_animated_root(obj):
    if getattr(obj, "type", None) != "ARMATURE":
        return False
    return bool(_animation_clip_names(obj) or _has_animated_descendant(obj) or _has_mesh_descendant(obj))


def _has_animated_descendant(obj):
    return any(_has_animation_data(child) for child in _walk_children(obj))


def _has_mesh_descendant(obj):
    return any(getattr(child, "type", None) == "MESH" for child in _walk_children(obj))


def _has_animation_data(obj):
    animation_data = getattr(obj, "animation_data", None)
    if not animation_data:
        return False
    if getattr(animation_data, "action", None):
        return True
    for track in getattr(animation_data, "nla_tracks", []) or []:
        if getattr(track, "strips", None):
            return True
    return False


def _animation_clip_names(root):
    names = []

    def add_name(name):
        if name and name not in names:
            names.append(str(name))

    for obj in [root] + list(_walk_children(root)):
        animation_data = getattr(obj, "animation_data", None)
        if not animation_data:
            continue
        action = getattr(animation_data, "action", None)
        if action:
            add_name(getattr(action, "name", None))
        for track in getattr(animation_data, "nla_tracks", []) or []:
            for strip in getattr(track, "strips", []) or []:
                action = getattr(strip, "action", None)
                add_name(getattr(action, "name", None))

    for action in getattr(bpy.data, "actions", []) or []:
        add_name(getattr(action, "name", None))
    return names


def _animated_export_objects(root):
    return [root] + list(_walk_children(root))


def _walk_children(obj):
    for child in getattr(obj, "children", []) or []:
        yield child
        yield from _walk_children(child)


def _is_descendant_of(obj, ancestor):
    current = getattr(obj, "parent", None)
    while current:
        if current == ancestor:
            return True
        current = getattr(current, "parent", None)
    return False


def _animated_asset_path(root, export_path):
    safe_name = _safe_filename(getattr(root, "name", "AnimatedModel"))
    return export_path.with_name(f"{export_path.stem}_animated").with_suffix("") / f"{safe_name}.glb"


def _safe_filename(value):
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe.strip("_") or "AnimatedModel"


def _path_for_metadata(path, project_root):
    try:
        return str(path.resolve().relative_to(project_root.resolve()).as_posix())
    except Exception:
        return str(path)


def _export_selected_objects(objects, export_path, active_object):
    previous_selection = list(getattr(bpy.context, "selected_objects", []) or [])
    previous_active = getattr(getattr(bpy.context, "view_layer", None), "objects", None)
    previous_active_object = getattr(previous_active, "active", None) if previous_active else None
    previous_matrix = _copy_matrix(getattr(active_object, "matrix_world", None))

    try:
        for obj in getattr(bpy.context.scene, "objects", []) or []:
            if hasattr(obj, "select_set"):
                obj.select_set(False)
        for obj in objects:
            if hasattr(obj, "select_set"):
                obj.select_set(True)
        if previous_active:
            previous_active.active = active_object
        _set_identity_matrix(active_object)
        bpy.ops.export_scene.gltf(**_gltf_export_kwargs(export_path, selected=True))
    finally:
        if previous_matrix is not None:
            active_object.matrix_world = previous_matrix
            _update_view_layer()
        for obj in getattr(bpy.context.scene, "objects", []) or []:
            if hasattr(obj, "select_set"):
                obj.select_set(False)
        for obj in previous_selection:
            if hasattr(obj, "select_set"):
                obj.select_set(True)
        if previous_active:
            previous_active.active = previous_active_object


def _copy_matrix(matrix):
    if matrix is None:
        return None
    return matrix.copy() if hasattr(matrix, "copy") else matrix


def _set_identity_matrix(obj):
    try:
        from mathutils import Matrix
        obj.matrix_world = Matrix.Identity(4)
        _update_view_layer()
    except Exception:
        pass


def _update_view_layer():
    view_layer = getattr(bpy.context, "view_layer", None)
    update = getattr(view_layer, "update", None)
    if update:
        update()


def _run_command(prefs, project_root):
    python_command, runtime_source = _resolve_runtime_python(prefs, project_root)
    main_script = _resolve_path_text(prefs.main_script, project_root)
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")
    if not main_script.is_file():
        raise FileNotFoundError(f"Main script is not a file: {main_script}")
    command = [str(python_command), str(main_script)]
    if prefs.run_args.strip():
        command.extend(shlex.split(prefs.run_args))
    return command, runtime_source, main_script


def _launch_command(command):
    if os.name != "nt":
        return command
    return ["cmd", "/k", _windows_command_line(command)]


def _windows_command_line(command):
    return subprocess.list2cmdline([str(part) for part in command])


def _resolve_runtime_python(prefs, project_root):
    raw = (prefs.python_command or "python").strip()
    manual_override = _is_manual_python_command(raw)

    if prefs.auto_detect_venv and not manual_override:
        detected = _detect_project_venv(project_root)
        if detected:
            python_path, venv_name = detected
            return python_path, f"{venv_name} (auto-detected)"

    python_command = _resolve_python_command(raw, project_root)
    if manual_override or not prefs.auto_detect_venv:
        return python_command, "manual override"
    return python_command, "system python"


def _runtime_python_label(prefs):
    project_root = Path(bpy.path.abspath(prefs.project_root)).resolve() if prefs.project_root else Path.cwd()
    try:
        python_command, runtime_source = _resolve_runtime_python(prefs, project_root)
        return f"{python_command} ({runtime_source})"
    except Exception as exc:
        return f"unresolved: {exc}"


def _is_manual_python_command(value):
    raw = (value or "").strip().strip('"')
    return bool(raw) and raw not in {"python", "py", "python.exe", "py.exe"}


def _detect_project_venv(project_root):
    for name in (".venv", "venv", "env"):
        venv_path = project_root / name
        python_path = _venv_python_path(venv_path)
        if python_path:
            return python_path, name
    return None


def _venv_python_path(venv_path):
    if not venv_path.is_dir():
        return None
    candidates = [
        venv_path / "Scripts" / "python.exe",
        venv_path / "bin" / "python",
        venv_path / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_python_command(value, project_root):
    raw = (value or "python").strip().strip('"')
    path = Path(bpy.path.abspath(raw))
    if not path.is_absolute():
        path = project_root / path

    if path.is_dir():
        python_path = _venv_python_path(path)
        if python_path:
            return python_path

    if path.name.lower() in {"activate", "activate.bat", "activate.ps1"}:
        candidate = path.parent / "python.exe"
        if candidate.exists():
            return candidate

    if path.exists():
        return path

    if raw in {"python", "py", "python.exe", "py.exe"}:
        return raw

    raise FileNotFoundError(f"Python command not found: {path}")


def _resolve_path_text(value, project_root):
    path = Path(bpy.path.abspath((value or "").strip().strip('"')))
    if not path.is_absolute():
        path = project_root / path
    return path


def _load_discovery_module(project_root):
    discovery_path = project_root / "EasyCells3D" / "ComponentDiscovery.py"
    spec = importlib.util.spec_from_file_location("easycells3d_component_discovery", discovery_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_prop_default(obj, key, default):
    if key not in obj:
        obj[key] = default


def _default_for(metadata):
    if "default" in metadata and metadata["default"] is not None:
        return metadata["default"]
    field_type = metadata.get("type", "Any")
    if field_type == "bool":
        return False
    if field_type == "int":
        return 0
    if field_type == "float":
        return 0.0
    if field_type in {"str", "string", "item", "component"}:
        return ""
    return ""


def _typed_value(value, metadata):
    field_type = metadata.get("type", "Any")
    try:
        if field_type == "bool":
            return bool(value)
        if field_type == "int":
            return int(value)
        if field_type == "float":
            return float(value)
    except (TypeError, ValueError):
        return _default_for(metadata)
    return value


def _ui_value(value, field):
    ref = _field_ref(field)
    if isinstance(value, dict) and ref in {"item", "component"}:
        return value.get("$ref") or value.get("$id") or ""
    return _typed_value(value, field)


def _export_field_value(value, field, component_type=""):
    ref = _field_ref(field)
    if ref == "item" and value:
        obj = bpy.data.objects.get(str(value))
        if obj:
            return {"$id": ensure_easycells_id(obj)}
        return {"$ref": str(value)}
    if ref == "component" and value:
        obj = bpy.data.objects.get(str(value))
        output = {"$component": str(component_type)}
        if obj:
            output["$id"] = ensure_easycells_id(obj)
        else:
            output["$ref"] = str(value)
        return output
    return _typed_value(value, field)


def _field_ref(field):
    return field.get("ref") or field.get("type")


def _component_ref_type(value):
    if isinstance(value, dict):
        return value.get("$component") or value.get("component") or value.get("type") or ""
    return ""


def _migrate_component(component):
    if "config" in component and "fields" not in component and "args" not in component:
        component = dict(component)
        component["args"] = {}
        component["fields"] = component.pop("config") or {}
    component.setdefault("args", {})
    component.setdefault("fields", {})
    return component


classes = (
    EC3D_AddonPreferences,
    EC3D_OT_refresh_components,
    EC3D_OT_add_component,
    EC3D_OT_remove_component,
    EC3D_OT_select_component,
    EC3D_OT_sync_components,
    EC3D_OT_export_scene,
    EC3D_OT_ensure_ids,
    EC3D_PT_viewport,
    EC3D_PT_components,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ec3d_component_to_add = bpy.props.EnumProperty(
        name="Component",
        items=component_items,
    )
    bpy.types.Scene.ec3d_selected_component_index = bpy.props.IntProperty(default=0)


def unregister():
    del bpy.types.Scene.ec3d_component_to_add
    del bpy.types.Scene.ec3d_selected_component_index
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
