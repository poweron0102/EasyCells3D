import os
from pathlib import Path

from EasyCells3D.ComponentDiscovery import discover_components
from EasyCells3D.Components import Component


class ComponentsReloader(Component):
    def __init__(
        self,
        reload_time: int = 5000,
        load_core: bool = True,
        load_physics: bool = True,
        load_network: bool = True,
    ):
        self.reload_time = reload_time
        self.last_check_time = 0
        self.load_core = load_core
        self.load_physics = load_physics
        self.load_network = load_network

        self.component_registry = {}
        self.module_mtimes = {}

    def init(self):
        print("Iniciando Hot-Reloader de Componentes...")
        self.register_internal_components()
        self.scan_and_reload(force_print=True)
        self.last_check_time = self.item.game.time

    def loop(self):
        if self.item.game.time - self.last_check_time > self.reload_time:
            self.scan_and_reload()
            self.last_check_time = self.item.game.time

    def register_internal_components(self):
        packages_to_load = []
        if self.load_core:
            packages_to_load.append("EasyCells3D.Components")
        if self.load_physics:
            packages_to_load.append("EasyCells3D.PhysicsComponents")
        if self.load_network:
            packages_to_load.append("EasyCells3D.NetworkComponents")
        self._discover_packages(tuple(packages_to_load))

    def scan_and_reload(self, force_print=False):
        components_dir = Path("UserComponents")
        if not components_dir.exists():
            print(f"Aviso: Pasta '{components_dir}' nao encontrada na raiz do projeto.")
            return

        changed = False
        for filepath in components_dir.rglob("*.py"):
            if filepath.name == "__init__.py":
                continue

            mtime = os.path.getmtime(filepath)
            rel_path = filepath.relative_to(components_dir).with_suffix("")
            module_name = f"UserComponents.{'.'.join(rel_path.parts)}"

            if module_name not in self.module_mtimes or self.module_mtimes[module_name] < mtime:
                self.module_mtimes[module_name] = mtime
                changed = True

        if changed or force_print:
            self._discover_packages(("UserComponents",))
            self.print_registry()

    def _discover_packages(self, packages: tuple[str, ...]):
        for metadata in discover_components(project_root=Path.cwd(), mode="runtime", packages=packages):
            args_info = []
            for arg in metadata.required_args + metadata.optional_args:
                suffix = "" if arg.required else f" = {arg.default!r}"
                args_info.append(f"{arg.name}: {arg.type}{suffix}")
            for name, field in metadata.fields.items():
                args_info.append(f"{name}: {field.type} = {field.default!r}")
            self.component_registry[metadata.name] = args_info

    def print_registry(self):
        print("\n" + "=" * 50)
        print("COMPONENTES DISPONIVEIS NA ENGINE")
        print("=" * 50)

        if not self.component_registry:
            print("  Nenhum componente encontrado.")

        for comp_name, args in self.component_registry.items():
            args_str = ", ".join(args)
            print(f"- {comp_name}({args_str})")

        print("=" * 50 + "\n")
