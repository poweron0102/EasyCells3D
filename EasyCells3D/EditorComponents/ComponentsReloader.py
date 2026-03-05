import importlib
import inspect
import os
import sys
import pkgutil
from pathlib import Path

from EasyCells3D.Components import Component


class ComponentsReloader(Component):

    def __init__(self, reload_time: int = 5000, load_core: bool = True, load_physics: bool = True, load_network: bool = True):
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
        if self.item.game.time - self.last_check_time > 5000:
            self.scan_and_reload()
            self.last_check_time = self.item.game.time

    def register_internal_components(self):
        """
        Registra componentes internos da engine (Core, Physics, Network) uma única vez no init.
        """
        packages_to_load = []
        if self.load_core: packages_to_load.append("EasyCells3D.Components")
        if self.load_physics: packages_to_load.append("EasyCells3D.PhysicsComponents")
        if self.load_network: packages_to_load.append("EasyCells3D.NetworkComponents")

        for package_name in packages_to_load:
            try:
                # Importa o pacote base
                package = importlib.import_module(package_name)
                self.inspect_module(package)

                # Se for um pacote (tem __path__), varre todos os submódulos para encontrar componentes
                if hasattr(package, "__path__"):
                    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                        try:
                            submodule = importlib.import_module(name)
                            self.inspect_module(submodule)
                        except Exception as e:
                            print(f"[Aviso] Falha ao carregar submódulo interno {name}: {e}")
            except ImportError:
                # Ignora silenciosamente se o pacote (ex: Physics ou Network) não estiver instalado no ambiente
                pass

    def scan_and_reload(self, force_print=False):
        components_dir = Path("UserComponents")

        # Se a pasta não existir, avisa e cancela
        if not components_dir.exists():
            print(f"Aviso: Pasta '{components_dir}' não encontrada na raiz do projeto.")
            return

        changed = False

        # Varre todos os arquivos .py dentro de UserComponents e subdiretórios
        for filepath in components_dir.rglob("*.py"):
            if filepath.name == "__init__.py":
                continue

            mtime = os.path.getmtime(filepath)
            rel_path = filepath.relative_to(components_dir).with_suffix('')
            module_name = f"UserComponents.{'.'.join(rel_path.parts)}"

            # Verifica se o arquivo é novo ou se foi alterado desde a última checagem
            if module_name not in self.module_mtimes or self.module_mtimes[module_name] < mtime:
                self.module_mtimes[module_name] = mtime
                changed = True

                try:
                    if module_name in sys.modules:
                        # O arquivo já existia, então fazemos o HOT-RELOAD
                        module = importlib.reload(sys.modules[module_name])
                        print(f"\n[🔄 Reload] Módulo atualizado: {filepath.name}")
                    else:
                        # O arquivo é novo, fazemos o IMPORT normal
                        module = importlib.import_module(module_name)
                        print(f"\n[🆕 Import] Novo módulo detectado: {filepath.name}")

                    # Extrai os dados das classes desse módulo
                    self.inspect_module(module)

                except Exception as e:
                    # Se houver erro de sintaxe no código salvo, não derruba a engine!
                    print(f"[❌ Erro] Falha ao compilar {filepath.name}:\n    {e}")

        # Se alguma coisa mudou, printa o relatório atualizado
        if changed or force_print:
            self.print_registry()

    def inspect_module(self, module):
        # inspect.getmembers pega todas as classes dentro do arquivo
        for name, obj in inspect.getmembers(module, inspect.isclass):

            # Garante que a classe foi escrita *neste* arquivo e não apenas importada de outro lugar
            if obj.__module__ == module.__name__:

                # Garantir que é um Componente da engine:
                if not issubclass(obj, Component): continue

                try:
                    # Pega a assinatura (parâmetros) do método __init__
                    sig = inspect.signature(obj.__init__)
                    args_info = []

                    for param_name, param in sig.parameters.items():
                        if param_name == "self":
                            continue

                        # Tenta ler a anotação de tipo (ex: speed: float)
                        tipo = "Any"
                        if param.annotation != inspect._empty:
                            # Pega o nome do tipo (ex: 'float', 'int', 'str')
                            tipo = getattr(param.annotation, '__name__', str(param.annotation))

                        args_info.append(f"{param_name}: {tipo}")

                    # Salva no registro da engine
                    self.component_registry[name] = args_info

                except ValueError:
                    # Caso a classe não tenha um __init__ explícito
                    self.component_registry[name] = []

    def print_registry(self):
        print("\n" + "=" * 50)
        print("📦 COMPONENTES DISPONÍVEIS NA ENGINE")
        print("=" * 50)

        if not self.component_registry:
            print("  Nenhum componente encontrado.")

        for comp_name, args in self.component_registry.items():
            args_str = ", ".join(args)
            print(f"🔹 {comp_name}({args_str})")

        print("=" * 50 + "\n")