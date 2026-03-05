import os
import sys
import importlib
import inspect
from pathlib import Path

from EasyCells3D.Components import Component
from EasyCells3D.Game import Game

# Caso queira filtrar apenas classes que herdam do seu Componente base:
# from EasyCells3D.Components.Component import Component 

# Variáveis globais de estado do level
_module_mtimes = {}
_component_registry = {}
_last_check_time = 0


def init(game: Game):
    print("Iniciando Hot-Reloader de Componentes...")
    # Força a primeira leitura de todos os arquivos na inicialização
    _scan_and_reload(force_print=True)


def loop(game: Game):
    global _last_check_time

    # Checa a pasta a cada 500ms para não sobrecarregar o HD/SSD
    if game.time - _last_check_time > 5000:
        _scan_and_reload()
        _last_check_time = game.time


def _scan_and_reload(force_print=False):
    components_dir = Path("UserComponents")

    # Se a pasta não existir, avisa e cancela
    if not components_dir.exists():
        print(f"Aviso: Pasta '{components_dir}' não encontrada na raiz do projeto.")
        return

    changed = False

    # Varre todos os arquivos .py dentro de UserComponents
    for filepath in components_dir.glob("*.py"):
        if filepath.name == "__init__.py":
            continue

        mtime = os.path.getmtime(filepath)
        module_name = f"UserComponents.{filepath.stem}"

        # Verifica se o arquivo é novo ou se foi alterado desde a última checagem
        if module_name not in _module_mtimes or _module_mtimes[module_name] < mtime:
            _module_mtimes[module_name] = mtime
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
                _inspect_module(module)

            except Exception as e:
                # Se houver erro de sintaxe no código salvo, não derruba a engine!
                print(f"[❌ Erro] Falha ao compilar {filepath.name}:\n    {e}")

    # Se alguma coisa mudou, printa o relatório atualizado
    if changed or force_print:
        _print_registry()


def _inspect_module(module):
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
                _component_registry[name] = args_info

            except ValueError:
                # Caso a classe não tenha um __init__ explícito
                _component_registry[name] = []


def _print_registry():
    print("\n" + "=" * 50)
    print("📦 COMPONENTES DISPONÍVEIS NA ENGINE")
    print("=" * 50)

    if not _component_registry:
        print("  Nenhum componente encontrado.")

    for comp_name, args in _component_registry.items():
        args_str = ", ".join(args)
        print(f"🔹 {comp_name}({args_str})")

    print("=" * 50 + "\n")