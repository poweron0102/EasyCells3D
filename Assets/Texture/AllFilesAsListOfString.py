# Printa todos os arquivos no diretoriao atual como uma lista de strings
import os
def list_files_in_directory(directory: str) -> list[str]:
    """Retorna uma lista de nomes de arquivos no diretório especificado."""
    try:
        return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        print(f"O diretório '{directory}' não foi encontrado.")
        return []
    except PermissionError:
        print(f"Permissão negada para acessar o diretório '{directory}'.")
        return []
# Exemplo de uso
if __name__ == "__main__":
    files = list_files_in_directory(".")
    print(files)