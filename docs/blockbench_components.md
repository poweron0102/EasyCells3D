# Blockbench EasyCells3D Components

O plugin do Blockbench fica em:

```text
tools/blockbench/easycells3d_components.js
```

Ele permite configurar componentes EasyCells3D em `Cube`s e `Group`s do Blockbench e exportar uma cena JSON para a engine.

## Instalacao

No Blockbench:

1. Abra `File > Plugins`.
2. Use `Load Plugin from File`.
3. Selecione `tools/blockbench/easycells3d_components.js`.

As acoes aparecem nos menus `File`, `File > Export` e `Edit`.

## Configuracao

Abra:

```text
File > EasyCells3D Settings
```

Campos principais:

- `Project Root`: raiz do projeto EasyCells3D;
- `Python Command`: Python usado para descobrir componentes, por exemplo `.venv\Scripts\python.exe`;
- `Main Script`: por exemplo `main.py`;
- `Scene Export Path`: por exemplo `Assets/Blockbench/scene.ec3d.json`;
- `Visual GLB Export Path`: por exemplo `Assets/Blockbench/scene.glb`;
- `Blockbench Units Per EasyCells Unit`: por padrao `16`;
- `Run Arguments`: argumentos extras opcionais.

O plugin aceita a pasta da venv ou o `python.exe` diretamente. Exemplo:

```text
C:\Users\natha\PycharmProjects\EasyCells3D\.venv
```

ou:

```text
C:\Users\natha\PycharmProjects\EasyCells3D\.venv\Scripts\python.exe
```

## Descoberta de Componentes

Use:

```text
File > Refresh EasyCells3D Components
```

O plugin chama:

```text
tools/blockbench/discover_components.py
```

Esse script usa `ComponentDiscovery` em modo `ast`, entao ele le os arquivos Python sem executar o codigo do jogo dentro do Blockbench.

## Editando Componentes

Selecione um `Cube` ou `Group` e use:

- `Edit > Add EasyCells3D Component`
- `Edit > Edit EasyCells3D Components`
- `Edit > Remove EasyCells3D Component`

O plugin grava duas propriedades persistentes nos objetos do Blockbench:

- `easycells_id`;
- `easycells_components`.

Essas propriedades usam a API `Property` do Blockbench, entao sao salvas no `.bbmodel`, entram no Undo e acompanham copy/paste/duplicacao.

## Exportando

Use:

```text
File > Export > Export EasyCells3D Scene
```

ou:

```text
File > Export > Export EasyCells3D Scene & Run
```

O plugin exporta dois arquivos:

- um `.glb` com o visual do Blockbench;
- um `.ec3d.json` com IDs, hierarquia logica, transforms e componentes.

O arquivo JSON tem formato:

```json
{
  "format": "easycells3d.blockbench.scene",
  "version": 1,
  "visual": {
    "model": "Assets/Blockbench/scene.glb",
    "kind": "glb",
    "match_by": ["easycells_id", "blockbench_uuid", "name"]
  },
  "unit_scale": 16,
  "nodes": [
    {
      "id": "abc",
      "name": "Cube",
      "parent": null,
      "kind": "cube",
      "translation": [0, 0, 0],
      "rotation_euler_degrees": [0, 0, 0],
      "scale": [1, 1, 1],
      "components": [
        {
          "type": "RotatingObj",
          "args": {},
          "fields": {
            "speed": 45
          }
        }
      ]
    }
  ]
}
```

## Carregando no Jogo

Use `BlockbenchSceneLoader`:

```python
from EasyCells3D import BlockbenchSceneLoader


def init(game):
    BlockbenchSceneLoader(game).load("Assets/Blockbench/scene.ec3d.json")
```

O loader le o campo `visual.model`, carrega o `.glb` com `SceneLoader`, tenta casar os objetos visuais por `easycells_id`, `blockbench_uuid` ou `name`, e aplica os componentes do JSON nos `Item`s visuais correspondentes.

Tambem e possivel passar o `.glb` explicitamente:

```python
BlockbenchSceneLoader(game).load(
    "Assets/Blockbench/scene.ec3d.json",
    model_path="Assets/Blockbench/scene.glb",
)
```

O fluxo de componentes e igual ao `SceneLoader`: primeiro cria todos os itens e componentes, depois aplica os campos `SerializeField`.

## Limites Atuais

O plugin tenta chamar automaticamente o codec glTF/GLB do Blockbench. Se a API da sua versao do Blockbench nao expuser compilacao binaria para plugins, o `.ec3d.json` ainda sera exportado e o plugin mostrara uma mensagem pedindo para exportar o `.glb` manualmente no mesmo caminho configurado.

O casamento entre JSON e GLB depende principalmente de nomes consistentes no Blockbench. Se dois objetos tiverem o mesmo nome, renomeie-os para nomes unicos antes de exportar.
