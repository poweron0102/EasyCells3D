# Blender GLB/GLTF Scene Loader

`SceneLoader` carrega arquivos `.glb` ou `.gltf` exportados do Blender e cria um `Item` para cada node da cena.

```python
from EasyCells3D import ComponentRegistry, SceneLoader
from UserComponents.ratating_obj import RotatingObj

ComponentRegistry.register("RotatingObj", RotatingObj)
SceneLoader(game).load("Assets/Blender/example_scene.glb")
```

Cada node preserva:

- `name`;
- `translation`, `rotation`, `scale` ou `matrix`;
- hierarquia pai/filho;
- componentes declarados em Custom Properties exportadas como `extras`.

## Registro dos Componentes no Python

O nome usado no Blender precisa estar registrado antes de carregar a cena:

```python
from EasyCells3D import ComponentRegistry
from UserComponents.ratating_obj import RotatingObj

ComponentRegistry.register("RotatingObj", RotatingObj)
```

Sem esse registro, o loader mostra um aviso e continua carregando os outros objetos.

## Configurando Componentes para o Blender

O plugin do Blender le um arquivo JSON em `tools/blender/easycells3d_components.json`. Esse arquivo diz quais componentes aparecem na interface do Blender e quais campos entram na config inicial.

O gerador [tools/generate_blender_component_config.py](../tools/generate_blender_component_config.py) cria esse JSON lendo uma lista explicita de classes:

```python
COMPONENTS = [
    "UserComponents.ratating_obj.RotatingObj",
]
```

Para adicionar outro componente ao plugin:

1. No componente Python, declare `blender_fields`:

```python
from EasyCells3D.Components import Component


class Collider3D(Component):
    blender_fields = {
        "shape": {
            "type": "string",
            "default": "box"
        },
        "isTrigger": {
            "type": "bool",
            "default": False
        }
    }

    def __init__(self, shape="box", isTrigger=False):
        self.shape = shape
        self.isTrigger = isTrigger
```

2. Adicione a classe na lista `COMPONENTS` do gerador:

```python
COMPONENTS = [
    "UserComponents.ratating_obj.RotatingObj",
    "UserComponents.collider3d.Collider3D",
]
```

3. Rode o gerador a partir da raiz do projeto:

```powershell
python -B tools\generate_blender_component_config.py
```

Isso atualiza:

```text
tools/blender/easycells3d_components.json
```

Exemplo de saida:

```json
{
  "components": [
    {
      "name": "RotatingObj",
      "class": "UserComponents.ratating_obj.RotatingObj",
      "fields": {
        "speed": {
          "type": "float",
          "default": 1.0
        }
      }
    }
  ]
}
```

Esse JSON e usado apenas pelo plugin do Blender. Em runtime, o que importa e o `ComponentRegistry.register(...)` no seu jogo.

## Instalando o Plugin no Blender

O plugin fica em:

```text
tools/blender/easycells3d_components.py
```

Para instalar:

1. No Blender, abra `Edit > Preferences > Add-ons`.
2. Clique em `Install...`.
3. Selecione `tools/blender/easycells3d_components.py`.
4. Ative o add-on `EasyCells3D Components`.
5. Mantenha `easycells3d_components.json` na mesma pasta do `.py`, porque o plugin le esse arquivo ao mostrar a lista de componentes.

Depois disso, selecione um objeto e abra:

```text
Properties > Object Properties > EasyCells3D Components
```

O painel mostra botoes como `Add RotatingObj`. Ao clicar, o plugin grava uma Custom Property chamada `components` no objeto selecionado.

## Adicionando Componentes pelo Blender

Com o plugin instalado:

1. Selecione o objeto na cena.
2. Va em `Object Properties > EasyCells3D Components`.
3. Clique em `Add NomeDoComponente`.
4. O plugin cria/atualiza a Custom Property `components`.
5. Para remover, use o botao `X` ao lado do componente.

No estado atual, o plugin cria a config com valores padrao e mostra a config como JSON. Para alterar valores, edite a Custom Property `components` no painel de propriedades customizadas do Blender.

Por exemplo, para um objeto com `RotatingObj`, deixe a propriedade `components` assim:

```json
[
  {
    "type": "RotatingObj",
    "config": {
      "speed": 5
    }
  }
]
```

O formato completo tambem e aceito:

```json
{
  "components": [
    {
      "type": "RotatingObj",
      "config": {
        "speed": 5
      }
    }
  ]
}
```

Para multiplos componentes no mesmo objeto:

```json
[
  {
    "type": "RotatingObj",
    "config": {
      "speed": 5
    }
  },
  {
    "type": "Collider3D",
    "config": {
      "shape": "box",
      "isTrigger": false
    }
  }
]
```

## Criando Componentes a Partir da Config

Se a classe tiver `from_config(config, context)`, o loader usa esse factory. Essa e a forma recomendada quando voce precisa converter tipos, validar campos ou resolver referencias para outros objetos.

```python
class RotatingObj(Component):
    blender_fields = {
        "speed": {
            "type": "float",
            "default": 1.0
        }
    }

    @staticmethod
    def from_config(config, context):
        return RotatingObj(speed=float(config.get("speed", 1.0)))
```

Caso nao exista `from_config`, o loader tenta chamar o construtor com `Component(**config)`.

O `context` recebido pelo factory contem:

- `game`;
- `item`;
- `node`;
- `scene_path`;
- `objects_by_name`;
- `objects_by_node_index`.

## Exportacao Recomendada

No Blender:

1. Use `File > Export > glTF 2.0`.
2. Escolha `GLB` para arquivo unico ou `glTF Separate` se quiser arquivos externos.
3. Marque `Include > Custom Properties`. Sem isso, a propriedade `components` nao entra no `.glb/.gltf`.
4. Aplique transforms somente se quiser zerar os transforms no arquivo. Para preservar transforms editaveis por objeto, exporte mantendo a hierarquia.
5. Salve o arquivo dentro de `Assets`, por exemplo `Assets/Blender/example_scene.glb`.

Depois de exportar, carregue no level:

```python
from EasyCells3D import ComponentRegistry, SceneLoader
from UserComponents.ratating_obj import RotatingObj


def init(game):
    ComponentRegistry.register("RotatingObj", RotatingObj)
    SceneLoader(game).load("Assets/Blender/example_scene.glb")
```

## Checklist de Problemas Comuns

- O componente nao aparece no Blender: confira se ele esta em `COMPONENTS` no gerador e rode `python -B tools\generate_blender_component_config.py`.
- O componente aparece no Blender, mas nao e criado no jogo: confira se voce chamou `ComponentRegistry.register("Nome", Classe)`.
- A config sumiu depois do export: confira se `Include > Custom Properties` esta marcado no exportador GLTF.
- Erro de tipo na config: implemente `from_config(config, context)` e converta os valores explicitamente.
- O plugin nao atualizou a lista: rode o gerador de novo e reinicie/recarregue o add-on no Blender.

## Limitacao Atual

O loader cria os `Items` e adiciona `StaticModel` nos nodes com mesh. Para cenas GLTF/GLB, o `StaticModel` usa cache compartilhado por arquivo e tenta desenhar apenas as meshes/primitives correspondentes ao node carregado. Se a versao do binding `pyray` nao expuser acesso direto a `model.meshes`/`model.materials`, ele cai automaticamente para `draw_model_ex`, preservando compatibilidade.
