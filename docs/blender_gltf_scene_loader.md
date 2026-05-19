# Blender GLB/GLTF Scene Loader

`SceneLoader` carrega arquivos `.glb` ou `.gltf` exportados do Blender, cria um `Item` para cada node da cena, preserva hierarquia/transforms e adiciona componentes declarados nas Custom Properties.

Fluxo basico no level:

```python
from EasyCells3D import Game, SceneLoader


def init(game: Game):
    SceneLoader(game).load("Assets/Blender/example_scene.glb")
```

O loader chama `ComponentRegistry.ensure_discovered()` automaticamente. Para cenas carregadas pelo `SceneLoader`, nao e mais necessario fazer:

```python
ComponentRegistry.register("RotatingObj", RotatingObj)
```

## Instalando o Add-on

O add-on fica em:

```text
tools/blender/easycells3d_components.py
```

No Blender:

1. Abra `Edit > Preferences > Add-ons`.
2. Clique em `Install...`.
3. Selecione `tools/blender/easycells3d_components.py`.
4. Ative `EasyCells3D Components`.
5. Abra as preferencias do add-on e configure:
   - `Project Root`: raiz do projeto EasyCells3D;
   - `Main Script`: por exemplo `main.py`;
   - `Default Export Path`: por exemplo `Assets/Blender/FoodPack.glb`;
   - `Python Command`: comando usado para rodar o projeto;
   - `Run Arguments`: argumentos extras opcionais.

## Descobrindo Componentes

No Blender, use `N > EasyCells3D > Refresh Components`.

O add-on usa `EasyCells3D/ComponentDiscovery.py` em modo `ast`, lendo os arquivos `.py` como texto. Isso evita executar codigo do jogo dentro do Python embutido do Blender, que pode nao ter `pyray`, audio, OpenGL ou outras dependencias.

Pacotes escaneados:

- `EasyCells3D.Components`
- `EasyCells3D.PhysicsComponents`
- `EasyCells3D.NetworkComponents`
- `UserComponents`

O script [tools/generate_blender_component_config.py](../tools/generate_blender_component_config.py) ainda existe como cache/diagnostico, mas nao depende mais de lista hardcoded.

## Declarando Campos Editaveis

Use `SerializeField` para campos opcionais editaveis no Blender:

```python
import math

from EasyCells3D.Components import Component
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.Serialization import SerializeField


class RotatingObj(Component):
    speed = SerializeField(default=1.0)

    def init(self):
        self.speed = math.radians(float(self.speed))

    def loop(self):
        angle = self.speed * self.game.delta_time
        delta = Quaternion.from_axis_angle(Vec3(0.0, 1.0, 0.0), angle)
        self.transform.rotation = (delta * self.transform.rotation).normalize()
```

Campos `SerializeField`:

- aparecem na UI do Blender;
- sao opcionais;
- recebem valor padrao;
- sao aplicados depois que todos os componentes da cena foram instanciados;
- nao entram no `__init__`.

Argumentos obrigatorios devem ficar no `__init__`:

```python
class Spawner(Component):
    spawn_delay = SerializeField(default=2.0)

    def __init__(self, enemy_type: str):
        self.enemy_type = enemy_type
```

## Referencias

Referencia para outro objeto:

```python
class FollowTarget(Component):
    target = SerializeField(default=None, ref="item")
```

O add-on garante uma Custom Property `easycells_id` com UUID em cada objeto. No carregamento, o `SceneLoader` resolve referencias por `easycells_id` e usa o nome do objeto como fallback.

Formato exportado:

```json
{
  "components": [
    {
      "type": "FollowTarget",
      "args": {},
      "fields": {
        "target": {
          "$id": "6f49f0f8cc574f15b9d43d8a9f6e7b03"
        }
      }
    }
  ],
  "easycells_id": "c4983f84aa2c43f59b0fe1f1935d9b34"
}
```

Referencia para componente tambem e aceita pelo loader quando o campo usa `ref="component"` e o valor inclui `$component`.

## Editando no Blender

Painel principal:

```text
N > EasyCells3D
```

Controles:

- `Refresh Components`
- `Ensure Stable IDs`
- `Sync Components`
- `Export`
- `Export & Run`

Painel por objeto:

```text
Object Properties > EasyCells3D Components
```

Use o dropdown para escolher o componente e clique em `Add`. O componente selecionado mostra os argumentos do `__init__` e os campos `SerializeField` como propriedades editaveis, sem editar JSON manualmente.

## Formato dos Extras GLTF

O formato atual separa argumentos de construtor e campos serializados:

```json
{
  "components": [
    {
      "type": "RotatingObj",
      "args": {},
      "fields": {
        "speed": 90
      }
    }
  ]
}
```

Com argumento obrigatorio:

```json
{
  "components": [
    {
      "type": "Spawner",
      "args": {
        "enemy_type": "Slime"
      },
      "fields": {
        "spawn_delay": 2.0
      }
    }
  ]
}
```

O formato antigo baseado em `"config"` foi aposentado. `blender_fields` e `from_config` tambem foram removidos do fluxo recomendado.

## Exportando e Rodando

O botao `Export` faz:

1. garante `easycells_id` nos objetos;
2. sincroniza a lista `components` com os valores editados na UI;
3. chama `bpy.ops.export_scene.gltf`;
4. habilita `export_custom_properties=True`.

O botao `Export & Run` faz tudo acima e executa o script principal configurado, usando a raiz do projeto como working directory.

## Checklist

- O componente nao aparece no Blender: confira `Project Root` e clique em `Refresh Components`.
- O componente aparece, mas nao e criado no jogo: confira se a classe herda de `Component` e esta em um dos pacotes escaneados.
- O valor nao foi exportado: use `Export` ou `Export & Run`, que chamam `Sync Components` antes da exportacao.
- Referencia quebrou apos renomear objeto: reexporte pelo add-on para garantir que o `easycells_id` esteja salvo.
- Dependencias do jogo faltam no Blender: o refresh usa AST e nao deve importa-las; se falhar, confira se `EasyCells3D/ComponentDiscovery.py` existe na raiz configurada.

## Limitacao Atual

O loader cria os `Items` e adiciona `StaticModel` nos nodes com mesh. Para cenas GLTF/GLB, o `StaticModel` usa cache compartilhado por arquivo e tenta desenhar apenas as meshes/primitives correspondentes ao node carregado. Se a versao do binding `pyray` nao expuser acesso direto a `model.meshes`/`model.materials`, ele cai automaticamente para `draw_model_ex`, preservando compatibilidade.
