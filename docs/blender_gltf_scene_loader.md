# Blender GLB/GLTF Scene Loader

`SceneLoader` carrega arquivos `.glb` ou `.gltf` exportados do Blender, cria um `Item` para cada node da cena, preserva hierarquia/transforms e adiciona componentes declarados nas Custom Properties.

O loader tambem interpreta alguns componentes nativos do glTF exportados pelo Blender:

- nodes com `mesh` recebem `StaticModel`;
- nodes com `camera` recebem `Camera3D`;
- nodes com `KHR_lights_punctual` recebem `Light3D`.
- nodes com `easycells_animated_model` recebem `AnimatedModel` e `Animator3D`.

`Light3D` preserva tipo, cor, intensidade, alcance e cones de spot light, mas ainda nao ilumina os modelos renderizados. Ele existe como ponte de dados para um pipeline futuro de shader/iluminacao dinamica.

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
   - `Auto-detect Project Venv`: usa automaticamente `.venv`, `venv` ou `env` na raiz do projeto quando disponivel;
   - `Python Command`: comando usado para rodar o projeto quando voce quiser sobrescrever a deteccao automatica;
   - `Run Arguments`: argumentos extras opcionais.

Quando a auto-deteccao estiver ligada e `Python Command` continuar como `python`, o add-on prioriza:

1. `.venv`
2. `venv`
3. `env`

No Windows ele procura `Scripts/python.exe`; em Linux/macOS ele procura `bin/python`. O painel mostra o runtime efetivo, por exemplo `Runtime: .venv/Scripts/python.exe (.venv auto-detected)`. Para forcar outro interpretador, desative `Auto-detect Project Venv` ou preencha `Python Command` com outro caminho/comando.

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
- `Force Sync Components`
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
4. habilita Custom Properties/extras;
5. habilita exportacao de cameras, luzes e animacoes quando o exportador glTF do Blender expuser essas opcoes;
6. exporta armatures animadas como GLBs separados e grava no objeto raiz a metadata `easycells_animated_model`.

O botao `Export & Run` faz tudo acima e executa o script principal configurado, usando a raiz do projeto como working directory.

No Windows, `Export & Run` abre o jogo em um console persistente. Se o jogo falhar por dependencia faltando, Python errado ou traceback em `main.py`, o console permanece aberto para mostrar o erro em vez de fechar imediatamente.

`Force Sync Components` existe como acao manual de diagnostico. Normalmente voce nao precisa clicar nele antes de exportar, porque `Export` e `Export & Run` ja sincronizam automaticamente os componentes antes de gerar o GLB.

## Checklist

- O componente nao aparece no Blender: confira `Project Root` e clique em `Refresh Components`.
- O componente aparece, mas nao e criado no jogo: confira se a classe herda de `Component` e esta em um dos pacotes escaneados.
- O valor nao foi exportado: use `Export` ou `Export & Run`, que sincronizam os componentes automaticamente antes da exportacao.
- O jogo abre e fecha rapido: use `Export & Run`; no Windows ele mantem o console aberto para mostrar o traceback.
- O jogo usa o Python errado: confira o texto `Runtime` no painel, desative `Auto-detect Project Venv` ou ajuste `Python Command`.
- Referencia quebrou apos renomear objeto: reexporte pelo add-on para garantir que o `easycells_id` esteja salvo.
- Dependencias do jogo faltam no Blender: o refresh usa AST e nao deve importa-las; se falhar, confira se `EasyCells3D/ComponentDiscovery.py` existe na raiz configurada.

## Observacao Sobre Render de GLB

O loader cria os `Items` e adiciona `StaticModel` nos nodes com mesh. Para cenas GLTF/GLB, o `StaticModel` usa cache compartilhado por arquivo e desenha as meshes/primitives correspondentes ao node carregado. O codigo aceita os nomes de campos usados por diferentes versoes do binding `pyray`, como `mesh_count`/`meshCount`, `material_count`/`materialCount` e `mesh_material`/`meshMaterial`.

Se uma versao futura do binding nao expuser acesso direto a `model.meshes`/`model.materials`, o `StaticModel` ainda pode cair para `draw_model_ex` como fallback de compatibilidade.

## Componentes Nativos do Blender

Ao exportar pelo glTF do Blender, cameras e luzes entram como recursos nativos do arquivo, nao como Custom Properties da EasyCells3D. O `SceneLoader` le esses campos diretamente:

- Camera perspective: usa `cameras[index].perspective.yfov` e converte de radianos para graus em `Camera3D.vfov`.
- Camera orthographic: cria `Camera3D` com `CAMERA_ORTHOGRAPHIC` e usa `ymag * 2` como tamanho inicial. O Raylib nao expoe todos os mesmos parametros de camera do glTF, entao `znear`, `zfar`, `xmag` e aspect ratio ainda nao sao reproduzidos com fidelidade completa.
- Luz point, spot e directional/sun: usa a extensao `KHR_lights_punctual` e cria `Light3D`.
- Personagem/armature animado: usa `easycells_animated_model` e cria `AnimatedModel` + `Animator3D`.

Se o mesmo objeto tiver um componente `Camera3D` ou `Light3D` declarado manualmente nas Custom Properties, o loader respeita a declaracao manual e nao cria o componente nativo duplicado.

Objetos `Empty` continuam virando apenas `Item`s com transform. Armatures, skins, constraints, fisica, audio, particles, light probes e world settings ainda nao sao convertidos automaticamente; para esses casos, use componentes EasyCells3D nas Custom Properties ou exporte a geometria ja convertida para mesh.

Se uma camera ou luz existente no Blender nao aparecer no jogo, confira o GLB reexportado: ele precisa conter `cameras` para cameras e `KHR_lights_punctual` para luzes. Recarregue/reinstale o add-on apos atualizar `tools/blender/easycells3d_components.py`, porque o Blender pode manter a versao antiga carregada em memoria.

## Animacoes 3D

O fluxo de animacao segue o modelo Unity-like:

- a cena principal continua sendo layout/cenario;
- cada armature com meshes filhos vira um asset GLB separado, mesmo antes de ter Actions configuradas;
- o objeto raiz da armature na cena recebe `AnimatedModel` e `Animator3D`;
- os nomes das Actions/NLA strips do Blender entram como `clip_names`, permitindo chamar `animator.play("Walk")`, `animator.play("Run")`, etc.

Se a armature ainda nao tiver Actions/NLA strips, o asset separado ainda sera criado para manter o posicionamento correto de personagem rigado/skinned; nesse caso o `Animator3D` existira, mas nao tera clips para tocar ate que animacoes sejam adicionadas no Blender e reexportadas.

O add-on exporta esses assets em uma pasta ao lado do GLB principal, por exemplo:

```text
Assets/Blender/scene.glb
Assets/Blender/scene_animated/PlayerRig.glb
```

No GLB principal, a metadata fica no objeto raiz animado:

```json
{
  "easycells_animated_model": {
    "path": "Assets/Blender/scene_animated/PlayerRig.glb",
    "clip_names": ["Walk", "Run", "Jump"],
    "autoplay": true
  }
}
```

Quando essa metadata existe, o `SceneLoader` nao cria `StaticModel` para os meshes filhos do personagem, evitando render duplicado. Componentes customizados no root continuam funcionando normalmente.

Componentes presos a ossos especificos, como arma na mao ou hitbox no pe, ainda precisam de uma etapa futura (`BoneSocket`/`BoneAttachment`). Por enquanto, coloque gameplay scripts e colisores principais no root do personagem.
