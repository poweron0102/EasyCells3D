# Guia de Desenvolvimento com EasyCells3D

EasyCells3D e uma biblioteca Python para desenvolvimento de jogos inspirada no modelo de trabalho da Unity. A ideia central e montar uma cena com objetos (`Item`) e anexar comportamentos (`Component`) a esses objetos. Cada objeto tem um `Transform`, e o jogo roda um ciclo de atualizacao e renderizacao usando Raylib via `pyray`.

Este documento explica as correlacoes com Unity e mostra como desenvolver usando a estrutura atual do projeto.

## Correlacoes com Unity

| Unity | EasyCells3D | Papel |
| --- | --- | --- |
| `Scene` | modulo em `Levels/` | Um level/cena Python com funcoes `init(game)` e `loop(game)`. |
| `GameObject` | `Item` | Objeto da cena. Pode ter componentes, filhos e transform. |
| `MonoBehaviour` | `Component` | Classe base para scripts de comportamento. Usa `init()` e `loop()`. |
| `Transform` | `Transform` | Posicao, rotacao e escala do objeto. Existe em todo `Item`. |
| `Start()` / `Awake()` | `Component.init()` | Chamado depois que o componente e adicionado ao item. |
| `Update()` | `Component.loop()` e `level.loop(game)` | Chamado a cada frame. |
| `AddComponent<T>()` | `item.AddComponent(Component())` | Anexa um componente ao objeto. |
| `GetComponent<T>()` | `item.GetComponent(Classe)` ou `self.GetComponent(Classe)` | Busca componente no item e, se necessario, em filhos. |
| `Destroy(gameObject)` | `item.Destroy()` | Remove o item, filhos e chama `on_destroy()` dos componentes. |
| `OnDestroy()` | `Component.on_destroy()` | Limpeza de recursos. |
| `Time.deltaTime` | `game.delta_time` | Tempo do frame em segundos. |
| `Camera` | `Camera2D` / `Camera3D` | Componentes de camera e renderizacao. |
| `SpriteRenderer` | `Sprite` | Renderizacao de sprites 2D. |
| `Animator` | `Animator2D` + `Animation2D` | Troca frames de spritesheet por tempo. |
| `Rigidbody2D` | `Rigidbody` | Corpo fisico 2D. |
| `Collider2D` | `Collider`, `RectCollider`, `TileMapCollider` | Colisao 2D por poligonos/SAT. |
| `Prefab` | funcao factory em `UserComponents/` | Exemplo: `load_player(game, folder_name)`. |
| `Coroutine` | `Scheduler` async | Executa coroutines async em fatias por frame. |
| `DontDestroyOnLoad` | `item.destroy_on_load = False` | Mantem o item ao trocar de level. |

## Estrutura do projeto

```text
EasyCells3D/
  Game.py                 Runtime principal, janela, troca de level e loop.
  Geometry.py             Vec2, Vec3 e Quaternion.
  scheduler.py            Coroutines async por frame e Tick.
  Components/             Componentes base, cameras, sprites, tilemaps, 3D.
  PhysicsComponents/      Colliders, Rigidbody e fisica 2D.
  NetworkComponents/      NetworkManager, RPC, NetworkVariable e NetworkTransform.
  UiComponents/           Componentes de UI.

Levels/
  platform.py             Exemplo 2D com tilemap, player e fisica.
  solar.py                Exemplo 3D com camera livre, esfera e filhos.
  test_rigidbody.py       Exemplo de fisica, raycast e animacao.

UserComponents/
  ratating_obj.py         Exemplo de componente customizado 3D.
  platform/Player.py      Exemplo de player, input, animacao e movimento.

Assets/
  imagens, spritesheets, audio, modelos, tilesets e mapas.
```

## Criando e rodando um jogo

O ponto de entrada fica em `main.py`. Ele cria uma instancia de `Game`, informa o level inicial e chama `run()`.

```python
from EasyCells3D import Game
import Levels.platform

if __name__ == "__main__":
    game = Game(
        Levels.platform,
        "Platform",
        show_fps=True,
        screen_resolution=(1280, 720),
        dynamic_resolution=True,
    )
    game.run()
```

Tambem e possivel passar o nome do level como string se ele estiver dentro de `Levels/`:

```python
game = Game("platform", "Platform")
```

## Criando um level

Um level e um modulo Python dentro de `Levels/` com duas funcoes:

```python
from EasyCells3D import Game

def init(game: Game):
    # chamado quando a cena carrega
    pass

def loop(game: Game):
    # chamado todo frame
    pass
```

Use `init()` para criar objetos, adicionar componentes e configurar a cena. Use `loop()` para regras globais do level, como reiniciar cena, detectar teclas de debug ou coordenar sistemas.

Exemplo minimo 2D:

```python
import pyray as rl

from EasyCells3D import Game
from EasyCells3D.Components import Camera2D, Sprite
from EasyCells3D.Geometry import Vec2

player = None

def init(game: Game):
    global player

    camera_item = game.CreateItem()
    camera_item.AddComponent(Camera2D())

    player = game.CreateItem()
    player.name = "Player"
    player.transform.position = Vec2(100, 100)
    player.AddComponent(Sprite("player32.png", (32, 32)))

def loop(game: Game):
    if rl.is_key_down(rl.KeyboardKey.KEY_D):
        player.transform.x += 120 * game.delta_time
```

## Items, componentes e Transform

`Item` e o equivalente a um GameObject. Ele guarda:

- `transform`: posicao, rotacao e escala local.
- `global_transform`: transform global calculado considerando os pais.
- `components`: componentes anexados.
- `children`: filhos na hierarquia.
- `destroy_on_load`: se `True`, o item e destruido ao trocar de level.

Criacao basica:

```python
item = game.CreateItem()
item.name = "Enemy"
item.transform.x = 50
item.transform.y = 120
item.transform.z = 1
item.transform.scale.x = 2
item.transform.scale.y = 2
```

Criando hierarquia:

```python
parent = game.CreateItem()
child = parent.CreateChild()
child.transform.x = 4
```

O filho usa transform local. O `global_transform` e calculado no update, parecido com a hierarquia da Unity.

## Criando componentes customizados

Componentes customizados devem herdar de `Component`.

```python
from EasyCells3D.Components import Component

class Mover(Component):
    def __init__(self, speed: float):
        self.speed = speed

    def init(self):
        # chamado depois que o componente entra no Item
        pass

    def loop(self):
        # chamado a cada frame
        self.transform.x += self.speed * self.game.delta_time

    def on_destroy(self):
        # liberar recursos, remover referencias, etc.
        pass
```

Uso:

```python
enemy = game.CreateItem()
enemy.AddComponent(Mover(80))
```

Dentro de um componente:

- `self.item` acessa o `Item`.
- `self.transform` acessa `item.transform`.
- `self.global_transform` acessa o transform global.
- `self.game` acessa a instancia de `Game`.
- `self.GetComponent(Classe)` busca outro componente no mesmo item/filhos.

Exemplo com dependencia:

```python
from EasyCells3D.Components import Component, Sprite

class FlipByDirection(Component):
    def init(self):
        self.sprite = self.GetComponent(Sprite)

    def loop(self):
        if self.transform.x < 0:
            self.sprite.horizontal_flip = True
```

## Renderizacao 2D

Para renderizar 2D, crie uma `Camera2D` e adicione componentes renderizaveis como `Sprite` e `TileMapRenderer`.

```python
from EasyCells3D.Components import Camera2D, Sprite

camera_item = game.CreateItem()
camera_item.AddComponent(Camera2D(zoom=1.0))

obj = game.CreateItem()
obj.AddComponent(Sprite("player32.png", (32, 32)))
```

`Sprite` carrega arquivos a partir de `Assets/`. Entao:

```python
Sprite("player32.png")
```

carrega:

```text
Assets/player32.png
```

Para ordenar sprites, use `transform.z`. A `Camera2D` ordena renderizaveis pela posicao `z`.

```python
background.transform.z = -10
player.transform.z = 1
ui_marker.transform.z = 10
```

## Animacao 2D

`Animator2D` controla o `Sprite` alterando `index_x` e `index_y` conforme uma tabela de animacoes.

```python
from EasyCells3D.Components import Animation2D, Animator2D, Sprite

player.AddComponent(Sprite("player32.png", (32, 32)))
player.AddComponent(Animator2D(
    {
        "idle": Animation2D(0.2, [0, 1, 2]),
        "run": Animation2D(0.08, [3, 4, 5, 6]),
        "hit": Animation2D(0.1, [7, 8, 9], on_end="idle"),
    },
    "idle",
))
```

Trocando animacao:

```python
animator = player.GetComponent(Animator2D)
animator.current_animation = "run"
```

`Animation2D` recebe:

- `speed`: intervalo entre frames.
- `frames`: indices horizontais do spritesheet.
- `index_y`: linha da animacao no spritesheet.
- `on_end`: animacao para trocar quando terminar.

## TileMap

`TileMap` guarda uma matriz de indices. `TileMapRenderer` desenha essa matriz usando um tileset.

```python
from EasyCells3D.Components import TileMap, TileMapRenderer

tile_map = game.CreateItem()
tile_map.AddComponent(TileMap([
    [5, 4, 4, 3],
    [11, 1, 1, 9],
]))
tile_map.AddComponent(TileMapRenderer("RockSet.png", 32))
```

Para carregar CSV de `Assets/`:

```python
from EasyCells3D.Components.TileMap import matrix_from_csv

tile_map.AddComponent(TileMap(matrix_from_csv("Pixel Adventure/plataforma_mapa.csv")))
```

O valor `-1` representa tile vazio no renderer.

## Fisica 2D

A fisica usa `Rigidbody` junto com algum `Collider`. O corpo fisico precisa de um collider no mesmo item para participar de colisoes.

```python
import pyray as rl

from EasyCells3D.PhysicsComponents import Rigidbody, RectCollider

player.AddComponent(RectCollider(rl.Rectangle(0, 0, 32, 32), debug=True, mask=2))
rb = player.AddComponent(Rigidbody(
    mass=10,
    drag=2,
    use_gravity=True,
    restitution=0,
))
```

Objetos estaticos devem usar `Rigidbody(is_kinematic=True, use_gravity=False)`:

```python
floor.AddComponent(RectCollider(rl.Rectangle(0, 0, 400, 32), mask=1))
floor.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))
```

No `init()` do level, inicie a simulacao:

```python
from EasyCells3D.PhysicsComponents import SATPhysicsWorld

game.physics_world = SATPhysicsWorld()
```

Movimento por fisica:

```python
rb.add_force(Vec2(800, 0))
rb.add_impulse(Vec2(0, -340))
rb.velocity.x = 100
```

### Masks e raycasts

Colliders usam `mask` para filtrar consultas. Um raycast estatico recebe uma mascara e testa apenas colliders compatíveis:

```python
from EasyCells3D.PhysicsComponents import Collider
from EasyCells3D import Vec2

hit = game.physics_world.ray_cast(
    origin=player.transform.positionVec2,
    direction=Vec2(0, 1),
    max_distance=20,
    mask=1,
)

if hit:
    collider, point, normal = hit
```

Use `game.physics_world.rect_cast(...)` para varrer um retangulo.

## Renderizacao 3D

Para 3D, use `Camera3D` e componentes que herdam de `Renderable3D`, como `Sphere` e `StaticModel`.

```python
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.Sphere import Sphere
from EasyCells3D.Geometry import Vec3

camera = game.CreateItem()
camera.AddComponent(Camera3D())
camera.AddComponent(FreeCam())
camera.transform.position = Vec3(0, 2, 8)

sphere = game.CreateItem()
sphere.AddComponent(Sphere(1, texture_path="copper_bulb_lit_powered.png"))
```

Para modelos:

```python
from EasyCells3D.Components.StaticModel import StaticModel

model = game.CreateItem()
model.AddComponent(StaticModel("model/Floor.obj"))
```

Caminhos tambem partem de `Assets/`.

## Scheduler e Tick

O `Scheduler` executa coroutines async em fatias por frame. Ele fica em `game.scheduler`.

Executar uma coroutine depois de um tempo:

```python
async def spawn_enemy():
    await game.scheduler.sleep(2.0)
    create_enemy()

game.scheduler.create_task(spawn_enemy())
```

Coroutine repetitiva:

```python
async def blink():
    while True:
        sprite.enable = not sprite.enable
        await game.scheduler.sleep(0.5)

game.scheduler.create_task(blink())
```

Criar uma task para iniciar depois e aguardar seu retorno:

```python
async def load_data():
    await game.scheduler.sleep(0.5)
    return 42

task = game.scheduler.prepare_task(load_data())
task.start()
value = await task
```

`Tick` e um cooldown simples:

```python
from EasyCells3D import Tick

class Shooter(Component):
    def __init__(self):
        self.cooldown = Tick(0.25)

    def loop(self):
        if self.cooldown():
            self.shoot()
```

## Troca de level

Use `game.new_game()` para carregar outro level ou reiniciar o atual.

```python
if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
    game.new_game("platform")
```

Ao trocar de level:

- itens com `destroy_on_load=True` sao destruidos;
- o scheduler e limpo;
- `init(game)` do novo level e chamado;
- a execucao do frame atual e interrompida internamente por `NewGame`.

Para manter um item entre levels:

```python
manager = game.CreateItem()
manager.destroy_on_load = False
```

## Rede

A camada de rede fica em `EasyCells3D.NetworkComponents`. Ela oferece:

- `NetworkManager`: cria servidor ou cliente TCP/UDP.
- `NetworkComponent`: base para componentes com identidade de rede.
- `@Rpc`: decorador para chamadas remotas.
- `NetworkVariable`: variavel sincronizada por rede.
- `NetworkTransform`: sincronizacao de transform usando UDP.

Exemplo de inicializacao:

```python
from EasyCells3D.NetworkComponents import NetworkManager

network_item = game.CreateItem()
network_item.destroy_on_load = False
network_item.AddComponent(NetworkManager(
    ip="localhost",
    port=25765,
    is_server=True,
))
```

Exemplo de componente com RPC:

```python
from EasyCells3D.NetworkComponents import NetworkComponent, Rpc, SendTo, Protocol

class DoorNetwork(NetworkComponent):
    @Rpc(send_to=SendTo.ALL, require_owner=True, protocol=Protocol.TCP)
    def open_door(self):
        self.transform.y -= 64
```

Ao chamar `open_door()`, a chamada e empacotada e enviada pela rede. Quando chega remotamente, o metodo e executado no componente de mesmo `identifier`.

## Padrao recomendado para desenvolver

1. Crie um level em `Levels/`.
2. Coloque scripts reutilizaveis em `UserComponents/`.
3. Para cada entidade importante, crie uma funcao factory, como `load_player(game, folder_name)`.
4. Dentro da factory, crie o `Item`, adicione sprite/collider/rigidbody/scripts e retorne o item.
5. Use `init()` do level para montar a cena.
6. Use `loop()` do level apenas para coordenacao global.
7. Coloque comportamento de entidade dentro de `Component.loop()`.

Exemplo de factory:

```python
import pyray as rl

from EasyCells3D import Game
from EasyCells3D.Components import Sprite
from EasyCells3D.PhysicsComponents import RectCollider, Rigidbody
from UserComponents.Enemy import Enemy

def load_enemy(game: Game, x: float, y: float):
    enemy = game.CreateItem()
    enemy.name = "Enemy"
    enemy.transform.x = x
    enemy.transform.y = y
    enemy.AddComponent(Sprite("enemy.png", (32, 32)))
    enemy.AddComponent(RectCollider(rl.Rectangle(0, 0, 32, 32), mask=2))
    enemy.AddComponent(Rigidbody(use_gravity=True))
    enemy.AddComponent(Enemy())
    return enemy
```

## Boas praticas especificas desta lib

- Crie camera antes dos renderizaveis, para `Sprite`, `TileMapRenderer`, `Sphere` e modelos entrarem na camera principal correta.
- Sempre adicione collider antes ou junto do `Rigidbody`.
- Defina `game.physics_world = SATPhysicsWorld()` no `init()` do level que usa fisica 2D.
- Use `game.delta_time` para movimento por frame.
- Use `Transform.positionVec2` quando estiver trabalhando em 2D.
- Use `Vec2`, `Vec3` e `Quaternion` da lib para evitar misturar tuplas e tipos do Raylib.
- Para assets, passe caminhos relativos a `Assets/`.
- Para reiniciar um level, use `game.new_game("nome_do_level")`.
- Para debug de colisao 2D, use `debug=True` nos colliders e tenha uma `Camera2D` ativa.

## Dependencias

O projeto usa principalmente:

```text
raylib
numpy
numba
scipy
pygame
```

Instalacao basica:

```bash
pip install raylib numpy numba scipy pygame
```

## Observacoes importantes

- A API segue um estilo parecido com Unity, mas continua sendo Python: scripts sao classes normais e levels sao modulos.
- Alguns arquivos antigos de UI parecem usar imports legados. Para desenvolvimento atual, os exemplos mais confiaveis estao em `Levels/platform.py`, `Levels/solar.py`, `Levels/test_rigidbody.py` e `UserComponents/platform/Player.py`.
- O nome `ratating_obj.py` esta escrito assim no projeto; se for renomear, atualize os imports.
- A fisica atual e 2D. A parte 3D cobre renderizacao, camera, modelos e transformacoes.
