# EasyCells3D

Uma game engine em **Python**, leve e baseada em componentes (estilo Unity), com suporte a **2D e 3D**, física, animação, rede, agendador assíncrono e carregamento de cenas serializadas. O render é feito com [Raylib](https://www.raylib.com/) (via `pyray`).

> Itens (`Item`) vivem no mundo, recebem **componentes** (`Component`) que descrevem comportamento, e são organizados em uma **hierarquia** de pais e filhos. As cenas são módulos Python com `init()` e `loop()`.

---

## Índice

- [Recursos](#recursos)
- [Instalação](#instalação)
- [Início rápido](#início-rápido)
- [Conceitos fundamentais](#conceitos-fundamentais)
  - [Game](#game)
  - [Cenas (Levels)](#cenas-levels)
  - [Item](#item)
  - [Component](#component)
  - [Transform](#transform)
- [Renderização 2D](#renderização-2d)
- [Renderização 3D](#renderização-3d)
- [Física e colisões](#física-e-colisões)
- [Campos serializáveis](#campos-serializáveis)
- [Carregando cenas (.ecscene)](#carregando-cenas-ecscene)
- [Agendador assíncrono (Scheduler)](#agendador-assíncrono-scheduler)
- [Rede](#rede)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Comparação com Unity](#comparação-com-unity)

---

## Recursos

- **Arquitetura baseada em componentes** — `Item` + `Component` + `Transform`, com hierarquia pai/filho.
- **2D e 3D** — câmeras, sprites, spritesheets, modelos `.gltf`, luzes.
- **Física** — `Rigidbody`, colisores poligonais (SAT), raycast e box cast, acelerados com Numba (JIT).
- **Animação** — animador 2D por spritesheet e animador 3D por clipes.
- **Agendador** — corrotinas `async`/`await` integradas ao loop do jogo (`sleep`, `next_frame`).
- **Serialização de cenas** — carregue cenas a partir de arquivos JSON `.ecscene`.
- **Rede** — servidores/clientes TCP e UDP.
- **Tilemaps** — renderização e colisão a partir de CSV/TSJ (Tiled).

---

## Instalação

A engine depende de Raylib, NumPy, Numba e SciPy:

```bash
pip install raylib numpy numba scipy
```

Clone o repositório e use a pasta `EasyCells3D/` como pacote dentro do seu projeto:

```bash
git clone https://github.com/poweron0102/EasyCells3D.git
cd EasyCells3D
```

> Requer **Python 3.12+** (o código usa a sintaxe de generics em métodos, ex.: `def AddComponent[T](...)`).

---

## Início rápido

O ponto de entrada cria um `Game`, aponta para uma cena inicial e chama `run()`.

```python
# main.py
from EasyCells3D import Game
import Levels.platform

if __name__ == '__main__':
    GAME = Game(
        Levels.platform,          # módulo da cena inicial (ou nome em string)
        "Meu Jogo",               # título da janela
        show_fps=True,            # mostra FPS no título
        screen_resolution=(1280, 720),
        dynamic_resolution=True,  # janela redimensionável
    )
    GAME.run()
```

Assinatura completa do construtor:

```python
Game(
    start_level: str | ModuleType,
    game_name: str,
    show_fps: bool = False,
    screen_resolution: tuple[int, int] = (800, 600),
    dynamic_resolution: bool = False,
    target_fps: int = -1,             # -1 = sem limite
    render_target: rl.RenderTexture = None,  # renderiza em textura (ex.: editor)
)
```

---

## Conceitos fundamentais

### Game

`Game` é o runtime: cria a janela, mantém a lista de itens e câmeras, controla o tempo e roda o loop principal.

| Atributo / método | Descrição |
|---|---|
| `Game.instance` | Singleton da instância ativa. |
| `game.delta_time` | Segundos desde o último frame. |
| `game.run_time` | Segundos decorridos na cena atual. |
| `game.CreateItem()` | Cria um novo `Item` na raiz da cena. |
| `game.new_game(level)` | Troca de cena (recarrega itens e scheduler). |
| `game.scheduler` | Instância do [Scheduler](#agendador-assíncrono-scheduler). |
| `game.cameras` | Lista de câmeras que serão renderizadas. |

O loop principal, a cada frame: roda os `init()` pendentes → atualiza todos os itens (`loop()` de cada componente) → roda `level.loop()` → atualiza o scheduler → renderiza todas as câmeras.

### Cenas (Levels)

Uma cena é simplesmente um **módulo Python** com duas funções:

```python
# Levels/minha_cena.py
from EasyCells3D import Game
from EasyCells3D.Components import Camera2D

def init(game: Game):
    """Chamada uma vez ao carregar a cena. Monte seus itens aqui."""
    camera = game.CreateItem()
    camera.AddComponent(Camera2D())

def loop(game: Game):
    """Chamada todo frame, depois dos componentes. Lógica global da cena."""
    pass
```

Trocar de cena em runtime (ex.: tecla `R` reinicia):

```python
import pyray as rl

def loop(game: Game):
    if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
        game.new_game("minha_cena")  # nome do módulo dentro de Levels/
```

### Item

`Item` é a entidade do mundo. Contém um `Transform`, um conjunto de componentes e filhos.

```python
# Criar item na raiz
inimigo = game.CreateItem()
inimigo.name = "Inimigo"
inimigo.transform.x = 100
inimigo.transform.y = 50

# Hierarquia: criar um filho
arma = inimigo.CreateChild()
arma.transform.x = 10          # posição local, relativa ao pai

# Adotar um item existente como filho
solto = game.CreateItem()
inimigo.AddChild(solto)

# Buscar componente (procura também nos filhos)
rb = inimigo.GetComponent(Rigidbody)

# Destruir item (e seus filhos)
inimigo.Destroy()
```

| Método | Descrição |
|---|---|
| `CreateChild()` | Cria um `Item` filho. |
| `AddChild(item)` | Move um item existente para ser filho deste. |
| `AddComponent(comp)` | Anexa um componente e retorna a instância. |
| `GetComponent(Tipo)` | Retorna o componente do tipo (busca nos filhos também). |
| `SetParent(parent)` | Reparenta o item. |
| `Destroy()` | Remove o item, filhos e dispara `on_destroy()`. |
| `global_transform` | Transform global em cache (atualizado 1x por frame). |

### Component

Toda lógica vive em componentes. Herde de `Component` e implemente o ciclo de vida.

```python
import math
from EasyCells3D.Components import Component
from EasyCells3D.Geometry import Quaternion, Vec3
from EasyCells3D.Serialization import SerializeField

class RotatingObj(Component):
    speed = SerializeField(default=1.0)   # campo editável/serializável

    def init(self):
        # Chamado uma vez, logo após ser adicionado ao item.
        print(f"Girando a {self.speed} graus/s")
        self.speed = math.radians(float(self.speed))

    def loop(self):
        # Chamado todo frame.
        angle = self.speed * self.game.delta_time
        delta = Quaternion.from_axis_angle(Vec3(0, 1, 0), angle)
        self.transform.rotation = (delta * self.transform.rotation).normalize()

    def on_destroy(self):
        # Limpeza opcional ao remover.
        pass
```

Dentro de um componente você tem acesso a:

- `self.item` — o `Item` dono.
- `self.game` — a instância de `Game`.
- `self.transform` — transform local do item (atalho para `self.item.transform`).
- `self.global_transform` — transform global em cache.
- `self.enable` — `False` pausa o `loop()` desse componente.

> **Ordem de execução:** `init()` roda no início do frame seguinte ao `AddComponent` (fica numa fila `to_init`). `loop()` roda todo frame enquanto `enable` for `True`.

### Transform

`Transform` guarda **posição** (`Vec3`), **rotação** (`Quaternion`) e **escala** (`Vec3`), com vários atalhos:

```python
t = item.transform

# Posição
t.position = Vec3(10, 5, 0)
t.x, t.y, t.z = 10, 5, 0
t.positionVec2 = Vec2(10, 5)   # útil em 2D (mantém o z)

# Rotação 2D conveniente (graus, eixo Z)
t.angle += 90

# Escala
t.scale.x = 2
t.scale.y = 2

# Vetores direcionais (3D) — também têm setters
forward = t.forward   # -Z
up      = t.up        # +Y
right   = t.right     # +X
t.forward = Vec3(0, 0, 1)  # reorienta para apontar nessa direção
```

Posições/rotações são **locais** ao pai. A transform global é calculada automaticamente a cada frame (acesse via `item.global_transform`).

---

## Renderização 2D

Use uma `Camera2D` e componentes `Sprite`.

```python
from EasyCells3D.Components import Camera2D, Sprite, Animation2D, Animator2D

def init(game):
    # Câmera 2D
    cam = game.CreateItem()
    cam.AddComponent(Camera2D(zoom=1.0))

    # Sprite simples
    player = game.CreateItem()
    player.AddComponent(Sprite("player.png", (32, 32)))  # caminho, tamanho do frame
    player.transform.z = 1   # z controla a ordem de desenho (camadas)

    # Animação por spritesheet
    player.AddComponent(Animator2D(
        {
            "idle": Animation2D(speed=0.1, frames=[0, 1, 2]),
            "run":  Animation2D(speed=0.05, frames=list(range(10))),
        },
        current_animation="idle",
    ))
```

Conversão de tela para mundo (ex.: posição do mouse):

```python
world_pos = cam.get_mouse_world_position()
```

### Tilemaps

A engine carrega mapas do [Tiled](https://www.mapeditor.org/) (CSV + TSJ):

```python
from EasyCells3D.Components import TileMap, TileMapRenderer
from EasyCells3D.Components.TileMap import matrix_from_csv, solids_set_from_tsj
from EasyCells3D.PhysicsComponents import TileMapCollider, Rigidbody

tile_map = game.CreateItem()
tile_map.AddComponent(TileMap(matrix_from_csv("mapa.csv")))
tile_map.AddComponent(TileMapRenderer("Terrain (16x16).png", 16))
tile_map.AddComponent(TileMapCollider(solids_set_from_tsj("mapa.tsj"), 16, debug=True))
tile_map.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))
tile_map.transform.scale.x = 2
tile_map.transform.scale.y = 2
```

---

## Renderização 3D

Use uma `Camera3D`. Componentes `StaticModel`, `Sphere`, `Light3D` e `AnimatedModel` desenham no mundo 3D.

```python
from EasyCells3D.Components import Camera3D
from EasyCells3D.Components.FreeCam import FreeCam
from EasyCells3D.Components.Sphere import Sphere
from EasyCells3D.Components.StaticModel import StaticModel
from EasyCells3D.Geometry import Vec3

def init(game):
    # Câmera 3D com controle livre (WASD + mouse)
    cam = game.CreateItem()
    cam.AddComponent(Camera3D(vfov=60.0))
    cam.AddComponent(FreeCam())

    # Esfera com textura
    esfera = game.CreateItem()
    esfera.AddComponent(Sphere(1, texture_path="bulb.png"))

    # Esfera filha (orbita o pai pela hierarquia)
    filha = esfera.CreateChild()
    filha.AddComponent(Sphere(1, texture_path="bulb.png"))
    filha.transform.position += Vec3(4, 0, 0)

    # Modelo estático (.gltf)
    chao = game.CreateItem()
    chao.AddComponent(StaticModel("model/Floor_Dirt.gltf"))
```

---

## Física e colisões

Os colisores são poligonais e usam o **Teorema dos Eixos Separadores (SAT)**. O `Rigidbody` integra forças e gravidade.

```python
import pyray as rl
from EasyCells3D.Geometry import Vec2
from EasyCells3D.PhysicsComponents import Rigidbody, RectCollider

def init(game):
    # Corpo dinâmico
    caixa = game.CreateItem()
    caixa.AddComponent(RectCollider(rl.Rectangle(0, 0, 32, 32), debug=True, mask=1))
    caixa.AddComponent(Rigidbody(mass=1.0, use_gravity=True, drag=0.1, restitution=0.5))

    # Corpo estático (cinemático, sem gravidade) — chão/paredes
    chao = game.CreateItem()
    chao.AddComponent(RectCollider(rl.Rectangle(0, 0, 400, 20), mask=1))
    chao.AddComponent(Rigidbody(is_kinematic=True, use_gravity=False))

    # IMPORTANTE: inicie a simulação depois de criar os corpos
    Rigidbody.start_physics()
```

Aplicando forças (dentro de um componente ou no `loop` da cena):

```python
rb = caixa.GetComponent(Rigidbody)
rb.add_force(Vec2(100, 0))       # força contínua
rb.add_impulse(Vec2(0, -200))    # impulso instantâneo (ex.: pulo)
```

`Rigidbody(...)` aceita: `mass`, `use_gravity`, `is_kinematic`, `drag`, `angular_drag`, `gravity_scale`, `restitution`. A gravidade global é `Rigidbody.Gravity` (padrão `Vec2(0, 980)`).

### Raycast e box cast

```python
from EasyCells3D.PhysicsComponents import Collider
from EasyCells3D.Geometry import Vec2

# Raio
hit = Collider.ray_cast_static(
    origin=Vec2(100, 100),
    direction=Vec2(1, 0).normalize(),
    max_distance=50,
    mask=1,
)
if hit:
    collider, ponto, normal, distancia = hit

# Caixa (rect cast)
hit = Collider.rect_cast_static(
    origin=Vec2(100, 100),
    size=Vec2(20, 20),
    angle=0,
    direction=Vec2(0, 1),
    max_distance=50,
    mask=1,
)
```

> `mask` é um bitmask de camadas de colisão — colisores só interagem com camadas compatíveis.

---

## Campos serializáveis

`SerializeField` declara atributos que aparecem no editor e são salvos/carregados em cenas `.ecscene`. Funciona como um descriptor: leia/escreva como um atributo normal.

```python
from EasyCells3D.Serialization import SerializeField

class Inimigo(Component):
    vida   = SerializeField(default=100)
    nome   = SerializeField(default="Goblin")
    cor    = SerializeField(default=[1, 0, 0])          # vetor (RGB)
    tipo   = SerializeField(default="a", choices=["a", "b", "c"])
    alvo   = SerializeField(default=None, ref="Item")   # referência a outro item

    def loop(self):
        self.vida -= 1 * self.game.delta_time
```

O tipo é inferido a partir do `default` (`int`, `float`, `bool`, `str`, `vector`) ou pode ser forçado com `field_type=`. Use `ref=` para referências e `choices=` para enums.

---

## Carregando cenas (.ecscene)

Cenas podem ser descritas em JSON (`.ecscene`) com itens, hierarquia, componentes e assets, e carregadas em runtime:

```python
from EasyCells3D import SceneLoader

def init(game):
    loader = SceneLoader(game)
    itens = loader.load("Levels/minha_cena.ecscene")
```

O formato é documentado em [docs/formato_ecscene.md](docs/formato_ecscene.md). Componentes são descobertos e instanciados automaticamente pelo `ComponentRegistry`.

---

## Agendador assíncrono (Scheduler)

O `Scheduler` integra corrotinas `async`/`await` ao loop do jogo — ideal para sequências temporizadas sem máquinas de estado.

```python
async def abrir_porta(game):
    await game.scheduler.sleep(1.0)     # espera 1 segundo
    porta.transform.y -= 50
    await game.scheduler.next_frame()   # espera 1 frame
    porta.transform.y -= 50

# Iniciar a tarefa (delay opcional antes de começar)
task = game.scheduler.create_task(abrir_porta(game), delay=0.5)

# Cancelar
game.scheduler.cancel(task)
```

| Método | Descrição |
|---|---|
| `create_task(coro, delay=0, key=None)` | Agenda e inicia uma corrotina. |
| `sleep(segundos)` | `await` pausa a corrotina pelo tempo dado. |
| `next_frame()` | `await` retoma no próximo frame. |
| `cancel(task ou key)` | Cancela uma tarefa. |
| `clear()` | Cancela tudo (feito automaticamente ao trocar de cena). |

---

## Rede

Suporte a servidores e clientes **TCP** e **UDP** (serialização via `pickle`).

```python
from EasyCells3D.NetworkTCP import NetworkServer  # ou NetworkUDP

server = NetworkServer(ip="0.0.0.0", port=5000, ip_version=4)
server.broadcast({"tipo": "spawn", "x": 100, "y": 50})
dados = server.read(client_id=0)
```

Documentação detalhada em [docs/network.html](docs/network.html).

---

## Estrutura do projeto

```
EasyCells3D/
├── main.py                 # ponto de entrada do jogo
├── editor.py               # editor (renderiza em RenderTexture)
├── EasyCells3D/            # pacote da engine
│   ├── Game.py             # runtime e loop principal
│   ├── Geometry.py         # Vec2, Vec3, Quaternion
│   ├── scheduler.py        # corrotinas async
│   ├── SceneLoader.py      # carregamento de .ecscene
│   ├── Serialization.py    # SerializeField
│   ├── ComponentRegistry.py
│   ├── Components/         # Component, Item, Transform, câmeras, sprites, modelos…
│   ├── PhysicsComponents/  # Rigidbody, Collider, TileMapCollider…
│   ├── NetworkComponents/  # componentes de rede
│   ├── UiComponents/       # UI
│   └── AssetTypes/         # gerenciamento de assets
├── Levels/                 # cenas (módulos com init/loop)
├── UserComponents/         # seus componentes customizados
├── Assets/                 # imagens, modelos, áudio…
└── docs/                   # documentação adicional
```

Veja também o [GUIA_DESENVOLVIMENTO.md](GUIA_DESENVOLVIMENTO.md) para um guia de desenvolvimento mais aprofundado.

---

## Comparação com Unity

Se você vem do Unity, o mapeamento mental é direto:

| Unity | EasyCells3D |
|---|---|
| `GameObject` | `Item` |
| `MonoBehaviour` | `Component` |
| `Transform` | `Transform` |
| `Start()` | `init()` |
| `Update()` | `loop()` |
| `OnDestroy()` | `on_destroy()` |
| `[SerializeField]` | `SerializeField(...)` |
| `Instantiate` | `game.CreateItem()` / `item.CreateChild()` |
| `GetComponent<T>()` | `GetComponent(T)` |
| Coroutines (`IEnumerator`) | corrotinas `async` do `Scheduler` |
| Cena (.unity) | módulo `Levels/*.py` ou `.ecscene` |

---

## Licença

Defina a licença do projeto aqui (ex.: MIT).
