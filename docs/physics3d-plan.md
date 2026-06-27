# Plano de Implementação — Física 3D (PyBullet)

Engine: EasyCells3D · Backend: PyBullet (`DIRECT`) · Destino: `EasyCells3D/PhysicsComponents3D/`

> Resumo do design fechado: o Bullet é a fonte da verdade para corpos DYNAMIC e o
> personagem; a engine empurra a pose para STATIC (spawn) e KINEMATIC (todo frame).
> `game.physics_world` é um slot único (base abstrata `PhysicsWorld` + impl
> `BulletPhysicsWorld`), criado no `level.init()` e destruído na troca de cena.
> Componente combinado `PhysicsBody3D`. Timestep modelo A (delta variável + substep
> interno do Bullet), tickado por hook fixo no `Game.run`.

---

## Princípios de execução

- **Incremental e testável**: cada milestone termina com uma cena rodável em `Levels/`
  que prova o comportamento (seguindo o precedente de `test_rigidbody`).
- **API pública primeiro**: `CharacterController3D` e helpers são construídos só sobre
  a API pública de `PhysicsBody3D`/`PhysicsWorld` — sem atalhos internos.
- **Projetar serializável desde já**: shapes são dataclasses com tag de tipo e os
  campos são `SerializeField`, mesmo sem integrar o `SceneLoader` na v1.
- **Cliente isolado**: todo acesso ao pybullet passa `physicsClientId=self._client`.

---

## Milestone 0 — Setup & scaffolding

**Objetivo:** dependência instalada e pacote criado.

- [ ] Adicionar `pybullet` ao gerenciamento de deps e ao `readme.md` (junto de
      raylib/numpy/numba/scipy).
- [ ] Criar `EasyCells3D/PhysicsComponents3D/__init__.py`.
- [ ] Smoke test isolado (script avulso): `connect(DIRECT)` → `stepSimulation` em mundo
      vazio → `disconnect`, confirmando que a lib roda no ambiente.

**Done quando:** `import pybullet` funciona e o smoke test passa.

---

## Milestone 1 — Container do mundo + hook no loop

**Objetivo:** `game.physics_world` existe, ticka e é demolido na troca de cena.

- [ ] `PhysicsWorld` (base abstrata) definindo a interface:
      `step(dt)`, `add_body(body)`, `remove_body(body)`, `raycast(...)`,
      `overlap_sphere(...)`, `set_gravity(v)`, `destroy()`, prop `debug_draw`.
- [ ] `BulletPhysicsWorld(PhysicsWorld)`:
  - `__init__`: `self._client = pybullet.connect(pybullet.DIRECT)`, gravidade
    `(0, -9.81, 0)`, guarda lista de corpos registrados.
  - `step(dt)`: empurra KINEMATIC (M3) → `stepSimulation(dt, maxSubSteps=10,
    fixedTimeStep=1/120, physicsClientId=self._client)` → sync DYNAMIC (M2).
  - `destroy()`: `pybullet.disconnect(self._client)`.
- [ ] **Editar `EasyCells3D/Game.py`:**
  - `__init__`: `self.physics_world: PhysicsWorld | None = None`.
  - `run()`: após `for item in ...: item.update()` e **antes** do render —
    `if self.physics_world: self.physics_world.step(self.delta_time)`.
  - `new_game()`: **antes** de `self.level.init(self)`, demolir o world antigo
    (`if self.physics_world: self.physics_world.destroy(); self.physics_world = None`).
- [ ] Level de teste: `init` faz `game.physics_world = BulletPhysicsWorld()`.

**Done quando:** rodar com world ativo não quebra nem derruba FPS; trocar de cena
não vaza cliente pybullet (verificar via contador de corpos/`getNumBodies`).

---

## Milestone 2 — Shapes primitivas + `PhysicsBody3D` (STATIC/DYNAMIC)

**Objetivo:** caixa dinâmica cai e assenta sobre chão estático.

- [ ] `shapes.py`: dataclasses com tag de tipo — `BoxShape(half_extents)`,
      `SphereShape(radius)`, `CapsuleShape(radius, height)`, `CylinderShape(...)`.
      Cada uma sabe construir o `collisionShape` no Bullet (`createCollisionShape`).
- [ ] `PhysicsBody3D(Component)`:
  - `__init__(shape, mass=0, body_type=DYNAMIC, is_trigger=False, ...)`.
  - `init()`: assert `self.game.physics_world is not None` (erro claro se faltar);
    constrói shape, aplica `setLocalScaling(global_transform.scale)`, cria o
    `btRigidBody` na pose `global_transform` (mundo), registra no world.
  - `loop()`: se DYNAMIC, lê pose do Bullet e escreve em `global_transform`
    (conversão **mundo→local** se o item tiver pai).
  - `on_destroy()`: remove do world **se o world registrado ainda for o vivo**
    (guarda por referência ao world).
  - `enable` como *property*: `False` remove o corpo da simulação; `True` reinsere
    na pose atual.
- [ ] Helpers de conversão `Quaternion`↔Bullet (quat xyzw) e `Vec3`↔lista.

**Done quando:** cena com chão `STATIC` (box) + caixa `DYNAMIC` caindo; o
`StaticModel` da caixa segue a física e ela para no chão.

---

## Milestone 3 — KINEMATIC + API de controle + troca de `body_type`

**Objetivo:** plataforma móvel carrega caixa; teleporte e troca de modo funcionam.

- [ ] No `step()`: antes de simular, para cada KINEMATIC ler `global_transform`,
      `resetBasePositionAndOrientation` + derivar/aplicar velocidade a partir do
      delta de pose (pra carregar/empurrar dinâmicos).
- [ ] API do `PhysicsBody3D`: `velocity` (get/set), `apply_force(Vec3)`,
      `apply_impulse(Vec3)`, `teleport(pos, rot=None)` (zera velocidade).
- [ ] `body_type` como *property* setável: reconfigura o corpo existente
      (massa/flags) sem recriar quando possível (DYNAMIC↔KINEMATIC↔STATIC).

**Done quando:** plataforma KINEMATIC (movida só pelo `Transform`) transporta uma
caixa dinâmica; `teleport()` reposiciona limpo; alternar um corpo DYNAMIC↔KINEMATIC
em runtime funciona (base pra cutscene).

---

## Milestone 4 — Triangle mesh + auto-build do `StaticModel`

**Objetivo:** mapa de `.gltf` colide sem trabalho manual.

- [ ] `TriangleMeshShape(vertices, indices)`, `ConvexHullShape(vertices)`,
      `CompoundShape(children)`.
- [ ] Extrair `vertices`/`indices` do `rl.Mesh` (via `StaticModel.model.meshes`).
- [ ] Auto-build: se `shape=None` **e** `STATIC` **e** o item tem `StaticModel`,
      construir `btBvhTriangleMeshShape` automaticamente.
- [ ] DYNAMIC com `shape=None` → erro explícito ("triangle mesh dinâmica é
      proibida; passe uma primitiva/convex hull").
- [ ] Helper pra carregar malha de colisão separada de outro gltf (`shape=` explícito).

**Done quando:** carregar um mapa gltf e ver objetos dinâmicos colidindo com a
geometria real do mapa.

---

## Milestone 5 — Eventos e queries

**Objetivo:** callbacks de colisão/trigger e raycast funcionando.

- [ ] Diff de contatos por frame (`getContactPoints`) → dispara
      `on_collision_enter(other)` / `on_collision_exit(other)`.
- [ ] Triggers: corpos `is_trigger=True` com flag de no-contact-response →
      `on_trigger_enter(other)` / `on_trigger_exit(other)`.
- [ ] `physics_world.raycast(origin, dir, max_dist, mask) -> Hit | None`
      (Hit: body, point, normal, distance).
- [ ] `physics_world.overlap_sphere(center, radius, mask) -> list[body]`.
- [ ] `collision_group` / `collision_mask` (filtros nativos do Bullet) no corpo.

**Done quando:** uma zona trigger dispara enter/exit ao atravessar; raycast pra
baixo acerta o chão; callback de colisão imprime no impacto.

---

## Milestone 6 — `CharacterController3D` (helper, agnóstico de câmera)

**Objetivo:** personagem controlável anda, pula e respeita rampas.

- [ ] Cápsula DYNAMIC, `angularFactor=0`, `allow_sleep=False`.
- [ ] `move(direction: Vec3)` (world-space; quem chama decide a câmera/input) seta
      a velocidade horizontal preservando a vertical.
- [ ] `jump()`: só se `is_grounded` (raycast curto pra baixo, `ground_check_distance`).
- [ ] Knobs: `move_speed`, `jump_height`, `ground_check_distance`, `max_slope`.
- [ ] Construído **só** sobre a API pública (serve de exemplo de referência).

**Done quando:** uma cápsula controlada pelo teclado anda sobre o mapa, pula, cai e
não sobe rampas além do `max_slope`.

---

## Milestone 7 — Debug draw

**Objetivo:** enxergar as shapes pra depurar colliders.

- [ ] `physics_world.debug_draw = True` → no render, wireframes via primitivas Raylib
      (box/esfera/cápsula) e linhas pra triangle mesh.
- [ ] Opt-in, custo zero quando desligado.

**Done quando:** o overlay de wireframe bate com os modelos visuais; cápsula do
personagem visível.

---

## Milestone 8 — Materiais & polimento

- [ ] Ligar `friction=0.5`, `restitution=0.0`, `linear_damping=0.0`,
      `angular_damping=0.05`, `gravity_scale ∈ {0.0, 1.0}`, `allow_sleep=True`.
- [ ] Revisão de cleanup, docstrings e um exemplo completo em `Levels/`.

---

## Edições fora de `PhysicsComponents3D/`

- `EasyCells3D/Game.py`: campo `physics_world`, hook de `step()` no `run()`,
  demolição do world no `new_game()`.
- `readme.md` / deps: adicionar `pybullet`.

> Nenhuma mudança necessária no `Component`/`Transform`/`SceneLoader` na v1
> (serialização fica adiada).

---

## Adiado pra v2 (consciente)

Interpolação anti-jitter · `apply_torque`/`angular_velocity` (set) · `gravity_scale`
arbitrário · controlador kinematic puro do personagem · serialização completa no
`.ecscene` (+ editor) · CCD (anti-tunneling) · `shape_cast`/sweep · debug de
contatos/raios.

---

## Ponto de tuning em aberto

Game-feel do `CharacterController3D` (sensação "escorregadia" da cápsula dinâmica)
depende do gênero/câmera do jogo concreto — ajustável nos knobs sem mexer na
arquitetura.
