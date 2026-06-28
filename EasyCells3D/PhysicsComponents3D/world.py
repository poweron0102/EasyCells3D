"""Mundo de física 3D.

``PhysicsWorld`` é a base abstrata que define a interface usada pela engine e
pelos componentes; ``BulletPhysicsWorld`` é a implementação concreta sobre o
PyBullet rodando em modo ``DIRECT`` (sem janela própria — quem desenha é a
Raylib).

Fluxo de uma simulação (modelo A — delta variável + substep fixo interno):

1. ``step(dt)`` empurra a pose dos corpos KINEMATIC para o Bullet e deriva a
   velocidade deles a partir do delta de pose (pra carregar/empurrar dinâmicos);
2. avança a simulação em passos fixos de ``fixed_timestep`` acumulando ``dt``;
3. puxa a pose dos corpos DYNAMIC de volta para o ``Transform`` da engine.

O Bullet é a fonte da verdade para DYNAMIC; a engine é a fonte da verdade para
STATIC (definido no spawn) e KINEMATIC (empurrado todo frame).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from ..Geometry import Vec3

if TYPE_CHECKING:
    from .body import PhysicsBody3D


class BodyType(IntEnum):
    """Como um corpo é tratado pela simulação.

    - ``STATIC``: imóvel, massa 0. A engine define a pose no spawn.
    - ``DYNAMIC``: simulado pelo Bullet (gravidade, forças, colisões).
    - ``KINEMATIC``: movido só pelo ``Transform``; empurra dinâmicos mas não é
      empurrado por eles.
    """

    STATIC = 0
    DYNAMIC = 1
    KINEMATIC = 2


@dataclass
class RaycastHit:
    """Resultado de um :meth:`PhysicsWorld.raycast`."""

    body: Any | None
    point: Vec3
    normal: Vec3
    distance: float


class PhysicsWorld(ABC):
    """Interface do mundo de física. Slot único em ``game.physics_world``."""

    @abstractmethod
    def step(self, dt: float) -> None:
        ...

    @abstractmethod
    def add_body(self, body: Any) -> None:
        ...

    @abstractmethod
    def remove_body(self, body: Any) -> None:
        ...

    @abstractmethod
    def raycast(self, origin: Vec3, direction: Vec3, max_dist: float,
                mask: int = -1) -> "RaycastHit | None":
        ...

    @abstractmethod
    def overlap_sphere(self, center: Vec3, radius: float,
                       mask: int = -1) -> list[Any]:
        ...

    @abstractmethod
    def set_gravity(self, gravity: Vec3) -> None:
        ...

    @abstractmethod
    def destroy(self) -> None:
        ...

    @property
    def debug_draw(self) -> bool:
        return self._debug_draw

    @debug_draw.setter
    def debug_draw(self, value: bool) -> None:
        self._debug_draw = bool(value)


class BulletPhysicsWorld(PhysicsWorld):
    """Mundo de física baseado no PyBullet (cliente ``DIRECT`` isolado)."""

    def __init__(
            self,
            gravity: Vec3 = Vec3(0.0, -9.81, 0.0),
            fixed_timestep: float = 1.0 / 120.0,
            max_substeps: int = 10,
    ):
        import pybullet as p
        self._p = p

        # Cliente DIRECT isolado — todo acesso ao pybullet passa este id.
        self._client = p.connect(p.DIRECT)
        self._gravity = gravity
        p.setGravity(gravity.x, gravity.y, gravity.z, physicsClientId=self._client)
        p.setTimeStep(fixed_timestep, physicsClientId=self._client)

        self.fixed_timestep = fixed_timestep
        self.max_substeps = max_substeps
        self._accumulator = 0.0
        self._debug_draw = False

        # Corpos registrados, e índice por uid do Bullet (pra mapear de volta
        # resultados de raycast/contatos -> PhysicsBody3D).
        self.bodies: list[PhysicsBody3D] = []
        self._bodies_by_uid: dict[int, PhysicsBody3D] = {}

        # Pares em contato no frame anterior, pra diffar enter/exit (M5).
        self._contacts: set[tuple[int, int]] = set()
        self._triggers: set[tuple[int, int]] = set()

    # -- propriedades / acesso ------------------------------------------------

    @property
    def client(self) -> int:
        return self._client

    @property
    def gravity(self) -> Vec3:
        return self._gravity

    def body_for_uid(self, uid: int) -> "PhysicsBody3D | None":
        return self._bodies_by_uid.get(uid)

    # -- registro de corpos ---------------------------------------------------

    def add_body(self, body: "PhysicsBody3D") -> None:
        if body in self.bodies:
            return
        self.bodies.append(body)
        if body.uid is not None:
            self._bodies_by_uid[body.uid] = body
            self._reconcile_trigger_pairs(body)

    def _reconcile_trigger_pairs(self, body: "PhysicsBody3D") -> None:
        """Desliga a *resposta* de colisão entre triggers e o resto (sensores).

        A detecção continua acontecendo via overlap query no ``step``; aqui só
        garantimos que nada é empurrado pelos triggers.
        """
        if body.is_trigger:
            for other in self.bodies:
                if other is body or other.uid is None:
                    continue
                self._disable_response(body.uid, other.uid)
        else:
            for other in self.bodies:
                if other.is_trigger and other.uid is not None and other is not body:
                    self._disable_response(other.uid, body.uid)

    def _disable_response(self, uid_a: int, uid_b: int) -> None:
        try:
            self._p.setCollisionFilterPair(uid_a, uid_b, -1, -1, 0,
                                           physicsClientId=self._client)
        except Exception:
            pass

    def remove_body(self, body: "PhysicsBody3D") -> None:
        if body in self.bodies:
            self.bodies.remove(body)
        if body.uid is not None:
            self._bodies_by_uid.pop(body.uid, None)
        # Limpa contatos pendentes envolvendo este corpo.
        self._contacts = {pair for pair in self._contacts if body.uid not in pair}
        self._triggers = {pair for pair in self._triggers if body.uid not in pair}

    def _reindex_uid(self, body: "PhysicsBody3D", old_uid: int | None) -> None:
        """Atualiza o índice quando o uid de um corpo muda (recriação)."""
        if old_uid is not None:
            self._bodies_by_uid.pop(old_uid, None)
        if body.uid is not None and body in self.bodies:
            self._bodies_by_uid[body.uid] = body

    # -- simulação ------------------------------------------------------------

    def set_gravity(self, gravity: Vec3) -> None:
        self._gravity = gravity
        self._p.setGravity(gravity.x, gravity.y, gravity.z, physicsClientId=self._client)

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return

        # 1) Empurra KINEMATIC (pose + velocidade derivada do delta de pose).
        for body in self.bodies:
            if body.body_type == BodyType.KINEMATIC:
                body._push_kinematic(dt)

        # Corpos com gravity_scale != 1 precisam de uma força de compensação
        # reaplicada a cada substep (o Bullet zera forças após cada step).
        grav_comp = [b for b in self.bodies
                     if b.body_type == BodyType.DYNAMIC
                     and b.uid is not None
                     and b.gravity_scale != 1.0]

        # 2) Avança a simulação em passos fixos (acumulador).
        self._accumulator += dt
        steps = 0
        while self._accumulator >= self.fixed_timestep and steps < self.max_substeps:
            for body in grav_comp:
                body._apply_gravity_compensation(self._gravity)
            self._p.stepSimulation(physicsClientId=self._client)
            self._accumulator -= self.fixed_timestep
            steps += 1
        # Evita acúmulo infinito se o frame estourar (spiral of death).
        if self._accumulator > self.fixed_timestep:
            self._accumulator = 0.0

        # 3) Puxa DYNAMIC de volta pro Transform da engine.
        for body in self.bodies:
            if body.body_type == BodyType.DYNAMIC:
                body._pull_dynamic()

        # 4) Eventos de colisão/trigger (M5).
        self._dispatch_collision_events()

    # -- eventos (M5) ---------------------------------------------------------

    def _dispatch_collision_events(self) -> None:
        p = self._p

        # Colisões "sólidas": pares de getContactPoints (triggers têm mask 0 e
        # não aparecem aqui).
        current_contacts: set[tuple[int, int]] = set()
        for pt in p.getContactPoints(physicsClientId=self._client):
            uid_a, uid_b = pt[1], pt[2]
            body_a = self._bodies_by_uid.get(uid_a)
            body_b = self._bodies_by_uid.get(uid_b)
            if body_a is None or body_b is None:
                continue
            current_contacts.add((uid_a, uid_b) if uid_a < uid_b else (uid_b, uid_a))

        # Triggers: detectados por overlap (AABB + getClosestPoints), já que não
        # geram resposta/contato.
        current_triggers: set[tuple[int, int]] = set()
        for body in self.bodies:
            if not body.is_trigger or body.uid is None:
                continue
            aabb_min, aabb_max = p.getAABB(body.uid, physicsClientId=self._client)
            for uid, _link in (p.getOverlappingObjects(aabb_min, aabb_max,
                                                       physicsClientId=self._client) or []):
                if uid == body.uid:
                    continue
                other = self._bodies_by_uid.get(uid)
                if other is None:
                    continue
                pts = p.getClosestPoints(body.uid, uid, 0.0, physicsClientId=self._client)
                if pts:
                    a, b = body.uid, uid
                    current_triggers.add((a, b) if a < b else (b, a))

        self._emit_pair_events(self._contacts, current_contacts,
                               "on_collision_enter", "on_collision_exit")
        self._emit_pair_events(self._triggers, current_triggers,
                               "on_trigger_enter", "on_trigger_exit")

        self._contacts = current_contacts
        self._triggers = current_triggers

    def _emit_pair_events(self, prev: set, current: set,
                          enter_name: str, exit_name: str) -> None:
        for pair in current - prev:
            self._fire_pair(pair, enter_name)
        for pair in prev - current:
            self._fire_pair(pair, exit_name)

    def _fire_pair(self, pair: tuple[int, int], event_name: str) -> None:
        a = self._bodies_by_uid.get(pair[0])
        b = self._bodies_by_uid.get(pair[1])
        if a is not None and b is not None:
            a._fire_event(event_name, b)
            b._fire_event(event_name, a)

    # -- queries (M5) ---------------------------------------------------------

    def raycast(self, origin: Vec3, direction: Vec3, max_dist: float,
                mask: int = -1) -> "RaycastHit | None":
        p = self._p
        dir_n = direction.normalize()
        to = origin + dir_n * max_dist
        kwargs = dict(
            rayFromPosition=[origin.x, origin.y, origin.z],
            rayToPosition=[to.x, to.y, to.z],
            physicsClientId=self._client,
        )
        if mask != -1:
            kwargs["collisionFilterMask"] = mask
        try:
            results = p.rayTest(**kwargs)
        except TypeError:
            # pybullet antigo sem collisionFilterMask em rayTest.
            kwargs.pop("collisionFilterMask", None)
            results = p.rayTest(**kwargs)

        if not results:
            return None
        uid, link, fraction, hit_pos, hit_normal = results[0]
        if uid < 0:
            return None
        point = Vec3(hit_pos[0], hit_pos[1], hit_pos[2])
        normal = Vec3(hit_normal[0], hit_normal[1], hit_normal[2])
        return RaycastHit(
            body=self._bodies_by_uid.get(uid),
            point=point,
            normal=normal,
            distance=fraction * max_dist,
        )

    def overlap_sphere(self, center: Vec3, radius: float,
                       mask: int = -1) -> "list[PhysicsBody3D]":
        p = self._p
        # Broadphase: candidatos cujo AABB cruza o da esfera.
        aabb_min = [center.x - radius, center.y - radius, center.z - radius]
        aabb_max = [center.x + radius, center.y + radius, center.z + radius]
        overlaps = p.getOverlappingObjects(aabb_min, aabb_max,
                                           physicsClientId=self._client) or []

        # Narrowphase: esfera temporária + getClosestPoints pra distância real.
        shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                          physicsClientId=self._client)
        probe = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=shape_id,
            basePosition=[center.x, center.y, center.z],
            physicsClientId=self._client,
        )
        result: list[PhysicsBody3D] = []
        seen: set[int] = set()
        try:
            for uid, _link in overlaps:
                if uid in seen or uid == probe:
                    continue
                seen.add(uid)
                body = self._bodies_by_uid.get(uid)
                if body is None:
                    continue
                if mask != -1 and not (body.collision_group & mask):
                    continue
                if p.getClosestPoints(probe, uid, 0.0, physicsClientId=self._client):
                    result.append(body)
        finally:
            p.removeBody(probe, physicsClientId=self._client)
        return result

    # -- ciclo de vida --------------------------------------------------------

    def destroy(self) -> None:
        try:
            self._p.disconnect(physicsClientId=self._client)
        except Exception:
            pass
        self.bodies.clear()
        self._bodies_by_uid.clear()
        self._contacts.clear()
        self._triggers.clear()
