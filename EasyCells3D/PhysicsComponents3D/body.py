"""``PhysicsBody3D`` — corpo físico combinado (collider + rigidbody).

Um único componente representa o corpo de colisão e seu comportamento dinâmico.
A *fonte da verdade* depende do ``body_type``:

- ``STATIC``   — a engine define a pose no spawn; o corpo nunca se move.
- ``DYNAMIC``  — o Bullet simula; a pose é puxada pro ``Transform`` todo frame.
- ``KINEMATIC``— a engine empurra a pose (via ``Transform``) pro Bullet todo
  frame; carrega/empurra dinâmicos mas não é empurrado por eles.

Tudo aqui é construído pensando em serialização (campos primitivos + shapes com
``to_dict``/``from_dict``), mesmo sem integrar o ``SceneLoader`` na v1.
"""
from __future__ import annotations

import math
import traceback
from typing import TYPE_CHECKING

from ..Components import Component
from ..Geometry import Quaternion, Vec3
from .bullet_math import bullet_to_quat, bullet_to_vec3, quat_to_bullet, vec3_to_bullet
from .shapes import CollisionShape, TriangleMeshShape
from .world import BodyType, BulletPhysicsWorld

if TYPE_CHECKING:
    from .world import PhysicsWorld


class PhysicsBody3D(Component):
    def __init__(
            self,
            shape: CollisionShape | None = None,
            mass: float = 0.0,
            body_type: BodyType = BodyType.DYNAMIC,
            is_trigger: bool = False,
            friction: float = 0.5,
            restitution: float = 0.0,
            linear_damping: float = 0.0,
            angular_damping: float = 0.05,
            gravity_scale: float = 1.0,
            allow_sleep: bool = True,
            lock_rotation: bool = False,
            collision_group: int = 1,
            collision_mask: int = -1,
    ):
        self.shape = shape
        self._body_type = body_type
        self.is_trigger = is_trigger
        self.friction = friction
        self.restitution = restitution
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.gravity_scale = gravity_scale
        self.allow_sleep = allow_sleep
        self.lock_rotation = lock_rotation
        self.collision_group = collision_group
        self.collision_mask = collision_mask

        # Massa dinâmica. STATIC/KINEMATIC sempre usam 0 no Bullet.
        if body_type == BodyType.DYNAMIC and mass <= 0.0:
            mass = 1.0
        self.mass = mass

        self.uid: int | None = None
        self._world: BulletPhysicsWorld | None = None
        self._p = None
        self._client: int | None = None
        self._initialized = False
        self._enabled = True

        # Pose anterior do KINEMATIC (pra derivar velocidade).
        self._prev_kin_pos: Vec3 | None = None
        self._prev_kin_rot: Quaternion | None = None

    # -- ciclo de vida --------------------------------------------------------

    def init(self):
        world = self.game.physics_world
        assert world is not None, (
            "PhysicsBody3D requer game.physics_world. Crie um "
            "BulletPhysicsWorld() no init() do level antes de adicionar corpos."
        )
        assert isinstance(world, BulletPhysicsWorld), (
            "PhysicsBody3D atualmente só suporta BulletPhysicsWorld."
        )
        self._world = world
        self._p = world._p
        self._client = world.client

        self._resolve_shape()

        if self._enabled:
            self._create_body()
        self._initialized = True
        world.add_body(self)

    def _resolve_shape(self):
        """Garante que ``self.shape`` é válida pro tipo de corpo (auto-build M4)."""
        if self.shape is None:
            built = self._auto_build_shape()
            if built is not None:
                self.shape = built

        if self.shape is None:
            raise ValueError(
                f"PhysicsBody3D em '{self.item.name}' sem shape. Passe uma shape "
                f"(ex.: BoxShape(...)) ou use STATIC com um StaticModel para "
                f"auto-construir uma triangle mesh."
            )

        if (self._body_type == BodyType.DYNAMIC
                and isinstance(self.shape, TriangleMeshShape)):
            raise ValueError(
                "Triangle mesh dinâmica é proibida (côncava). Use uma primitiva "
                "ou ConvexHullShape para corpos DYNAMIC."
            )

    def _auto_build_shape(self) -> CollisionShape | None:
        """Se STATIC e o item tem StaticModel, monta uma triangle mesh dele."""
        if self._body_type != BodyType.STATIC:
            return None
        try:
            from ..Components.StaticModel import StaticModel
        except Exception:
            return None
        sm = self.GetComponent(StaticModel)
        if sm is None or sm.model is None:
            return None
        from .shapes import extract_mesh_geometry
        vertices, indices = extract_mesh_geometry(sm.model)
        if not vertices or not indices:
            return None
        return TriangleMeshShape(vertices, indices)

    def _create_body(self):
        p = self._p
        gt = self.item.global_transform_get()
        shape_id = self.shape.build(self._client, gt.scale)

        base_mass = self.mass if self._body_type == BodyType.DYNAMIC else 0.0
        self.uid = p.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=shape_id,
            basePosition=vec3_to_bullet(gt.position),
            baseOrientation=quat_to_bullet(gt.rotation),
            physicsClientId=self._client,
        )

        self._apply_dynamics()
        self._apply_collision_filter()

        if self._body_type == BodyType.KINEMATIC:
            self._prev_kin_pos = gt.position
            self._prev_kin_rot = gt.rotation

    def _apply_dynamics(self):
        p = self._p
        kwargs = dict(
            lateralFriction=self.friction,
            restitution=self.restitution,
            linearDamping=self.linear_damping,
            angularDamping=self.angular_damping,
            physicsClientId=self._client,
        )
        if not self.allow_sleep or self._body_type != BodyType.DYNAMIC:
            kwargs["activationState"] = p.ACTIVATION_STATE_DISABLE_SLEEPING
        p.changeDynamics(self.uid, -1, **kwargs)

    def _apply_collision_filter(self):
        # Grupo/máscara nativos pra filtragem de colisão. A *não-resposta* dos
        # triggers é tratada pelo world via setCollisionFilterPair (a regra de
        # group/mask deste build não basta sozinha pra criar um sensor).
        p = self._p
        try:
            p.setCollisionFilterGroupMask(
                self.uid, -1,
                collisionFilterGroup=self.collision_group,
                collisionFilterMask=self.collision_mask,
                physicsClientId=self._client,
            )
        except Exception:
            pass

    def on_destroy(self):
        # Só remove se o world registrado ainda for o vivo (evita mexer num
        # cliente já destruído na troca de cena).
        if (self._world is not None
                and self.game.physics_world is self._world
                and self.uid is not None):
            try:
                self._p.removeBody(self.uid, physicsClientId=self._client)
            except Exception:
                pass
            self._world.remove_body(self)
        self.uid = None

    # -- enable como property -------------------------------------------------

    @property
    def enable(self) -> bool:
        return self._enabled

    @enable.setter
    def enable(self, value: bool):
        value = bool(value)
        if value == self._enabled:
            return
        self._enabled = value
        if not self._initialized:
            return
        if value:
            # Reinsere na pose atual do Transform.
            self._create_body()
            self._world.add_body(self)
        else:
            if self.uid is not None:
                try:
                    self._p.removeBody(self.uid, physicsClientId=self._client)
                except Exception:
                    pass
                self._world.remove_body(self)
                self.uid = None

    # -- body_type como property setável (M3) ---------------------------------

    @property
    def body_type(self) -> BodyType:
        return self._body_type

    @body_type.setter
    def body_type(self, value: BodyType):
        if value == self._body_type:
            return
        old = self._body_type
        self._body_type = value

        if not self._initialized or self.uid is None:
            return

        p = self._p
        if value == BodyType.DYNAMIC:
            if self.mass <= 0.0:
                self.mass = 1.0
            p.changeDynamics(self.uid, -1, mass=self.mass, physicsClientId=self._client)
            self._wake()
        else:
            # STATIC/KINEMATIC: massa 0 e velocidade zerada.
            p.changeDynamics(self.uid, -1, mass=0.0, physicsClientId=self._client)
            p.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0],
                                physicsClientId=self._client)
            if value == BodyType.KINEMATIC:
                gt = self.item.global_transform_get()
                self._prev_kin_pos = gt.position
                self._prev_kin_rot = gt.rotation

        # Reaplica sleep/activation conforme o novo tipo.
        self._apply_dynamics()
        _ = old

    # -- sincronização (chamado pelo world.step) ------------------------------

    def _push_kinematic(self, dt: float):
        if self.uid is None:
            return
        p = self._p
        gt = self.item.global_transform_get()
        pos, rot = gt.position, gt.rotation
        p.resetBasePositionAndOrientation(
            self.uid, vec3_to_bullet(pos), quat_to_bullet(rot),
            physicsClientId=self._client,
        )

        if self._prev_kin_pos is not None and dt > 0.0:
            lin = (pos - self._prev_kin_pos) / dt
            ang = self._angular_velocity_from(self._prev_kin_rot, rot, dt)
            p.resetBaseVelocity(
                self.uid, vec3_to_bullet(lin), vec3_to_bullet(ang),
                physicsClientId=self._client,
            )
        self._prev_kin_pos = pos
        self._prev_kin_rot = rot

    def _pull_dynamic(self):
        if self.uid is None:
            return
        p = self._p
        pos, orn = p.getBasePositionAndOrientation(self.uid, physicsClientId=self._client)
        world_pos = bullet_to_vec3(pos)
        world_rot = bullet_to_quat(orn)

        if self.lock_rotation:
            lin, _ang = p.getBaseVelocity(self.uid, physicsClientId=self._client)
            p.resetBaseVelocity(self.uid, lin, [0, 0, 0], physicsClientId=self._client)

        # Escreve no Transform local (mundo -> local) e atualiza o cache global
        # pra o render deste frame já refletir a nova pose.
        self.item.global_position_set(world_pos)
        self.item.global_rotation_set(world_rot)
        gt = self.item.global_transform
        gt.position = world_pos
        gt.rotation = world_rot

    @staticmethod
    def _angular_velocity_from(prev: Quaternion, cur: Quaternion, dt: float) -> Vec3:
        dq = cur * prev.inverse()
        dq = dq.normalize()
        w = max(-1.0, min(1.0, dq.w))
        angle = 2.0 * math.acos(w)
        s = math.sqrt(max(0.0, 1.0 - w * w))
        if s < 1e-6 or dt <= 0.0:
            return Vec3.zero()
        axis = Vec3(dq.x / s, dq.y / s, dq.z / s)
        if angle > math.pi:
            angle -= 2.0 * math.pi
        return axis * (angle / dt)

    def _apply_gravity_compensation(self, gravity: Vec3):
        """Corrige a gravidade do corpo pra ``gravity_scale`` (o Bullet só tem
        gravidade global). Aplica ``(scale - 1) * g * massa`` por substep."""
        if self.uid is None or self.mass <= 0.0:
            return
        factor = (self.gravity_scale - 1.0) * self.mass
        force = Vec3(gravity.x * factor, gravity.y * factor, gravity.z * factor)
        p = self._p
        pos, _ = p.getBasePositionAndOrientation(self.uid, physicsClientId=self._client)
        p.applyExternalForce(self.uid, -1, vec3_to_bullet(force), pos,
                             p.WORLD_FRAME, physicsClientId=self._client)

    def _wake(self):
        if self.uid is not None:
            try:
                self._p.changeDynamics(self.uid, -1,
                                       activationState=self._p.ACTIVATION_STATE_WAKE_UP,
                                       physicsClientId=self._client)
            except Exception:
                pass

    # -- API de controle (M3) -------------------------------------------------

    @property
    def velocity(self) -> Vec3:
        if self.uid is None:
            return Vec3.zero()
        lin, _ang = self._p.getBaseVelocity(self.uid, physicsClientId=self._client)
        return bullet_to_vec3(lin)

    @velocity.setter
    def velocity(self, value: Vec3):
        if self.uid is None:
            return
        _lin, ang = self._p.getBaseVelocity(self.uid, physicsClientId=self._client)
        self._p.resetBaseVelocity(self.uid, vec3_to_bullet(value), ang,
                                  physicsClientId=self._client)
        self._wake()

    def apply_force(self, force: Vec3):
        """Aplica uma força contínua (no centro de massa, world-space).

        A força vale por uma chamada de ``stepSimulation``; chame todo frame
        enquanto quiser que ela aja.
        """
        if self.uid is None:
            return
        p = self._p
        pos, _ = p.getBasePositionAndOrientation(self.uid, physicsClientId=self._client)
        p.applyExternalForce(self.uid, -1, vec3_to_bullet(force), pos,
                             p.WORLD_FRAME, physicsClientId=self._client)
        self._wake()

    def apply_impulse(self, impulse: Vec3):
        """Aplica um impulso instantâneo (muda a velocidade direto)."""
        if self.uid is None or self.mass <= 0.0:
            return
        cur = self.velocity
        self.velocity = cur + impulse / self.mass

    def teleport(self, position: Vec3, rotation: Quaternion | None = None):
        """Reposiciona o corpo de forma limpa (zera a velocidade)."""
        if self.uid is None:
            return
        p = self._p
        if rotation is None:
            _, orn = p.getBasePositionAndOrientation(self.uid, physicsClientId=self._client)
        else:
            orn = quat_to_bullet(rotation)
        p.resetBasePositionAndOrientation(self.uid, vec3_to_bullet(position), orn,
                                          physicsClientId=self._client)
        p.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

        self.item.global_position_set(position)
        if rotation is not None:
            self.item.global_rotation_set(rotation)
        if self._body_type == BodyType.KINEMATIC:
            self._prev_kin_pos = position
            self._prev_kin_rot = rotation if rotation is not None else self._prev_kin_rot

    # -- eventos (M5) ---------------------------------------------------------

    def _fire_event(self, name: str, other: "PhysicsBody3D"):
        """Encaminha um evento de física pros componentes do item.

        Qualquer componente no mesmo item que defina ``on_collision_enter``,
        ``on_collision_exit``, ``on_trigger_enter`` ou ``on_trigger_exit``
        recebe o outro ``PhysicsBody3D`` como argumento.
        """
        for comp in self.item._unique_components():
            fn = getattr(comp, name, None)
            if callable(fn):
                try:
                    fn(other)
                except Exception as exc:
                    print(f"Error in {comp}.{name}: {exc}")
                    traceback.print_exc()
