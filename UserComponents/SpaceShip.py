import math

import pygame as pg
from EasyCells3D import Tick

from EasyCells3D.Components import Component, Camera
from EasyCells3D.Geometry import Vec3, Quaternion


class SpaceShip(Component):
    """
    Componente que implementa os controles de uma nave espacial 3D,
    incluindo o controle de uma câmera seguidora.
    """

    def __init__(self, camera: Camera, acceleration: float = 5.0, rotation_speed: float = 5.0,
                 ship_turn_speed: float = 4.0, mouse_sensitivity: float = 0.15, damping: float = 0.98,
                 max_pitch_angle: float = 30.0, max_camera_angle: float = 25.0,
                 min_cam_dist: float = 5.0, max_cam_dist: float = 15.0):
        """
        Inicializa o componente da nave espacial.

        :param camera: A instância da câmera que seguirá a nave.
        :param acceleration: A aceleração da nave em unidades/s^2.
        :param rotation_speed: A velocidade de rotação em graus/s.
        :param ship_turn_speed: A velocidade com que a nave se alinha à câmera.
        :param mouse_sensitivity: Sensibilidade da rotação com o mouse.
        :param damping: Fator de frenagem passiva (quanto menor, mais rápido para).
        :param min_cam_dist: A distância mínima da câmera à nave.
        :param max_cam_dist: A distância máxima da câmera à nave.
        :param cam_offset: Um deslocamento vertical para a posição da câmera.
        :param max_pitch_angle: O ângulo máximo de inclinação (pitch) em graus.
        """
        super().__init__()
        self.camera = camera
        self.acceleration = acceleration
        self.rotation_speed_rad = math.radians(rotation_speed)  # Para rolagem (roll)
        self.ship_turn_speed = ship_turn_speed
        self.mouse_sensitivity = mouse_sensitivity
        self.damping = damping
        self.velocity = Vec3(0, 0, 0)
        self.max_pitch_cos = math.cos(math.radians(max_pitch_angle))
        self.max_camera_angle = max_camera_angle
        self.min_cam_dist = min_cam_dist
        self.max_cam_dist = max_cam_dist

        self.mouse_on = False
        self.escape_tick = Tick(1)
        pg.mouse.set_visible(self.mouse_on)
        pg.event.set_grab(not self.mouse_on)

        self.start_camera_position = Vec3(0, 0, 0)
        self.start_camera_forward = Vec3(0, 0, 0)

    def init(self):
        self.start_camera_position = self.camera.transform.position
        self.start_camera_forward = self.camera.transform.forward


    def loop(self):
        dt = self.game.delta_time
        keys = pg.key.get_pressed()

        # --- Controle do Mouse ---
        if keys[pg.K_ESCAPE] and self.escape_tick():
            self.mouse_on = not self.mouse_on
            pg.mouse.set_visible(self.mouse_on)
            pg.event.set_grab(not self.mouse_on)

        # --- Rotação da Câmera e da Nave ---
        if not self.mouse_on:
            dx, dy = pg.mouse.get_rel()

            # 1. Rotação instantânea da câmera (local)
            cam_rot_x = Quaternion.from_axis_angle(Vec3(0, 1, 0), -dx * self.mouse_sensitivity * dt)
            cam_rot_y = Quaternion.from_axis_angle(Vec3(1, 0, 0), -dy * self.mouse_sensitivity * dt)
            self.camera.transform.rotation = cam_rot_x * cam_rot_y * self.camera.transform.rotation

            # 2. Limita o ângulo da câmera em relação à nave
            forward_cam = self.camera.transform.rotation.rotate_vector(Vec3(0, 0, 1))
            angle_with_ship_forward = math.degrees(math.acos(forward_cam.dot(Vec3(0, 0, 1))))
            if angle_with_ship_forward > self.max_camera_angle:
                axis = Vec3(0, 0, 1).cross(forward_cam).normalize()
                self.camera.transform.rotation = Quaternion.from_axis_angle(axis, math.radians(self.max_camera_angle))

            # 3. A nave interpola suavemente em direção à rotação da câmera
            target_ship_rot = self.transform.rotation * self.camera.transform.rotation
            self.transform.rotation = self.transform.rotation.lerp(target_ship_rot, dt * self.ship_turn_speed)
            # A câmera "relaxa" de volta para a frente da nave
            self.camera.transform.rotation = self.camera.transform.rotation.lerp(Quaternion.identity(), dt * self.ship_turn_speed)

        # Roll (rolagem) com as teclas A e D
        if keys[pg.K_a]:
            roll_rotation = Quaternion.from_axis_angle(self.transform.forward, self.rotation_speed_rad * dt)
            self.transform.rotation = roll_rotation * self.transform.rotation
        if keys[pg.K_d]:
            roll_rotation = Quaternion.from_axis_angle(self.transform.forward, -self.rotation_speed_rad * dt)
            self.transform.rotation = roll_rotation * self.transform.rotation

        self.transform.rotation = self.transform.rotation.normalize()

        # --- Movimento ---
        accelerating = False
        # Aceleração para frente com W
        if keys[pg.K_w]:
            self.velocity += self.transform.forward * self.acceleration * dt
            accelerating = True

        # Freio/Ré com S
        if keys[pg.K_s]:
            # Aplica uma força de frenagem na direção oposta da velocidade
            brake_force = self.velocity.normalize() * -1 * self.acceleration * dt
            # Garante que o freio não inverta a velocidade, apenas a reduza a zero
            if self.velocity.magnitude() > brake_force.magnitude():
                self.velocity += brake_force
            else:
                self.velocity = Vec3.zero()
            accelerating = True

        # Frenagem passiva (arrasto espacial) se não estiver acelerando/freiando
        if not accelerating:
            self.velocity *= (self.damping ** dt)

        # Atualiza a posição da nave com base na velocidade
        self.transform.position += self.velocity * dt

        # --- Controle da Câmera ---
        # A distância da câmera aumenta com a velocidade
        # Usamos a velocidade máxima teórica para normalizar (ex: 10s de aceleração)
        max_speed = self.acceleration * 10
        speed_ratio = min(self.velocity.magnitude() / max_speed, 1.0) if max_speed > 0 else 0
        cam_dist = self.min_cam_dist + (self.max_cam_dist - self.min_cam_dist) * speed_ratio

        # Ajusta a posição local da câmera para trás, mantendo a altura e o offset inicial
        target_cam_pos = self.start_camera_position.normalize() * cam_dist
        self.camera.transform.position = self.camera.transform.position.lerp(target_cam_pos, dt * 5)