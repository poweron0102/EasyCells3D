from EasyCells3D import Tick
from EasyCells3D.Components import Component, Camera
from EasyCells3D.Geometry import Vec3, Quaternion

import pygame as pg

class FreeCam(Component):
    camera: Camera
    
    
    def __init__(self, speed: float = 4.0, mouse_sensitivity: float = 0.2):
        """
        Cria um componente de câmera livre controlável pelo mouse e teclado.
        :param speed: A velocidade de movimento da câmera.
        :param mouse_sensitivity: A sensibilidade da rotação da câmera com o mouse.
        """
        super().__init__()
        self.speed = speed
        self.mouse_sensitivity = mouse_sensitivity
        self.mouse_on = True
        self.escape_tick = Tick(1)
    
    def init(self):
        self.camera = self.GetComponent(Camera)
    
    def loop(self):
        if not self.mouse_on:
            dx, dy = pg.mouse.get_rel()

            if dx != 0:
                yaw_rotation = Quaternion.from_axis_angle(Vec3(0, 1, 0), -dx * self.mouse_sensitivity * self.game.delta_time)
                self.camera.transform.rotation = yaw_rotation * self.camera.transform.rotation

            if dy != 0:
                right_vector = self.camera.transform.right
                pitch_rotation = Quaternion.from_axis_angle(right_vector, -dy * self.mouse_sensitivity * self.game.delta_time)

                potential_rotation = pitch_rotation * self.camera.transform.rotation

                if potential_rotation.rotate_vector(Vec3(0, 1, 0)).y > 0:
                    self.camera.transform.rotation = potential_rotation

            self.camera.transform.rotation = self.camera.transform.rotation.normalize()


        forward_vector = self.camera.transform.forward
        right_vector = self.camera.transform.right

        movement_input = Vec3(0.0, 0.0, 0.0)

        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            movement_input += forward_vector
        if keys[pg.K_s]:
            movement_input -= forward_vector
        if keys[pg.K_a]:
            movement_input -= right_vector
        if keys[pg.K_d]:
            movement_input += right_vector

        if movement_input.magnitude() > 0:
            movement_input = movement_input.normalize()

        self.camera.transform.position += movement_input * (self.speed * self.game.delta_time)

        if keys[pg.K_ESCAPE] and self.escape_tick():
            self.mouse_on = not self.mouse_on
            pg.mouse.set_visible(self.mouse_on)
            pg.event.set_grab(not self.mouse_on)