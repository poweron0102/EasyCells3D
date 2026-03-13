from enum import IntEnum

import pyray as rl

from EasyCells3D import Vec2


class ControllerType(IntEnum):
    KEYBOARD = 0
    GAMEPAD1 = 1
    GAMEPAD2 = 2
    GAMEPAD3 = 3
    GAMEPAD4 = 4


class Action(IntEnum):
    JUMP = 0
    RUN = 1
    ATTACK = 2
    PAUSE = 3


class Input:
    """
    Responsável por abstrair a entrada do usuário (Teclado e Gamepad).
    """

    @property
    def gamepad_id(self):
        return self.controller_type.value - 1

    def __init__(self, controller_type: ControllerType):
        self.controller_type = controller_type

    def get_axis(self) -> Vec2:
        value = Vec2(0, 0)

        if self.controller_type == ControllerType.KEYBOARD:
            if rl.is_key_down(rl.KeyboardKey.KEY_RIGHT) or rl.is_key_down(rl.KeyboardKey.KEY_D):
                value.x = 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_LEFT) or rl.is_key_down(rl.KeyboardKey.KEY_A):
                value.x = -1.0

            if rl.is_key_down(rl.KeyboardKey.KEY_UP) or rl.is_key_down(rl.KeyboardKey.KEY_W):
                value.y = 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_DOWN) or rl.is_key_down(rl.KeyboardKey.KEY_S):
                value.y = -1.0

            return value.normalize()

        # Gamepad
        if rl.is_gamepad_available(self.gamepad_id):
            value.x = rl.get_gamepad_axis_movement(self.gamepad_id, rl.GamepadAxis.GAMEPAD_AXIS_LEFT_X)
            value.y = rl.get_gamepad_axis_movement(self.gamepad_id, rl.GamepadAxis.GAMEPAD_AXIS_LEFT_Y)

            return value

        return Vec2.zero()

    def is_action_just_pressed(self, action: Action) -> bool:
        """
        Retorna True no frame em que a ação foi pressionada.
        JUMP = Space or X
        RUN = Ctrl or R1
        ATTACK = Mouse1 or Square
        PAUSE = ESQ or Start
        """
        if self.controller_type == ControllerType.KEYBOARD:
            if action == Action.JUMP: return rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE)
            if action == Action.RUN: return rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT_CONTROL)
            if action == Action.ATTACK: return rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT)
            if action == Action.PAUSE: return rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE)
        elif rl.is_gamepad_available(self.gamepad_id):
            if action == Action.JUMP: return rl.is_gamepad_button_pressed(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)
            if action == Action.RUN: return rl.is_gamepad_button_pressed(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1)
            if action == Action.ATTACK: return rl.is_gamepad_button_pressed(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_LEFT)
            if action == Action.PAUSE: return rl.is_gamepad_button_pressed(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_MIDDLE_RIGHT)

        return False

    def is_action_pressed(self, action: Action) -> bool:
        """
        Retorna True se a ação esta pressionada.
        JUMP = Space or X
        RUN = Ctrl or R1
        ATTACK = Mouse1 or Square
        PAUSE = ESQ or Start
        """
        if self.controller_type == ControllerType.KEYBOARD:
            if action == Action.JUMP: return rl.is_key_down(rl.KeyboardKey.KEY_SPACE)
            if action == Action.RUN: return rl.is_key_down(rl.KeyboardKey.KEY_LEFT_CONTROL)
            if action == Action.ATTACK: return rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
            if action == Action.PAUSE: return rl.is_key_down(rl.KeyboardKey.KEY_ESCAPE)
        elif rl.is_gamepad_available(self.gamepad_id):
            if action == Action.JUMP: return rl.is_gamepad_button_down(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)
            if action == Action.RUN: return rl.is_gamepad_button_down(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_TRIGGER_1)
            if action == Action.ATTACK: return rl.is_gamepad_button_down(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_RIGHT_FACE_LEFT)
            if action == Action.PAUSE: return rl.is_gamepad_button_down(self.gamepad_id, rl.GamepadButton.GAMEPAD_BUTTON_MIDDLE_RIGHT)

        return False

    @staticmethod
    def get_connected_controllers() -> list[ControllerType]:
        """
        Retorna uma lista de controladores conectados (incluindo Teclado).
        """
        controllers = [ControllerType.KEYBOARD]

        for controller in ControllerType:
            if controller == ControllerType.KEYBOARD:
                continue

            # O ID do gamepad é o valor do enum - 1 (definido na property gamepad_id)
            gp_id = controller.value - 1
            if rl.is_gamepad_available(gp_id):
                controllers.append(controller)

        return controllers
