from EasyCells3D.Components import Component
from EasyCells3D.PhysicsComponents import Rigidbody
from UserComponents.platform.Player import Player


class OneWayPlatform(Component):
    rigidbody: Rigidbody

    def init(self):
        self.rigidbody = self.GetComponent(Rigidbody)

    def loop(self):
        self.rigidbody.enable = True

        if Player.player_main.input.get_axis().y < -0.5:
            self.rigidbody.enable = False

        if Player.player_main.rb.velocity.y < -100:
            self.rigidbody.enable = False
