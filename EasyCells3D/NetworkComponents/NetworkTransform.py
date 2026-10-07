from .NetworkComponent import NetworkComponent, SendTo, Rpc, NetworkManager, Protocol
from ..Geometry import Vec3, Quaternion

from struct import pack, unpack


class NetworkTransform(NetworkComponent):
    def __init__(
            self,
            identifier: int,
            owner: int,
            sync_frequency: float = 0.015,
            
            sync_x: bool = True,
            sync_y: bool = True,
            sync_z: bool = True,
            
            sync_rot_x: bool = True,
            sync_rot_y: bool = True,
            sync_rot_z: bool = True,
            
            sync_scale_x: bool = True,
            sync_scale_y: bool = True,
            sync_scale_z: bool = True,
            interpolation_speed: float = 0.0,
            teleport_distance: float = 5.0,
            heartbeat_interval: float = 0.5,
    ):
        super().__init__(identifier, owner)
        self.sync_frequency = sync_frequency
        
        self.sync_x = sync_x
        self.sync_y = sync_y
        self.sync_z = sync_z
        
        self.sync_rot_x = sync_rot_x
        self.sync_rot_y = sync_rot_y
        self.sync_rot_z = sync_rot_z

        self.sync_scale_x = sync_scale_x
        self.sync_scale_y = sync_scale_y
        self.sync_scale_z = sync_scale_z

        self.cont = 0
        self.last_sent = b""
        self.interpolation_speed = interpolation_speed
        self.teleport_distance = teleport_distance
        self.heartbeat_interval = heartbeat_interval
        self._last_sync_time = float("-inf")
        self._target_position = None

    def init(self):
        super().init()
        self.game.scheduler.create_task(self.sync(), key=self)

    async def sync(self):
        while True:
            if self.owner == NetworkManager.instance.id:
                data = self.serialize()
                if (data[4:] != self.last_sent[4:] or
                        self.game.run_time - self._last_sync_time >= self.heartbeat_interval):
                    self.last_sent = data
                    self._last_sync_time = self.game.run_time
                    self.sync_transform(data)
            await self.game.scheduler.sleep(self.sync_frequency)

    def loop(self):
        if self.owner == NetworkManager.instance.id:
            self._target_position = None
        elif self._target_position is not None:
            position = self.transform.position
            t = min(1.0, self.interpolation_speed * self.game.delta_time)
            self.transform.position = position + (self._target_position - position) * t

    def on_destroy(self):
        self.game.scheduler.cancel(self)
        super().on_destroy()

    @Rpc(send_to=SendTo.NOT_ME, require_owner=True, protocol=Protocol.UDP)
    def sync_transform(self, data: bytes):
        self.deserialize(data)

    def serialize(self) -> bytes:
        data: bytes = b""

        self.cont += 1
        data += pack("i", self.cont)

        position = self.transform.position
        if self.sync_x:
            data += pack("f", position.x)
        if self.sync_y:
            data += pack("f", position.y)
        if self.sync_z:
            data += pack("f", position.z)
        
        rotation = self.transform.rotation.to_euler_angles()
        if self.sync_rot_x:
            data += pack("f", rotation.x)
        if self.sync_rot_y:
            data += pack("f", rotation.y)
        if self.sync_rot_z:
            data += pack("f", rotation.z)
            
        scale = self.transform.scale
        if self.sync_scale_x:
            data += pack("f", scale.x)
        if self.sync_scale_y:
            data += pack("f", scale.y)
        if self.sync_scale_z:
            data += pack("f", scale.z)
        

        return data

    def deserialize(self, data: bytes):
        flags = (self.sync_x, self.sync_y, self.sync_z,
                 self.sync_rot_x, self.sync_rot_y, self.sync_rot_z,
                 self.sync_scale_x, self.sync_scale_y, self.sync_scale_z)
        if not isinstance(data, bytes) or len(data) != 4 * (1 + sum(flags)):
            return
        previous_position = Vec3(self.transform.x, self.transform.y, self.transform.z)
        index = 0

        cont = unpack("i", data[index:index + 4])[0]
        index += 4

        if cont <= self.cont:
            return

        self.cont = cont

        if self.sync_x:
            self.transform.x = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_y:
            self.transform.y = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_z:
            self.transform.z = unpack("f", data[index:index + 4])[0]
            index += 4

        if self.sync_rot_x or self.sync_rot_y or self.sync_rot_z:
            euler = self.transform.rotation.to_euler_angles()
            if self.sync_rot_x:
                euler.x = unpack("f", data[index:index + 4])[0]
                index += 4
            if self.sync_rot_y:
                euler.y = unpack("f", data[index:index + 4])[0]
                index += 4
            if self.sync_rot_z:
                euler.z = unpack("f", data[index:index + 4])[0]
                index += 4
            self.transform.rotation = Quaternion.from_euler_angles(euler)

        if self.sync_scale_x:
            self.transform.scale.x = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_scale_y:
            self.transform.scale.y = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_scale_z:
            self.transform.scale.z = unpack("f", data[index:index + 4])[0]
            index += 4

        if self.interpolation_speed > 0:
            self._target_position = Vec3(self.transform.x, self.transform.y, self.transform.z)
            if (self._target_position - previous_position).magnitude() < self.teleport_distance:
                self.transform.position = previous_position
