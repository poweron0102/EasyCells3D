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

    def init(self):
        super().init()
        self.game.scheduler.add_generator(self.sync())

    def sync(self):
        if self.owner == NetworkManager.instance.id:
            while True:
                data = self.serialize()
                if data[4:] != self.last_sent[4:]:
                    self.last_sent = data
                    self.sync_transform(data)
                yield self.sync_frequency

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
        index = 0

        cont = unpack("i", data[index:index + 4])[0]
        index += 4

        if cont < self.cont:
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
