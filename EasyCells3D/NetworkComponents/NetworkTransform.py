from .NetworkComponent import NetworkComponent, SendTo, Rpc, NetworkManager

from struct import pack, unpack


class NetworkTransform(NetworkComponent):
    def __init__(
            self,
            identifier: int,
            owner: int,
            sync_frequency: float = 0.015,
            sync_x: bool = True,
            sync_y: bool = True,
            sync_z: bool = False,
            sync_angle: bool = False,
            sync_scale: bool = False
    ):
        super().__init__(identifier, owner)
        self.sync_frequency = sync_frequency
        self.sync_x = sync_x
        self.sync_y = sync_y
        self.sync_z = sync_z
        self.sync_angle = sync_angle
        self.sync_scale = sync_scale

        self.last_sent = b""

    def init(self):
        super().init()
        self.game.scheduler.add_generator(self.sync())

    def sync(self):
        if self.owner == NetworkManager.instance.id:
            while True:
                data = self.serialize()
                if data != self.last_sent:
                    self.last_sent = data
                    self.sync_transform(data)
                yield self.sync_frequency

    @Rpc(send_to=SendTo.NOT_ME, require_owner=True)
    def sync_transform(self, data: bytes):
        self.deserialize(data)

    def serialize(self) -> bytes:
        data: bytes = b""

        if self.sync_x:
            data += pack("f", self.transform.x)
        if self.sync_y:
            data += pack("f", self.transform.y)
        if self.sync_z:
            data += pack("f", self.transform.z)
        if self.sync_angle:
            data += pack("f", self.transform.angle)
        if self.sync_scale:
            data += pack("f", self.transform.scale)

        return data

    def deserialize(self, data: bytes):
        index = 0

        if self.sync_x:
            self.transform.x = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_y:
            self.transform.y = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_z:
            self.transform.z = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_angle:
            self.transform.angle = unpack("f", data[index:index + 4])[0]
            index += 4
        if self.sync_scale:
            self.transform.scale = unpack("f", data[index:index + 4])[0]
            index += 4

