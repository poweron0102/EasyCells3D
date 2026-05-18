import pyray as rl

from EasyCells3D.Components.Animator3D import AnimationClip
from EasyCells3D.Components.Camera3D import Renderable3D
from EasyCells3D.Components.StaticModel import quaternion_to_axis_angle, resolve_asset_path


class AnimatedModel(Renderable3D):
    def __init__(
            self,
            model_path: str,
            clip_names: list[str] | None = None,
            clip_fps: float = 24.0,
            color: rl.Color = rl.WHITE,
            base_path: str = "Assets"
    ):
        super().__init__()
        self.model_path = model_path
        self.clip_names = clip_names or []
        self.clip_fps = clip_fps
        self.color = color
        self.base_path = base_path
        self.model: rl.Model | None = None
        self.animations = []
        self.clips: dict[str, AnimationClip] = {}

    def init(self):
        super().init()
        resolved_path = resolve_asset_path(self.model_path, self.base_path)
        self.model = rl.load_model(resolved_path)
        self.animations = list(rl.load_model_animations(resolved_path) or [])
        self._build_clips()

    def _build_clips(self):
        self.clips.clear()
        for index, animation in enumerate(self.animations):
            name = self._animation_name(animation, index)
            frame_count = int(getattr(animation, "frameCount", getattr(animation, "frame_count", 0)))
            clip = AnimationClip(name=name, index=index, frame_count=frame_count, fps=self.clip_fps)
            self.clips[name] = clip

    def _animation_name(self, animation, index: int) -> str:
        if index < len(self.clip_names):
            return self.clip_names[index]

        raw_name = getattr(animation, "name", None)
        if isinstance(raw_name, bytes):
            decoded = raw_name.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            if decoded:
                return decoded
        if isinstance(raw_name, str) and raw_name:
            return raw_name
        return f"Animation_{index}"

    def get_clip(self, name: str) -> AnimationClip:
        return self.clips[name]

    def get_animation(self, clip: AnimationClip):
        return self.animations[clip.index]

    def on_destroy(self):
        super().on_destroy()
        if self.animations:
            try:
                rl.unload_model_animations(self.animations, len(self.animations))
            except TypeError:
                for animation in self.animations:
                    rl.unload_model_animation(animation)
            self.animations = []
        if self.model:
            rl.unload_model(self.model)
            self.model = None

    def render(self):
        if not self.model:
            return
        pos = self.global_transform.position.to_raylib()
        axis, angle = quaternion_to_axis_angle(self.global_transform.rotation)
        scale = rl.Vector3(self.global_transform.scale.x, self.global_transform.scale.y, self.global_transform.scale.z)
        rl.draw_model_ex(self.model, pos, axis, angle, scale, self.color)
