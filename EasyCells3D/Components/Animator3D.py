from typing import Callable, Any

import pyray as rl
from dataclasses import dataclass, field
from EasyCells3D.Components import Component


@dataclass
class AnimationEvent:
    frame: int
    name: str
    payload: Any = None
    callback: Callable | None = None


@dataclass
class AnimationClip:
    name: str
    index: int = 0
    frame_count: int = 0
    fps: float = 24.0
    loop: bool = True
    events: list[AnimationEvent] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    def add_event(self, frame: int, name: str, payload=None, callback=None) -> AnimationEvent:
        event = AnimationEvent(frame, name, payload, callback)
        self.events.append(event)
        self.events.sort(key=lambda e: e.frame)
        return event

class Animator3D(Component):
    def __init__(self, current_animation: str | None = None, speed: float = 1.0, autoplay: bool = True):
        self.model: AnimatedModel | None = None
        self.current_clip: AnimationClip | None = None
        self.current_animation = current_animation
        self.speed = speed
        self.autoplay = autoplay
        self.playing = autoplay
        self.time = 0.0
        self.frame = 0
        self.event_handlers: dict[str, list] = {}

    def init(self):
        from EasyCells3D.Components.AnimatedModel import AnimatedModel

        self.model = self.GetComponent(AnimatedModel)
        if not self.model:
            raise ValueError("Animator3D precisa de um AnimatedModel no mesmo Item ou em filhos.")

        if self.current_animation is None and self.model.clips:
            self.current_animation = next(iter(self.model.clips))
        if self.current_animation:
            self.play(self.current_animation, restart=True)

    def loop(self):
        if not self.playing or not self.current_clip or not self.model:
            return

        previous_frame = self.frame
        self.time += self.game.delta_time * self.speed
        clip = self.current_clip
        total_frames = max(clip.frame_count, 1)
        raw_frame = int(self.time * clip.fps)

        if clip.loop:
            self.frame = raw_frame % total_frames
        else:
            self.frame = min(raw_frame, total_frames - 1)
            if self.frame >= total_frames - 1:
                self.playing = False

        self._apply_frame()
        self._fire_events(previous_frame, self.frame, wrapped=clip.loop and self.frame < previous_frame)

    def play(self, name: str, restart: bool = False, speed: float | None = None):
        if not self.model:
            self.current_animation = name
            return

        if self.current_clip and self.current_clip.name == name and not restart:
            self.playing = True
            return

        self.current_animation = name
        self.current_clip = self.model.get_clip(name)
        self.time = 0.0
        self.frame = 0
        if speed is not None:
            self.speed = speed
        self.playing = True
        self._apply_frame()

    def stop(self):
        self.playing = False

    def resume(self):
        self.playing = True

    def add_event(self, animation: str, frame: int, name: str, payload=None, callback=None) -> AnimationEvent:
        clip = self.model.get_clip(animation) if self.model else None
        if clip is None:
            raise ValueError("Animator3D ainda nao foi inicializado.")
        return clip.add_event(frame, name, payload, callback)

    def on_event(self, name: str, callback):
        self.event_handlers.setdefault(name, []).append(callback)

    def _apply_frame(self):
        if not self.model or not self.model.model or not self.current_clip:
            return
        animation = self.model.get_animation(self.current_clip)
        rl.update_model_animation(self.model.model, animation, self.frame)

    def _fire_events(self, previous_frame: int, current_frame: int, wrapped: bool):
        if not self.current_clip:
            return
        for event in self.current_clip.events:
            should_fire = previous_frame < event.frame <= current_frame
            if wrapped:
                should_fire = event.frame > previous_frame or event.frame <= current_frame
            if should_fire:
                self._dispatch_event(event)

    def _dispatch_event(self, event: AnimationEvent):
        if event.callback:
            event.callback(self, event)

        for callback in self.event_handlers.get(event.name, []):
            callback(self, event)

        for component in set(self.item.components.values()):
            handler = getattr(component, "on_animation_event", None)
            if handler and component is not self:
                handler(self, event)
