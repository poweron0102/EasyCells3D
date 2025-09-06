# from .Component import Component
# from .Sprite import Sprite
#
#
# class Animation:
#     speed: float
#     frames: list[int]
#     current_frame: int = 0
#     animator: 'Animator'
#
#     def __init__(self, speed: float, frames: list[int], on_end: str = None):
#         self.speed = speed
#         self.frames = frames
#         self.on_end = on_end
#
#     def next_frame(self) -> int | None:
#         self.current_frame += 1
#         if self.current_frame >= len(self.frames):
#             self.current_frame = 0
#             if self.on_end:
#                 self.animator.current_animation = self.on_end
#                 return None
#
#         return self.frames[self.current_frame]
#
#
# class Animator(Component):
#     sprite: Sprite
#
#     _current_animation: str
#
#     @property
#     def current_animation(self):
#         return self._current_animation
#
#     @current_animation.setter
#     def current_animation(self, value: str | None):
#         if self._current_animation is None and value is not None:
#             self._current_animation = value
#             self.game.scheduler.add_dict_generator(self, self.run_animation())
#             return
#         self._current_animation = value
#         if value is None:
#             self.stop_animation()
#             return
#
#         self.dict_animations[self.current_animation].current_frame = 0
#         self.game.scheduler.change_time_dict_generator(self, 0)
#
#     def __init__(self, dict_animations: dict[str, Animation], current_animation: str):
#         self.dict_animations: dict[str, Animation] = dict_animations
#         self._current_animation: str = current_animation
#         for key in self.dict_animations:
#             self.dict_animations[key].animator = self
#
#     def init(self):
#         self.sprite = self.GetComponent(Sprite)
#         self.game.scheduler.add_dict_generator(self, self.run_animation())
#
#     def run_animation(self):
#         while True:
#             animation = self.dict_animations[self.current_animation]
#             next_index = animation.next_frame()
#             self.sprite.index = next_index if next_index is not None else self.sprite.index
#             yield animation.speed
#
#     def stop_animation(self):
#         self.game.scheduler.remove_dict_generator(self)
