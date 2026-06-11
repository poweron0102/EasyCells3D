from __future__ import annotations

import inspect
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Coroutine, TYPE_CHECKING

from .NewGame import NewGame

if TYPE_CHECKING:
    from .Game import Game


@dataclass
class _SleepRequest:
    seconds: float

    def __await__(self):
        yield self


@dataclass
class _NextFrameRequest:
    def __await__(self):
        yield self


class SchedulerTask:
    def __init__(
            self,
            coroutine: Coroutine[Any, Any, Any],
            ready_time: float,
            ready_frame: int,
            key: Any = None,
    ):
        self.coroutine = coroutine
        self.ready_time = ready_time
        self.ready_frame = ready_frame
        self.key = key
        self.cancelled = False
        self.done = False

    def cancel(self):
        self.cancelled = True
        self.close()

    def close(self):
        if self.done:
            return
        self.done = True
        try:
            self.coroutine.close()
        except RuntimeError:
            pass


class Scheduler:
    instance: 'Scheduler' = None

    def __init__(self, game: Game):
        self.game = game
        self._tasks: list[SchedulerTask] = []
        self._tasks_by_key: dict[Any, SchedulerTask] = {}
        self._pending_tasks: deque[SchedulerTask] = deque()
        self._lock = threading.RLock()
        self._frame = 0

        Scheduler.instance = self

    def update(self):
        self._drain_pending_tasks()
        self._frame += 1

        for task in list(self._tasks):
            if task.cancelled or task.done:
                self._remove_task(task)
                continue

            if task.ready_frame > self._frame or task.ready_time > self.game.run_time:
                continue

            try:
                request = task.coroutine.send(None)
                self._schedule_next_step(task, request)
            except StopIteration:
                self._remove_task(task)
            except (KeyboardInterrupt, SystemExit, NewGame) as e:
                raise e
            except Exception as e:
                print(f"Error in {task.coroutine}:\n    {e}")
                traceback.print_exc()
                self._remove_task(task)

    def create_task(
            self,
            coroutine: Coroutine[Any, Any, Any],
            delay: float = 0,
            key: Any = None,
    ) -> SchedulerTask:
        if not inspect.iscoroutine(coroutine):
            raise TypeError("Scheduler.create_task expects an async coroutine object")

        task = SchedulerTask(
            coroutine=coroutine,
            ready_time=self.game.run_time + max(delay, 0),
            ready_frame=self._frame,
            key=key,
        )

        with self._lock:
            if key is not None:
                self.cancel(key)
            self._pending_tasks.append(task)
            if key is not None:
                self._tasks_by_key[key] = task

        return task

    def cancel(self, task_or_key: SchedulerTask | Any):
        with self._lock:
            if isinstance(task_or_key, SchedulerTask):
                task = task_or_key
            else:
                task = self._tasks_by_key.get(task_or_key)

            if task is None:
                return

            task.cancel()
            if task.key is not None and self._tasks_by_key.get(task.key) is task:
                self._tasks_by_key.pop(task.key, None)

    def sleep(self, seconds: float) -> _SleepRequest:
        return _SleepRequest(max(seconds, 0))

    def next_frame(self) -> _NextFrameRequest:
        return _NextFrameRequest()

    def clear(self):
        with self._lock:
            for task in self._tasks:
                task.cancel()
            for task in self._pending_tasks:
                task.cancel()
            self._tasks.clear()
            self._tasks_by_key.clear()
            self._pending_tasks.clear()

    def _drain_pending_tasks(self):
        with self._lock:
            while self._pending_tasks:
                task = self._pending_tasks.popleft()
                if not task.cancelled:
                    self._tasks.append(task)

    def _schedule_next_step(self, task: SchedulerTask, request: Any):
        if isinstance(request, _SleepRequest):
            task.ready_time = self.game.run_time + request.seconds
            task.ready_frame = self._frame
            return

        if isinstance(request, _NextFrameRequest):
            task.ready_time = self.game.run_time
            task.ready_frame = self._frame + 1
            return

        raise TypeError(
            "Scheduler tasks can only await scheduler.sleep(...) "
            "or scheduler.next_frame()"
        )

    def _remove_task(self, task: SchedulerTask):
        if task in self._tasks:
            self._tasks.remove(task)
        if task.key is not None and self._tasks_by_key.get(task.key) is task:
            self._tasks_by_key.pop(task.key, None)
        task.close()


class Tick:
    def __init__(self, time: float):
        self.time = time
        self.next_ready_time = 0
        self.on = True

    def turn_off(self):
        self.on = False

    def turn_on(self):
        self.on = True
        self.next_ready_time = Scheduler.instance.game.run_time

    def reset(self):
        self.on = False
        self.next_ready_time = Scheduler.instance.game.run_time + self.time

    def __call__(self) -> bool:
        scheduler = Scheduler.instance
        if scheduler is None:
            return False

        if not self.on and scheduler.game.run_time >= self.next_ready_time:
            self.on = True

        if self.on:
            self.on = False
            self.next_ready_time = scheduler.game.run_time + self.time
            return True

        return False
