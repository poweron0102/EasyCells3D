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


@dataclass
class _TaskWaitRequest:
    task: 'SchedulerTask'

    def __await__(self):
        result = yield self
        return result


class SchedulerTaskCancelled(Exception):
    pass


class SchedulerTask:
    def __init__(
            self,
            scheduler: 'Scheduler',
            coroutine: Coroutine[Any, Any, Any],
            key: Any = None,
    ):
        self.scheduler = scheduler
        self.coroutine = coroutine
        self.ready_time = scheduler.game.run_time
        self.ready_frame = scheduler._frame
        self.key = key
        self.started = False
        self.cancelled = False
        self.done = False
        self.result_value: Any = None
        self.exception: BaseException | None = None
        self._waiters: list[SchedulerTask] = []
        self._send_value: Any = None
        self._throw_exception: BaseException | None = None

    def __await__(self):
        result = yield _TaskWaitRequest(self)
        return result

    def start(self, delay: float = 0, key: Any = None) -> 'SchedulerTask':
        return self.scheduler._start_task(self, delay, key)

    def cancel(self):
        if self.done:
            return
        self.cancelled = True
        self.exception = SchedulerTaskCancelled("Scheduler task was cancelled")
        self.scheduler._complete_task(self)

    def close(self):
        if self.done:
            return
        self.done = True
        try:
            self.coroutine.close()
        except RuntimeError:
            pass

    def result(self):
        if not self.done:
            raise RuntimeError("Scheduler task has not finished yet")
        if self.exception is not None:
            raise self.exception
        return self.result_value


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
                if task._throw_exception is not None:
                    exception = task._throw_exception
                    task._throw_exception = None
                    request = task.coroutine.throw(exception)
                else:
                    send_value = task._send_value
                    task._send_value = None
                    request = task.coroutine.send(send_value)
                self._schedule_next_step(task, request)
            except StopIteration as e:
                task.result_value = e.value
                self._remove_task(task)
            except (KeyboardInterrupt, SystemExit, NewGame) as e:
                raise e
            except Exception as e:
                task.exception = e
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

        return self.prepare_task(coroutine).start(delay=delay, key=key)

    def prepare_task(
            self,
            coroutine: Coroutine[Any, Any, Any],
            key: Any = None,
    ) -> SchedulerTask:
        if not inspect.iscoroutine(coroutine):
            raise TypeError("Scheduler.prepare_task expects an async coroutine object")

        task = SchedulerTask(
            scheduler=self,
            coroutine=coroutine,
            key=key,
        )

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

    def _start_task(
            self,
            task: SchedulerTask,
            delay: float = 0,
            key: Any = None,
    ) -> SchedulerTask:
        with self._lock:
            if task.done:
                raise RuntimeError("Cannot start a finished SchedulerTask")
            if task.started:
                return task

            if key is not None:
                task.key = key

            if task.key is not None:
                self.cancel(task.key)
                self._tasks_by_key[task.key] = task

            task.started = True
            task.ready_time = self.game.run_time + max(delay, 0)
            task.ready_frame = self._frame
            self._pending_tasks.append(task)

        return task

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

        if isinstance(request, _TaskWaitRequest):
            awaited = request.task
            if awaited.done:
                self._resume_waiter_from_task(task, awaited)
            else:
                awaited._waiters.append(task)
                task.ready_time = float("inf")
                task.ready_frame = self._frame
            return

        raise TypeError(
            "Scheduler tasks can only await scheduler.sleep(...) "
            "or scheduler.next_frame(), or another SchedulerTask"
        )

    def _remove_task(self, task: SchedulerTask):
        if task in self._tasks:
            self._tasks.remove(task)
        if task.key is not None and self._tasks_by_key.get(task.key) is task:
            self._tasks_by_key.pop(task.key, None)
        self._complete_task(task)

    def _complete_task(self, task: SchedulerTask):
        task.close()
        for waiter in list(task._waiters):
            self._resume_waiter_from_task(waiter, task)
        task._waiters.clear()

    def _resume_waiter_from_task(self, waiter: SchedulerTask, awaited: SchedulerTask):
        if waiter.done or waiter.cancelled:
            return

        if awaited.exception is not None:
            waiter._throw_exception = awaited.exception
        else:
            waiter._send_value = awaited.result_value

        waiter.ready_time = self.game.run_time
        waiter.ready_frame = self._frame + 1


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
