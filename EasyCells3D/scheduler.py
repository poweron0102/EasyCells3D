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
    task: 'SchedulerTask | SchedulerFuture'

    def __await__(self):
        result = yield self
        return result


class SchedulerTaskCancelled(Exception):
    pass


class SchedulerFuture:
    """A result completed by an external producer on the game's thread."""

    def __init__(self, scheduler: 'Scheduler'):
        self.scheduler = scheduler
        self.done = False
        self.cancelled = False
        self.result_value = None
        self.exception = None
        self._waiters = []
        self._done_callbacks = []
        scheduler._futures.add(self)

    def __await__(self):
        result = yield _TaskWaitRequest(self)
        return result

    def result(self):
        if not self.done:
            raise RuntimeError("Scheduler result has not finished yet")
        if self.exception is not None:
            raise self.exception
        return self.result_value

    def set_result(self, value):
        if not self.done:
            self.result_value = value
            self._finish()

    def set_exception(self, exception):
        if not self.done:
            self.exception = exception
            self._finish()

    def cancel(self):
        if not self.done:
            self.cancelled = True
            self.set_exception(SchedulerTaskCancelled("Scheduler future was cancelled"))

    def add_done_callback(self, callback):
        if self.done:
            callback(self)
        else:
            self._done_callbacks.append(callback)

    def _finish(self):
        self.done = True
        self.scheduler._futures.discard(self)
        self._notify_done()

    def _notify_done(self):
        for waiter in self._waiters:
            self.scheduler._resume_waiter_from_task(waiter, self)
        self._waiters.clear()
        callbacks, self._done_callbacks = self._done_callbacks, []
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                traceback.print_exc()


class SchedulerTask(SchedulerFuture):
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
        self._waiting_on: SchedulerFuture | None = None
        self._done_callbacks = []

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
        if self.coroutine.cr_running:
            # A handler can destroy its own component or change scenes. Close it
            # after send()/throw() returns, rather than reentering the coroutine.
            return
        try:
            self.coroutine.close()
        except Exception:
            traceback.print_exc()

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
        self._futures: set[SchedulerFuture] = set()

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
                if task.done:
                    task.coroutine.close()
                else:
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

    def cancel(self, task_or_key: SchedulerFuture | Any):
        with self._lock:
            if isinstance(task_or_key, SchedulerFuture):
                task = task_or_key
            else:
                task = self._tasks_by_key.get(task_or_key)

            if task is None:
                return

            task.cancel()
            key = getattr(task, "key", None)
            if key is not None and self._tasks_by_key.get(key) is task:
                self._tasks_by_key.pop(key, None)

    def sleep(self, seconds: float) -> _SleepRequest:
        return _SleepRequest(max(seconds, 0))

    def next_frame(self) -> _NextFrameRequest:
        return _NextFrameRequest()

    def clear(self):
        with self._lock:
            for task in list(self._tasks):
                task.cancel()
            for task in list(self._pending_tasks):
                task.cancel()
            for future in list(self._futures):
                future.cancel()
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
            if awaited.scheduler is not self:
                raise ValueError("Cannot await a result from another Scheduler")
            if awaited is task:
                raise ValueError("A task cannot await itself")
            if awaited.done:
                self._resume_waiter_from_task(task, awaited)
            else:
                awaited._waiters.append(task)
                task._waiting_on = awaited
                task.ready_time = float("inf")
                task.ready_frame = self._frame
            return

        raise TypeError(
            "Scheduler tasks can only await scheduler.sleep(...) "
            "or scheduler.next_frame(), or a SchedulerTask/SchedulerFuture"
        )

    def _remove_task(self, task: SchedulerTask):
        if task in self._tasks:
            self._tasks.remove(task)
        if task.key is not None and self._tasks_by_key.get(task.key) is task:
            self._tasks_by_key.pop(task.key, None)
        self._complete_task(task)

    def _complete_task(self, task: SchedulerTask):
        if task._waiting_on is not None:
            awaited, task._waiting_on = task._waiting_on, None
            if task in awaited._waiters:
                awaited._waiters.remove(task)
        task.close()
        task._notify_done()

    def _resume_waiter_from_task(self, waiter: SchedulerTask, awaited: SchedulerFuture):
        if waiter.done or waiter.cancelled:
            return

        waiter._waiting_on = None
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
