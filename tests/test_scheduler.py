import unittest
from types import SimpleNamespace
from unittest.mock import patch

from EasyCells3D.scheduler import Scheduler, SchedulerFuture, SchedulerTaskCancelled


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        singleton = patch.object(Scheduler, 'instance', None)
        singleton.start()
        self.addCleanup(singleton.stop)
        self.game = SimpleNamespace(run_time=0)
        self.scheduler = Scheduler(self.game)
        self.addCleanup(self.scheduler.clear)

    def test_external_result_resumes_multiple_waiters_once_on_next_update(self):
        future = SchedulerFuture(self.scheduler)
        received = []

        async def consume():
            received.append(await future)

        tasks = [self.scheduler.create_task(consume()) for _ in range(2)]
        self.scheduler.update()
        future.set_result(42)
        future.set_result(99)
        self.assertEqual(received, [])
        self.scheduler.update()
        self.assertEqual(received, [42, 42])
        self.assertTrue(all(task.done for task in tasks))
        self.assertFalse(self.scheduler._futures)

    def test_await_local_task_still_preserves_sleep_and_return_value(self):
        async def child():
            await self.scheduler.sleep(2)
            return 7

        child_task = self.scheduler.create_task(child())

        async def parent():
            return 1 + await child_task

        task = self.scheduler.create_task(parent())
        self.scheduler.update()
        self.game.run_time = 1
        self.scheduler.update()
        self.assertFalse(task.done)
        self.game.run_time = 2
        self.scheduler.update()
        self.assertFalse(task.done)
        self.scheduler.update()
        self.assertEqual(task.result(), 8)

    def test_cancel_future_throws_into_waiter_and_cancelled_waiter_detaches(self):
        future = SchedulerFuture(self.scheduler)
        received = []

        async def consume():
            try:
                await future
            except SchedulerTaskCancelled:
                received.append('cancelled')

        cancelled = self.scheduler.create_task(consume())
        active = self.scheduler.create_task(consume())
        self.scheduler.update()
        cancelled.cancel()
        self.assertEqual(future._waiters, [active])
        self.scheduler.cancel(future)
        self.scheduler.update()
        self.assertEqual(received, ['cancelled'])
        self.assertIsNone(active.exception)

    def test_clear_cancels_unstarted_tasks_and_futures_and_notifies_once(self):
        callbacks = []

        async def run():
            self.fail('A cancelled pending task must not start')

        task = self.scheduler.create_task(run())
        future = SchedulerFuture(self.scheduler)
        for result in (task, future):
            result.add_done_callback(lambda result: callbacks.append(result))
        self.scheduler.clear()
        self.scheduler.clear()
        self.scheduler.update()
        self.assertEqual(callbacks, [task, future])
        self.assertFalse(self.scheduler._futures)
        self.assertTrue(task.cancelled and future.cancelled)

    def test_cancellation_during_execution_closes_coroutine_after_it_yields(self):
        events = []

        async def run():
            try:
                self.scheduler.clear()
                await self.scheduler.next_frame()
                events.append('resumed')
            finally:
                events.append('closed')

        task = self.scheduler.create_task(run())
        self.scheduler.update()
        self.scheduler.update()
        self.assertEqual(events, ['closed'])
        self.assertTrue(task.cancelled)
        self.assertIsInstance(task.exception, SchedulerTaskCancelled)


if __name__ == '__main__':
    unittest.main()
