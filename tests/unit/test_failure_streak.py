"""Pausing the queue after consecutive failed generations.

Without it a backend that refuses every connection consumes the whole queue and
staging, one failed run every few seconds, because a generation's failure is
logged on its pool worker and never reaches the run path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib.translations import _
from teikr.runs import run_controller as rc
from teikr.runs.failure_streak import PAUSE_AFTER_FAILURES, FailureStreak, failure_streak
from teikr.runs.job_queue import SDRunsQueue, ServerStagingQueue
from teikr.runs.run_controller import RunController


class DirectBridge:
    def invoke(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@pytest.fixture
def started(monkeypatch):
    """Threads the controller would start, recorded instead of run."""
    calls = []
    monkeypatch.setattr(rc.Utils, "start_thread",
                        lambda target, use_asyncio=True, args=None: calls.append((target, args or [])))
    monkeypatch.setattr(rc.Utils, "prevent_sleep", lambda enable: None)
    return calls


def make_controller(pending=0, staged=0, sidebar=False):
    app = SimpleNamespace(
        job_queue=SDRunsQueue(),
        server_staging_queue=ServerStagingQueue(),
        _thread_bridge=DirectBridge(),
        app_actions=MagicMock(),
        notification_ctrl=MagicMock(),
    )
    if sidebar:
        app.sidebar_panel = MagicMock()
    for i in range(pending):
        app.job_queue.add(f"job-{i}")
    for i in range(staged):
        app.server_staging_queue.add("cmd", {"i": i}, "client", f"staged-{i}")
    app.job_queue.job_running = True
    return RunController(app), app


def fail(times):
    for _i in range(times):
        failure_streak.record_failure(ConnectionRefusedError("[WinError 10061] refused"))


def targets(calls):
    return [getattr(target, "__name__", target) for target, _args in calls]


class TestFailureStreak:
    def test_counts_failures_and_keeps_the_last_error(self):
        streak = FailureStreak()
        streak.record_failure(ValueError("first"))
        streak.record_failure(ValueError("second"))
        assert streak.snapshot() == (2, "second")

    def test_a_success_resets_it(self):
        streak = FailureStreak()
        streak.record_failure(ValueError("x"))
        streak.record_success()
        assert streak.snapshot() == (0, None)


class TestWrapTaskRecordsOutcomes:
    @staticmethod
    def wrapped(task_fn):
        from teikr.generators.base import BaseImageGenerator
        fake = SimpleNamespace(
            _arm_pending_release=lambda: None,
            _timing_context=SimpleNamespace(),
            _timing_descriptors=lambda kwargs: None,
            gen_config=SimpleNamespace(),
            _record_recent_adapters=lambda *a, **k: None,
            _handle_error=lambda error, name: None,
            release_pending=lambda: None,
        )
        return BaseImageGenerator._wrap_task(fake, task_fn)

    def test_a_failed_task_is_counted_and_still_raises(self):
        def image_edit():
            raise ConnectionRefusedError("refused")
        with pytest.raises(ConnectionRefusedError):
            self.wrapped(image_edit)()
        assert failure_streak.snapshot() == (1, "refused")

    def test_a_successful_task_resets_the_count(self):
        fail(2)
        assert self.wrapped(lambda: "done")() == "done"
        assert failure_streak.snapshot()[0] == 0


class TestPostRunPausesOnFailures:
    def test_below_the_limit_staging_keeps_draining(self, started):
        ctrl, app = make_controller(staged=5)
        fail(PAUSE_AFTER_FAILURES - 1)
        ctrl._post_run()
        assert app.job_queue.paused is False
        assert app.server_staging_queue.pending_count() == 4
        assert targets(started) == ["_run_promoted_async"]

    def test_at_the_limit_staged_requests_stay_staged(self, started):
        ctrl, app = make_controller(staged=5)
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        assert app.job_queue.paused is True
        assert app.server_staging_queue.pending_count() == 5
        assert "_run_promoted_async" not in targets(started)

    def test_at_the_limit_the_next_job_goes_back_to_the_front(self, started):
        ctrl, app = make_controller(pending=3)
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        assert app.job_queue.paused is True
        assert app.job_queue.pending_jobs == ["job-0", "job-1", "job-2"]
        assert "_run_async" not in targets(started)

    def test_the_user_is_alerted_from_a_worker(self, started):
        ctrl, app = make_controller(pending=1, staged=2)
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        app.notification_ctrl.alert.assert_not_called()
        [(target, args)] = [c for c in started if getattr(c[0], "__name__", "") == "_alert_queue_paused"]
        target(*args)
        title, message = app.notification_ctrl.alert.call_args.args
        assert title == _("Queue Paused")
        assert app.notification_ctrl.alert.call_args.kwargs == {"kind": "error"}
        assert "10061" in message

    def test_nothing_waiting_means_nothing_to_pause(self, started):
        ctrl, app = make_controller()
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        assert app.job_queue.paused is False
        assert started == []

    def test_a_success_in_between_prevents_the_pause(self, started):
        ctrl, app = make_controller(staged=2)
        fail(PAUSE_AFTER_FAILURES - 1)
        failure_streak.record_success()
        fail(1)
        ctrl._post_run()
        assert app.job_queue.paused is False

    def test_the_sidebar_button_says_resume_when_there_is_one(self, started):
        ctrl, app = make_controller(staged=1, sidebar=True)
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        app.sidebar_panel.pause_queue_btn.setText.assert_called_with(_("Resume Queue"))

    def test_an_already_paused_queue_is_neither_alerted_nor_drained(self, started):
        ctrl, app = make_controller(staged=3)
        app.job_queue.paused = True
        fail(PAUSE_AFTER_FAILURES)
        ctrl._post_run()
        assert app.server_staging_queue.pending_count() == 3
        assert started == []


class TestResumeResetsTheCount:
    def test_resume_paused_queue(self, started):
        ctrl, app = make_controller(pending=2)
        app.job_queue.job_running = False
        app.job_queue.paused = True
        fail(PAUSE_AFTER_FAILURES)
        ctrl.resume_paused_queue()
        assert failure_streak.snapshot()[0] == 0
        assert app.job_queue.paused is False
        assert targets(started) == ["_run_async"]

    def test_unpausing_with_the_button(self, started):
        ctrl, app = make_controller(sidebar=True)
        app.job_queue.paused = True
        fail(PAUSE_AFTER_FAILURES)
        ctrl.toggle_pause_queue()
        assert failure_streak.snapshot()[0] == 0

    def test_pausing_with_the_button_keeps_the_count(self, started):
        ctrl, app = make_controller(sidebar=True)
        fail(2)
        ctrl.toggle_pause_queue()
        assert failure_streak.snapshot()[0] == 2
