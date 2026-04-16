"""Tests for ProgressReporter and ProgressMessage."""
from ascribe_link.progress import ProgressReporter, ProgressMessage, JobReporter
from ascribe_link.job_registry import Job


def test_progress_message_is_frozen_dataclass():
    msg = ProgressMessage(seq=0, text="hello", ts=1.0)
    assert msg.seq == 0
    assert msg.text == "hello"
    assert msg.ts == 1.0


def test_progress_reporter_noop_report_does_not_raise():
    reporter = ProgressReporter()
    # Should silently succeed with no job bound
    reporter.report("anything")
    reporter.report("")


def test_progress_reporter_is_base_class():
    # JobReporter will inherit from this
    assert hasattr(ProgressReporter, "report")


def _make_job() -> Job:
    return Job(id="j1", specimen_id="sphere", params={}, room_id="ascribe")


def test_job_reporter_appends_to_job():
    job = _make_job()
    reporter = JobReporter(job)
    reporter.report("hello")
    reporter.report("world")
    assert [m.text for m in job.messages] == ["hello", "world"]
    assert job.next_seq == 2


def test_job_reporter_is_progress_reporter():
    job = _make_job()
    reporter = JobReporter(job)
    assert isinstance(reporter, ProgressReporter)
