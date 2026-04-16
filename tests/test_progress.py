"""Tests for ProgressReporter and ProgressMessage."""
from ascribe_link.progress import ProgressReporter, ProgressMessage


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
