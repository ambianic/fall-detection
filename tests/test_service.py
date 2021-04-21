import pytest
from ambianic.util import ServiceExit, ManagedService, ThreadedJob


class MockManagedService(ManagedService):

    def __init__(self):
        super().__init__()
        self._healthcheck_called = False
        self._heal_called = False

    def healthcheck(self):
        super().healthcheck()
        self._healthcheck_called = True

    def heal(self):
        super().heal()
        self._heal_called = True


def test_threaded_job_init_no_job():
    with pytest.raises(AssertionError):
        tj = ThreadedJob(job=None)


def test_threaded_job_init_no_ms():
    with pytest.raises(AssertionError):
        tj = ThreadedJob(job=[])


def test_threaded_job_init_ms():
    ms = MockManagedService()
    tj = ThreadedJob(job=ms)
    assert tj.job == ms


def test_healthcheck():
    ms = MockManagedService()
    tj = ThreadedJob(job=ms)
    tj.healthcheck()
    assert ms._healthcheck_called


def test_heal():
    ms = MockManagedService()
    tj = ThreadedJob(job=ms)
    tj.heal()
    assert ms._heal_called
