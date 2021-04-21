import pytest
from ambianic import pipeline


def test_init():
    def hc_callback():
        pass
    hc = pipeline.HealthChecker(health_status_callback=hc_callback)
    assert hc._health_status_callback == hc_callback


def test_receive_next_sample():
    called = False

    def hc_callback():
        nonlocal called
        called = True
    hc = pipeline.HealthChecker(health_status_callback=hc_callback)
    hc.receive_next_sample()
    assert called
