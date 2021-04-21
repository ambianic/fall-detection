import pytest
from ambianic import pipeline


def test_init():
    pe = pipeline.PipeElement()
    assert pe.state == pipeline.PIPE_STATE_STOPPED


def test_start():
    pe = pipeline.PipeElement()
    pe.start()
    assert pe.state == pipeline.PIPE_STATE_RUNNING


def test_stop():
    pe = pipeline.PipeElement()
    pe.stop()
    assert pe.state == pipeline.PIPE_STATE_STOPPED


def test_connect_to_next_element():
    pe = pipeline.PipeElement()
    pe_next = pipeline.PipeElement()
    pe.connect_to_next_element(next_element=pe_next)
    assert pe._next_element == pe_next


def test_connect_to_bad_next_element():
    pe = pipeline.PipeElement()
    # try linking to the wrong type of object
    pe_next = []
    with pytest.raises(AssertionError):
        pe.connect_to_next_element(next_element=pe_next)


def test_healthcheck():
    pe = pipeline.PipeElement()
    hb = pe._latest_heartbeat
    hb1, status = pe.healthcheck()
    assert hb == hb1
