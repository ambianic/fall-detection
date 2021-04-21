import pytest
import ambianic
from ambianic import __main__, config, load_config
from ambianic.server import AmbianicServer
from ambianic.util import ServiceExit, ManagedService
from ambianic.pipeline.interpreter import PipelineServer
import os
import threading
import signal
import time
from pathlib import Path

def test_no_work_dir():
    with pytest.raises(AssertionError):
        srv = AmbianicServer(work_dir=None)


def test_bad_work_dir():
    srv = AmbianicServer(work_dir='/_/_/_dir_does_not_exist___')
    with pytest.raises(AssertionError):
        srv.start()


class MockAmbianicServer(AmbianicServer):

    def __init__(self, work_dir=None, heartbeat_flag=None):
        super().__init__(work_dir)
        self._heartbeat_flag = heartbeat_flag
        self._main_heartbeat_logged = False
        self.config_changed = False

    def _heartbeat(self):
        super()._heartbeat()
        if self._heartbeat_flag:
            self._heartbeat_flag.set()

    def _log_heartbeat(self):
        super()._log_heartbeat()
        self._main_heartbeat_logged = True

    def dispatch(self, event):
        super().dispatch(event)
        self.config_changed = True


def _start_mock_server(**kwargs):
    srv = MockAmbianicServer(**kwargs)
    t = threading.Thread(
        target=srv.start,
        daemon=True)
    t.start()
    return (srv, t)


def _stop_mock_server(server=None, thread=None):
    assert server
    assert thread
    server.stop()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_no_pipelines():
    _dir = os.path.dirname(os.path.abspath(__file__))
    config.clean()
    config.load_file(path=os.path.join(_dir, 'test-config-no-pipelines.yaml'))
    hb_flag = threading.Event()
    srv, t = _start_mock_server(work_dir=_dir, heartbeat_flag=hb_flag)
    hb_flag.wait(timeout=3)
    assert hb_flag.is_set()
    pps = srv._servers['pipelines']
    assert isinstance(pps, ambianic.pipeline.interpreter.PipelineServer)
    assert not pps.pipeline_server_job.job._pipelines
    _stop_mock_server(server=srv, thread=t)


def test_main():

    _dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['AMBIANIC_DIR'] = _dir

    config.clean()
    config.load_file(path=os.path.join(_dir, 'test-config-no-pipelines.yaml'))    

    t = threading.Thread(
        target=__main__.main,
        daemon=True)
    t.start()
    t.join(timeout=1)
    __main__.stop()
    t.join(timeout=3)
    assert not t.is_alive()


def test_system_shutdown_signal():
    with pytest.raises(ServiceExit):
        __main__._service_shutdown(signum=signal.SIGINT, frame=None)


class _BadPipelineServer(ManagedService):

    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self._heal_called = False

    def healthcheck(self):
        super().healthcheck()
        # return an old enough heartbeat time to trigger a health concern
        latest_heartbeat = time.monotonic() - \
            ambianic.server.MANAGED_SERVICE_HEARTBEAT_THRESHOLD - 10
        print('_BadPipelineServer latest_heartbeat - now: {}'.
              format(latest_heartbeat))
        return latest_heartbeat, "BAD"

    def heal(self):
        super().heal()
        self._heal_called = True


def test_heartbeat_threshold():
    _dir = os.path.dirname(os.path.abspath(__file__))
    config.load_file(path=os.path.join(_dir, 'test-config-no-pipelines.yaml'))
    # replace default with test pipeline server
    # remove all root servers which we won't test here
    ambianic.server.ROOT_SERVERS.clear()
    ambianic.server.ROOT_SERVERS['pipelines'] = _BadPipelineServer
    srv, t = _start_mock_server(work_dir=_dir)
    t.join(timeout=2)
    pps = srv._servers['pipelines']
    assert isinstance(pps, _BadPipelineServer)
    assert pps._heal_called
    _stop_mock_server(server=srv, thread=t)


def test_main_heartbeat_log():
    _dir = os.path.dirname(os.path.abspath(__file__))
    load_config(os.path.join(_dir, 'test-config-no-pipelines.yaml'), True)
    # remove all root servers which we will not test here
    ambianic.server.ROOT_SERVERS.clear()
    # set heartbeat log interval to a small enough
    # interval so the test passes faster
    ambianic.server.MAIN_HEARTBEAT_LOG_INTERVAL = 0.1
    srv, t = _start_mock_server(work_dir=_dir)
    t.join(timeout=2)
    assert srv._main_heartbeat_logged
    _stop_mock_server(server=srv, thread=t)

def test_config_change():
    _dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(_dir, 'test-config-no-pipelines.yaml')
    load_config(config_file, True)
    hb_flag = threading.Event()
    srv, t = _start_mock_server(work_dir=_dir, heartbeat_flag=hb_flag)
    hb_flag.wait(timeout=3)
    Path(config_file).touch()
    time.sleep(3)
    assert hb_flag.is_set()
    assert srv.config_changed
    _stop_mock_server(server=srv, thread=t)