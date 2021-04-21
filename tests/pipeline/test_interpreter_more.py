"""More test cases for ambianic.interpreter module."""
from ambianic import pipeline
from ambianic.pipeline import interpreter
from ambianic.pipeline.avsource.av_element import AVSourceElement
from ambianic.pipeline.interpreter import \
    PipelineServer, Pipeline, HealingThread, PipelineServerJob
import logging
import time
import threading


log = logging.getLogger()
log.setLevel(logging.DEBUG)


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    # Reset default class
    interpreter.PIPELINE_CLASS = None
    interpreter.Pipeline.PIPELINE_OPS['source'] = AVSourceElement


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
     method."""
    # Reset default class
    interpreter.PIPELINE_CLASS = None
    interpreter.Pipeline.PIPELINE_OPS['source'] = AVSourceElement


class _TestSourceElement(pipeline.PipeElement):
    """Produce one sample and exit start loop."""

    def __init__(self, **element_config):
        super().__init__()
        self.config = element_config
        self.start_called = False
        self.stop_called = False

    def start(self):
        self.start_called = True
        super().start()
        # send one sample down the pipe
        self.receive_next_sample(sample=[1, 2, 3])

    def stop(self):
        self.stop_called = True
        super().stop()


def _get_config(source_class=None):
    # override source op with a mock test class
    Pipeline.PIPELINE_OPS['source'] = source_class
    server_config = {
        'pipelines': {
            'pipeline_one': [
                {'source': {'uri': 'test'}}
            ]
        },
    }
    return server_config


def test_pipeline_server_init():
    conf = _get_config(_TestSourceElement)
    server = PipelineServerJob(conf)
    assert len(server._pipelines) == 1
    assert len(server._threaded_jobs) == 1


def _get_config_invalid_element(source_class=None):
    # override source op with a mock test class
    Pipeline.PIPELINE_OPS['source'] = source_class
    pipeline_config = [
        {'source': {'uri': 'test'}},
        {'scifi': {'one': 'day soon'}},
    ]
    return pipeline_config


class _TestPipeline(Pipeline):

    def __init__(self, pname=None, pconfig=None):
        self._test_on_unknown_pipe_element_called = False
        self._test_on_healing_already_in_progress_called = False
        self._test_on_start_no_elements_called = False
        super().__init__(pname=pname, pconfig=pconfig)

    def _on_unknown_pipe_element(self, name=None):
        self._test_on_unknown_pipe_element_called = True
        log.debug('_on_unknown_pipe_element called')
        super()._on_unknown_pipe_element(name=name)

    def _on_healing_already_in_progress(self):
        self._test_on_healing_already_in_progress_called = True
        super()._on_healing_already_in_progress()

    def _on_start_no_elements(self):
        self._test_on_start_no_elements_called = True
        super()._on_start_no_elements()


def test_pipeline_init_invalid_element():
    conf = _get_config_invalid_element(_TestSourceElement)
    pipeline = _TestPipeline(pname='test', pconfig=conf)
    assert pipeline._test_on_unknown_pipe_element_called
    assert len(pipeline._pipe_elements) == 1
    assert isinstance(pipeline._pipe_elements[0], _TestSourceElement)


class _TestSourceElement2(pipeline.PipeElement):
    """Produce samples until stop signal."""

    def __init__(self, **element_config):
        super().__init__()
        self.config = element_config
        self._test_element_started = threading.Event()

    def start(self):
        super().start()
        self._test_element_started.set()
        # generate samples until stopped
        while self.state == pipeline.PIPE_STATE_RUNNING:
            self.receive_next_sample(sample=[1, 2, 3])


def test_pipeline_server_start_stop():
    conf = _get_config(_TestSourceElement2)
    server = PipelineServerJob(conf)
    assert len(server._pipelines) == 1
    assert len(server._threaded_jobs) == 1
    source_pe = server._pipelines[0]._pipe_elements[0]
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()
    server.start()
    source_pe._test_element_started.wait(timeout=3)
    assert source_pe.state == pipeline.PIPE_STATE_RUNNING
    assert server._threaded_jobs[0].is_alive()
    server.stop()
    # give it enough time to clean up resources
    # in child threads (if any).
    time.sleep(3)
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()


def test_pipeline_server_config_change():
    conf = _get_config(_TestSourceElement2)
    PipelineServer(conf)

    del conf['pipelines']["pipeline_one"][0]


class _TestSourceElement3(pipeline.PipeElement):
    """Produce samples until stop signal."""

    def __init__(self, **element_config):
        super().__init__()
        self._test_heal_called = threading.Event()
        self._test_sample_released = threading.Event()
        log.debug('heal() not called yet')

    def heal(self):
        self._test_heal_called.set()
        log.debug('heal() called')

    def start(self):
        super().start()
        # send one samples until stopped
        while self.state == pipeline.PIPE_STATE_RUNNING:
            self.receive_next_sample(sample=[1, 2, 3])
            # artifitial delay to force heal()
            log.debug('delaying next sample to cause heal()')
            time.sleep(2)
            self._test_sample_released.set()
            time.sleep(2)


def test_pipeline_server_heal():
    conf = _get_config(_TestSourceElement3)
    server = PipelineServerJob(conf)
    assert len(server._pipelines) == 1
    assert len(server._threaded_jobs) == 1
    source_pe = server._pipelines[0]._pipe_elements[0]
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()
    server.MAX_HEARTBEAT_INTERVAL = 1
    server.start()
    source_pe._test_sample_released.wait(timeout=5)
    assert source_pe.state == pipeline.PIPE_STATE_RUNNING
    assert server._threaded_jobs[0].is_alive()
    server.healthcheck()
    assert source_pe._test_heal_called.wait(timeout=5)
    server.stop()
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()


class _TestPipelineServer2(PipelineServerJob):

    def __init__(self, config=None):
        super().__init__(config=config)
        self._test_on_terminal_health_called = threading.Event()

    def _on_terminal_pipeline_health(self, pipeline=None, lapse=None):
        log.debug('_on_terminal_pipeline_health called')
        super()._on_terminal_pipeline_health(pipeline, lapse)
        self._test_on_terminal_health_called.set()


def test_pipeline_terminal_health():
    conf = _get_config(_TestSourceElement3)
    server = _TestPipelineServer2(conf)
    assert len(server._pipelines) == 1
    assert len(server._threaded_jobs) == 1
    source_pe = server._pipelines[0]._pipe_elements[0]
    server.TERMINAL_HEALTH_INTERVAL = 1
    server.start()
    source_pe._test_sample_released.wait(timeout=5)
    server.healthcheck()
    assert server._test_on_terminal_health_called.wait(timeout=5)
    server.stop()
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()


class _TestDummyElement(pipeline.PipeElement):
    """Dummy pass through element."""

    def __init__(self, **element_config):
        super().__init__()
        self._config = element_config
        self._test_heal_called = False

    _sample_processed = False

    def process_sample(self, sample=None):
        assert sample == [1, 2, 3]
        self._sample_processed = True
        yield {'sample': sample}


def _get_pipeline_config_2_elements():
    # override source op with a mock test class
    Pipeline.PIPELINE_OPS['source'] = _TestSourceElement
    Pipeline.PIPELINE_OPS['dummy'] = _TestDummyElement
    pipeline_config = [
        {'source': {'uri': 'test'}},
        {'dummy': {'dummy': 'config'}}
        ]
    return pipeline_config


def test_pipeline_start2():
    conf = _get_pipeline_config_2_elements()
    pipeline = _TestPipeline(pname='test', pconfig=conf)
    assert len(pipeline._pipe_elements) == 2
    assert isinstance(pipeline._pipe_elements[0], _TestSourceElement)
    assert isinstance(pipeline._pipe_elements[1], _TestDummyElement)
    pipeline.start()
    dummy = pipeline._pipe_elements[1]
    assert dummy._sample_processed
    pipeline.stop()


class _TestSourceElement4(pipeline.PipeElement):
    """Produce samples until stop signal."""

    def __init__(self, **element_config):
        super().__init__()
        self.config = element_config

    def start(self):
        super().start()
        self.receive_next_sample(sample=[1, 2, 3])

    def heal(self):
        # delay to test 2xheal()
        time.sleep(2)


def test_pipeline_heal2():
    Pipeline.PIPELINE_OPS['source'] = _TestSourceElement4
    pipeline_config = [
        {'source': {'uri': 'test'}},
        ]
    pipeline = _TestPipeline(pname='test', pconfig=pipeline_config)
    assert len(pipeline._pipe_elements) == 1
    assert isinstance(pipeline._pipe_elements[0], _TestSourceElement4)
    pipeline.start()
    pipeline.heal()
    assert not pipeline._test_on_healing_already_in_progress_called
    pipeline.heal()
    assert pipeline._test_on_healing_already_in_progress_called
    pipeline.stop()


def test_pipeline_start_no_elements():
    Pipeline.PIPELINE_OPS['source'] = _TestSourceElement4
    pipeline_config = [{'source': "unavailable"}]
    pipeline = _TestPipeline(pname='test', pconfig=pipeline_config)
    assert len(pipeline._pipe_elements) == 0
    pipeline.start()
    assert pipeline._test_on_start_no_elements_called


def test_healing_thread():
    _target_called = False

    def target():
        nonlocal _target_called
        _target_called = True
        raise RuntimeError()

    _on_finished_called = False

    def on_finished():
        nonlocal _on_finished_called
        _on_finished_called = True
        raise RuntimeError()

    healer = HealingThread(target=target, on_finished=on_finished)
    healer.run()
    assert _target_called
    assert _on_finished_called


class _TestPipelineServer5(PipelineServerJob):

    _test_on_threaded_job_ended_called = False

    def _on_pipeline_job_ended(self, threaded_job=None):
        self._test_on_threaded_job_ended_called = True
        super()._on_pipeline_job_ended(threaded_job=threaded_job)


class _TestSourceElement5(pipeline.PipeElement):
    """Produce one sample and exit start loop."""

    def __init__(self, **element_config):
        super().__init__()
        self.config = element_config

    def start(self):
        super().start()
        # send one sample down the pipe
        self.receive_next_sample(sample=[1, 2, 3])
        super().stop()


def test_on_pipeline_job_ended():
    conf = _get_config(_TestSourceElement5)
    server = _TestPipelineServer5(conf)
    assert len(server._pipelines) == 1
    assert len(server._threaded_jobs) == 1
    source_pe = server._pipelines[0]._pipe_elements[0]
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs[0].is_alive()
    server.start()
    # give time to pipeline job to exit
    time.sleep(2)
    server.healthcheck()
    assert source_pe.state == pipeline.PIPE_STATE_STOPPED
    assert not server._threaded_jobs
