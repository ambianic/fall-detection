import logging
import os
from ambianic import pipeline, config
from ambianic.pipeline import interpreter
from ambianic.pipeline.avsource.av_element import AVSourceElement

from dynaconf.utils import DynaconfDict
# mocked_settings = DynaconfDict({'FOO': 'BAR'})

log = logging.getLogger(__name__)

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    # Reset default class
    interpreter.PIPELINE_CLASS = None
    interpreter.Pipeline.PIPELINE_OPS['source'] = AVSourceElement
    _TestPipeline.PIPELINE_OPS['source'] = AVSourceElement
    config.reload()


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
     method."""
    # Reset default class
    interpreter.PIPELINE_CLASS = None
    interpreter.Pipeline.PIPELINE_OPS['source'] = AVSourceElement
    _TestPipeline.PIPELINE_OPS['source'] = AVSourceElement
    config.reload()


class _TestPipeline(interpreter.Pipeline):
    """Extend Pipeline to detect changes"""

    def __init__(self, pname=None, pconfig=None, data_dir=None):
        super().__init__(pname, pconfig, data_dir)
        self.on_change_called = False
        self.restart_called = False
        log.debug("_TestPipeline.__init__")

    def restart(self):
        super().restart()
        self.restart_called = True
        log.debug("_TestPipeline.restart")


class _TestSourceElement(pipeline.PipeElement):

    def __init__(self, **element_config):
        super().__init__()
        self.config = element_config
        self.start_called = False
        self.stop_called = False
        self.on_config_change_called = False
        log.debug("_TestSourceElement.__init__")

    def start(self):
        super().start()
        self.start_called = True
        # send one sample down the pipe
        self.receive_next_sample([1, 2, 3])
        log.debug("_TestSourceElement.start")

    def heal(self):
        """Empty implementation of abstractmethod."""

    def stop(self):
        super().stop()
        self.stop_called = True

    def receive_next_sample(self, *sample):
        """Empty implementation of abstractmethod."""


def _get_pipelines_config():
    return {
        'pipeline_one': [
            {'source': {'uri': 'test'}}
        ]
    }


def _one_pipeline_setup(pipelines_config=None, set_source_el=True):
    if pipelines_config is None:
        pipelines_config = _get_pipelines_config()
    # override source op with a mock test class
    if set_source_el:
        log.info("set source=_TestSourceElement")
        interpreter.Pipeline.PIPELINE_OPS['source'] = _TestSourceElement
    return interpreter.get_pipelines(
        pipelines_config=DynaconfDict(pipelines_config)
    )


def test_get_pipelines_none():
    p = interpreter.get_pipelines(pipelines_config=[])
    assert not p


def test_derived_pipe_element():
    derived_element = _TestSourceElement(element_config='something')
    assert derived_element.state == pipeline.PIPE_STATE_STOPPED


def test_get_pipelines_one():
    p = _one_pipeline_setup()
    print('p[0]: {p0}'.format(p0=p[0]))
    assert isinstance(p[0], interpreter.Pipeline)
    assert p[0].name == 'pipeline_one'
    assert isinstance(p[0]._pipe_elements[0], _TestSourceElement)
    assert p[0]._pipe_elements[0].config['uri'] == 'test'


def test_get_pipelines_two():
    # override source op with a mock test class
    interpreter.Pipeline.PIPELINE_OPS['source'] = _TestSourceElement
    p = interpreter.get_pipelines(
        pipelines_config={
            'pipeline_one': [
                {'source': {'uri': 'test'}}
            ],
            'pipeline_two': [
                {'source': {'uri': 'test2'}}
            ]
        }
    )
    assert isinstance(p[0], interpreter.Pipeline)
    assert p[0].name == 'pipeline_one'
    assert isinstance(p[0]._pipe_elements[0], _TestSourceElement)
    assert isinstance(p[1], interpreter.Pipeline)
    assert p[0]._pipe_elements[0].config['uri'] == 'test'
    assert p[1].name == 'pipeline_two'
    assert isinstance(p[1]._pipe_elements[0], _TestSourceElement)
    assert p[1]._pipe_elements[0].config['uri'] == 'test2'


def test_pipeline_start():
    p = _one_pipeline_setup()
    p[0].load_elements()
    assert p[0]._pipe_elements[0].state == pipeline.PIPE_STATE_STOPPED
    pe = p[0]._pipe_elements[0]
    assert isinstance(pe, _TestSourceElement)
    assert not pe.start_called
    p[0].start()
    assert p[0]._pipe_elements[0].state == pipeline.PIPE_STATE_RUNNING
    assert pe.start_called


def test_pipeline_stop():
    p = _one_pipeline_setup()
    p[0].start()
    pe = p[0]._pipe_elements[0]
    assert pe.state == pipeline.PIPE_STATE_RUNNING
    assert pe.start_called
    assert not pe.stop_called
    p[0].stop()
    # make sure the correct element is still in the correct pipe position
    pe = p[0]._pipe_elements[0]
    assert isinstance(pe, _TestSourceElement)
    assert pe.state == pipeline.PIPE_STATE_STOPPED



def test_pipeline_source_config():
    """Test the source ref is resolved"""

    config.update({
        "sources": {
            "source1": {
                "uri": "test",
                "type": "video",
                "live": False
            }
        },
        "pipelines": {
            "pipeline1": [
                {"source": "source1"}
            ]
        }
    })

    p = _one_pipeline_setup(pipelines_config=config.pipelines)
    p[0].load_elements()

    log.debug(p[0]._pipe_elements[0])
    assert p[0]._pipe_elements[0].config["uri"] == "test"


# def test_pipeline_ai_model_config():
#     """Test the source ref is resolved"""

#     config.update({
#         "ai_models": {
#             "test": {
#                 "labels": "ai_models/coco_labels.txt",
#                 "model": {
#                     "tflite": "ai_models/mobilenet_ssd_v2_coco_quant_postprocess.tflite",
#                 }
#             }
#         },
#         "pipelines": {
#             "pipeline1": [
#                 {"source": {"uri": "test"}},
#                 {"detect_objects": {
#                     "ai_model": "test",
#                     "confidence_threshold": 0.0,
#                     "label_filter": ['person', 'car']
#                 }}
#             ]
#         }
#     })

#     p = _one_pipeline_setup(pipelines_config=config.pipelines)
#     p[0].load_elements()

#     log.debug(p[0]._pipe_elements[1])
#     assert len(p[0]._pipe_elements) == 2
#     assert p[0]._pipe_elements[1]._tfengine._confidence_threshold == 0.0
#     assert p[0]._pipe_elements[1]._label_filter == ['person', 'car']