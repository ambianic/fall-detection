"""More test cases for ambianic.interpreter module."""
from ambianic import pipeline, config
from ambianic.pipeline import interpreter
from ambianic.pipeline.avsource.av_element import AVSourceElement
from ambianic.pipeline.interpreter import \
    PipelineServer, Pipeline, HealingThread, PipelineServerJob
import logging
import time
import threading
import os
from dynaconf.utils import DynaconfDict
# mocked_settings = DynaconfDict({'FOO': 'BAR'})


log = logging.getLogger()
log.setLevel(logging.DEBUG)

def test_pipeline_server_start_stop():

    _dir = os.path.dirname(os.path.abspath(__file__))

    config.update({
        "sources": {
            "test": {
                "uri": "file://" + os.path.join(
                    _dir,
                    "avsource/test2-cam-person1.mkv"
                )
            }
        },
        "pipelines": {
            "test": [
                {"source": "test"}
            ]
        }
    })

    srv = PipelineServer(config)
    srv.start()
    srv.stop()
