"""Test configuration functions."""
import logging
import logging.handlers
import os
from time import sleep
import ambianic
from ambianic.server import AmbianicServer
from ambianic import server, config, logger, load_config
import yaml
import pytest

log = logging.getLogger(__name__)

_dir = os.path.dirname(os.path.abspath(__file__))

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    config.reload()


def test_get_workdir_env():
    os.environ['AMBIANIC_DIR'] = "/foo"
    assert ambianic.get_work_dir() == "/foo"
    os.environ['AMBIANIC_DIR'] = ""
    assert ambianic.get_work_dir() == ambianic.DEFAULT_WORK_DIR


def test_no_pipelines():
    load_config(os.path.join(_dir, 'test-config-no-pipelines.yaml'), True)
    assert config.get("pipelines", None) is None


def test_secrets():
    default_secret = ambianic.__SECRETS_FILE
    ambianic.__SECRETS_FILE = os.path.join(_dir, "secrets.yaml")
    cfg = load_config(os.path.join(_dir, 'test-config-secrets.yaml'), True)
    ambianic.__SECRETS_FILE = default_secret
    assert cfg.get("question") == 42
    assert cfg.deeeper.question.on.life == 42