"""Test configuration functions."""
import logging
import logging.handlers
import os
from ambianic import config, logger

log = logging.getLogger(__name__)

_dir = os.path.dirname(os.path.abspath(__file__))


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    config.reload()


def test_log_config_with_file():
    log_path = os.path.join(_dir, '.__test-log.txt')
    log_config = {
        'file': log_path,
    }
    logger.configure(config=log_config)
    handlers = logging.getLogger().handlers
    log_fn = None
    for h in handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            log_fn = h.baseFilename
            assert log_fn == log_config['file']
    # at least one log file name should be configured
    assert log_fn


def test_log_config_without_file():
    log_config = {}
    logger.configure(config=log_config)
    handlers = logging.getLogger().handlers
    for h in handlers:
        assert not isinstance(h, logging.handlers.RotatingFileHandler)


def test_log_config_with_debug_level():
    log_config = {
        'level': 'DEBUG'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    effective_level = root_logger.getEffectiveLevel()
    lname = logging.getLevelName(effective_level)
    assert lname == log_config['level']


def test_log_config_with_warning_level():
    log_config = {
        'level': 'WARNING'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    effective_level = root_logger.getEffectiveLevel()
    lname = logging.getLevelName(effective_level)
    assert lname == log_config['level']


def test_log_config_without_level():
    log_config = {}
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    effective_level = root_logger.getEffectiveLevel()
    assert effective_level == logger.DEFAULT_FILE_LOG_LEVEL


def test_log_config_bad_level1():
    log_config = {
        'level': '_COOCOO_'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    effective_level = root_logger.getEffectiveLevel()
    assert effective_level == logger.DEFAULT_FILE_LOG_LEVEL


def test_log_config_bad_level2():
    log_config = {
        'level': 2.56
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    effective_level = root_logger.getEffectiveLevel()
    assert effective_level == logger.DEFAULT_FILE_LOG_LEVEL


def test_log_config_with_debug_console_level():
    log_config = {
        'console_level': 'DEBUG'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = logging.getLevelName(h.level)
            assert console_log_level == log_config['console_level']


def test_log_config_with_warning_console_level():
    log_config = {
        'console_level': 'WARNING'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = logging.getLevelName(h.level)
            assert console_log_level == log_config['console_level']




def test_log_config_with_info_console_level():
    log_config = {
        'console_level': 'INFO'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = logging.getLevelName(h.level)
            assert console_log_level == log_config['console_level']


def test_log_config_without_console_level():
    log_config = {}
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = h.level
            assert console_log_level == logger.DEFAULT_CONSOLE_LOG_LEVEL


def test_log_config_bad_console_level1():
    log_config = {
        'console_level': '_COOCOO_'
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = h.level
            assert console_log_level == logger.DEFAULT_CONSOLE_LOG_LEVEL



def test_log_config_bad_console_level2():
    log_config = {
        'console_level': 2.56
    }
    logger.configure(config=log_config)
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            console_log_level = h.level
            assert console_log_level == logger.DEFAULT_CONSOLE_LOG_LEVEL
