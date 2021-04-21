"""Logging functionalities wrapper"""
import logging
import logging.handlers
import os
import pathlib

DEFAULT_FILE_LOG_LEVEL = logging.INFO
DEFAULT_CONSOLE_LOG_LEVEL = logging.WARN

log = logging.getLogger(__name__)


def _get_log_level(log_level, default_log_level):
    numeric_level = default_log_level
    if log_level:
        try:
            numeric_level = getattr(logging, log_level.upper(),
                                    default_log_level)
        except AttributeError as e:
            log.warning("Invalid log level: %s . Error: %s", log_level, e)
            log.warning('Defaulting log level to %s', default_log_level)
    fmt = None
    if numeric_level <= logging.INFO:
        format_cfg = '%(asctime)s %(levelname)-4s ' \
            '%(pathname)s.%(funcName)s(%(lineno)d): %(message)s'
        datefmt_cfg = '%Y-%m-%d %H:%M:%S'
        fmt = logging.Formatter(fmt=format_cfg,
                                datefmt=datefmt_cfg, style='%')
    else:
        fmt = logging.Formatter()

    return numeric_level, fmt


def configure(config=None):
    if config is None:
        config = {}

    log_level = config.get("level", None)

    console_log_level = config.get("console_level", log_level)

    numeric_level, fmt = _get_log_level(
        log_level, default_log_level=DEFAULT_FILE_LOG_LEVEL)

    root_logger = logging.getLogger()
    # remove any other handlers that may be assigned previously
    # and could cause unexpected log collisions
    root_logger.handlers = []
    # add a console handler that only shows errors and warnings    
    ch = logging.StreamHandler()

    console_numeric_level, console_fmt = _get_log_level(
        console_log_level, default_log_level=DEFAULT_CONSOLE_LOG_LEVEL)
    ch.setLevel(console_numeric_level)
    # add formatter to ch
    ch.setFormatter(console_fmt)
    # add ch to logger
    root_logger.addHandler(ch)
    # add a file handler if configured
    log_filename = config.get('file', None)
    if log_filename:
        log_directory = os.path.dirname(log_filename)
        with pathlib.Path(log_directory) as log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            print("Log messages directed to {}".format(log_filename))
        handler = logging.handlers.RotatingFileHandler(
            log_filename,
            # each log file will be up to 10MB in size
            maxBytes=100*1024*1024,
            # 20 backup files will be kept. Older will be erased.
            backupCount=20
        )
        handler.setFormatter(fmt)
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
    effective_level = log.getEffectiveLevel()
    assert numeric_level == effective_level
    log.info('Logging configured with level %s',
             logging.getLevelName(effective_level))
    if effective_level <= logging.DEBUG:
        log.debug('Configuration yaml dump:')
        log.debug(config)