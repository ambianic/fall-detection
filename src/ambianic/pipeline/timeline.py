"""Pipeline event timeline read/write/search functions."""

from concurrent_log_handler import ConcurrentRotatingFileHandler
import logging
import yaml
import uuid
import os
import pathlib

log = logging.getLogger(__name__)
TIMELINE_EVENT_LOGGER_NAME = __name__ + '__timeline__event__logger__'
PIPELINE_CONTEXT_KEY = 'pipeline_context'


class PipelineEvent:
    """Encapsulates information for a pipeline timeline event."""

    def __init__(self, message: str = None, **kwargs):
        """Create a new event instance.

        :Parameters:
        ----------
        message : String
            Human readable display message for the event.
        **kwargs : type
            Additional event arguments.

        """
        self.message = message
        self.kwargs = kwargs
        self.args = {}
        self.args['message'] = self.message
        self.args['args'] = self.kwargs

    def __str__(self):
        """Format event as yaml string."""
        s = yaml.dump(self.kwargs)
        return s


class PipelineContext:
    """Runtime dynamic context for a pipeline.

    Carries information
    such as pipeline name and pipe element stack
    up to and including the element firing the event.

    """

    def __init__(self, unique_pipeline_name: str = None):
        """Instantiate timeline context for a pipeline.

        :Parameters:
        ----------
        unique_pipeline_name : str
            The unique runtime name of a pipeline.

        """
        self._unique_pipeline_name = unique_pipeline_name
        self._element_stack = []
        self._data_dir = None

    @property
    def unique_pipeline_name(self):
        """Return pipeline unique name."""
        return self._unique_pipeline_name

    @property
    def data_dir(self):
        """Return system wide configured data dir."""
        return self._data_dir

    @data_dir.setter
    def data_dir(self, dd=None):
        """Set system wide configured data dir."""
        self._data_dir = dd

    def push_element_context(self, element_context: dict = None):
        """Push new element information to the context stack."""
        self._element_stack.append(element_context)

    def pop_element_context(self) -> dict:
        """Pop element information from the context stack."""
        return self._element_stack.pop()


class PipelineEventFormatter(logging.Formatter):
    """Custom logging formatter for pipeline events."""

    def format(self, record: logging.LogRecord = None) -> str:
        """Populate event information and return as yaml formatted string."""
        # s = super().format(record)
        s = None
        e = {}
        e['id'] = uuid.uuid4().hex
        e['message'] = record.getMessage()
        # log.warning('record.message: %r', record.getMessage())
        # log.warning('record.args: %r', record.args)
        e['created'] = record.created
        e['priority'] = record.levelname

        e['args'] = record.args

        e['source_code'] = {}
        e['source_code']['pathname'] = record.pathname
        e['source_code']['funcName'] = record.funcName
        e['source_code']['lineno'] = record.lineno
        ctx = record.args.get(PIPELINE_CONTEXT_KEY, None)
        if ctx:
            e[PIPELINE_CONTEXT_KEY] = ctx.toDict()
        # use array enclosure a[] to mainain the log file
        # yaml compliant as new events are appended
        # - event1:
        # - event2:
        # - ...
        a = [e]
        s = yaml.dump(a)
        return s


def configure_timeline(config: dict = None):
    """Initialize timeline event logger.

    Sets up pipeline event logger once to be reused by pipelines
    in the current runtime.
    Should be called before any pipeline starts.

    A good place to initialize it is around the time when the root logger
    is initialized.

    :Parameters:
    ------------

    config : dict
        A dictionary of configuration parameters.

    """
    if config is None:
        config = {}
    log_filename = config.get('event_log', None)
    if not log_filename:
        log_filename = 'timeline-event-log.yaml'
    log_directory = os.path.dirname(log_filename)
    with pathlib.Path(log_directory) as log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Timeline event log messages directed to {}".
              format(log_filename))
    event_log = logging.getLogger(TIMELINE_EVENT_LOGGER_NAME)
    event_log.setLevel(logging.INFO)
    # Use rotating files as log message handler
    handler = ConcurrentRotatingFileHandler(
        log_filename,
        # each event file will keep up to 100K data
        maxBytes=100*1024,
        # 100 backup files will be kept. Older will be erased.
        backupCount=100
    )
    fmt = PipelineEventFormatter()
    handler.setFormatter(fmt)
    # remove any other handlers that may be assigned previously
    # and could cause unexpected log collisions
    event_log.handlers = []
    # add custom event handler
    event_log.addHandler(handler)


def get_event_log(pipeline_context: PipelineContext = None) \
  -> logging.Logger:
    """Get an instance of pipeline event logger.

    :Parameters:
    ----------
    pipe_context : PipelineContext

    :Returns:
    -------
    type
        Implementation of logging.Logger that handles pipeline events

    """
    pipeline_event_log = logging.getLogger(TIMELINE_EVENT_LOGGER_NAME)
    # wrap logger in an adapter that carries pipeline context
    # such as pipeline name and current pipe element.
    pipeline_event_log = logging.LoggerAdapter(
        pipeline_event_log,
        {PIPELINE_CONTEXT_KEY: pipeline_context})
    return pipeline_event_log
