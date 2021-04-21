"""Main module for Ambianic AI pipelines."""
import logging
import abc
import time
from typing import Iterable
from ambianic.util import ManagedService
from ambianic.pipeline.timeline import PipelineContext

log = logging.getLogger(__name__)

# Define pipe lifecycle states
PIPE_STATE_STOPPED = 0
PIPE_STATE_RUNNING = 10
PIPE_STATES = [PIPE_STATE_RUNNING, PIPE_STATE_STOPPED]


class PipeElement(ManagedService):
    """The basic building block of an Ambianic pipeline."""

    def __init__(self,
                 element_name=None,
                 context: PipelineContext = None,
                 event_log: logging.Logger = None,
                 **kwargs):
        """Create a PipeElement instance."""
        super().__init__()
        self._name = element_name
        self._state = PIPE_STATE_STOPPED
        self._next_element = None
        self._latest_heartbeat = time.monotonic()
        self._context = context
        self._timeline_event_log = event_log

    @property
    def name(self) -> str:
        """Return this element's reference name in pipeline definitions."""
        return self._name

    @property
    def context(self) -> PipelineContext:
        """Pipeline execution context.

        :Returns:
        -------
        type: PipelineContext
            pipeline execution context

        """
        return self._context

    def push_context(self, element_context: dict = None):
        """Push this element information to the context stack.

        Invoke before the element yields its first sample output
        for a given input sample.

        :Parameters:
        ----------
        element_context : dict
            Contextual info about this element.

        """
        if element_context is None:
            element_context = {}
        element_context['class'] = self.__class__.__name__
        self._context.push_element_context(element_context)

    def pop_context(self) -> dict:
        """Pop element information from the context stack.

        Invoke after the element yields its last sample output
        for a given input sample.

        :Returns:
        -------
        type: dict
            Element context info.

        """
        return self._context.pop_element_context()

    @property
    def event_log(self) -> logging.Logger:
        """Get timeline event log for the current pipe execution context."""
        return self._timeline_event_log

    @property
    def state(self):
        """Lifecycle state of the pipe element."""
        return self._state

    def start(self):
        """Only sourcing elements (first in a pipeline) need to override.

        It is invoked once when the enclosing pipeline is started. It should
        continue to run until the corresponding stop() method is invoked on the
        same object from a separate pipeline lifecycle manager thread.

        It is recommended for overriding methods to invoke this base method
        via super().start() before proceeding with custom logic.

        """
        self._state = PIPE_STATE_RUNNING

    def heal(self):  # pragma: no cover
        """Override with adequate implementation of a healing procedure.

        heal() is invoked by a lifecycle manager when its determined that
        the element does not respond within reasonable timeframe.
        This can happen for example if the element depends on external IO
        resources, which become unavailable for an extended period of time.

        The healing procedure should be considered a chance to recover or find
        an alternative way to proceed.

        If heal does not reset the pipe element back to a responsive state,
        it is likely that the lifecycle manager will stop the
        element and its ecnlosing pipeline.

        """
        pass

    def healthcheck(self):
        """Check the health of this element.

        :returns: (timestamp, status) tuple with most recent heartbeat
        timestamp and health status code ('OK' normally).
        """
        status = 'OK'  # At some point status may carry richer information
        return self._latest_heartbeat, status

    def heartbeat(self):
        """Set the heartbeat timestamp to time.monotonic().

        Keeping the heartbeat timestamp current informs
        the lifecycle manager that this element is functioning
        well.

        """
        now = time.monotonic()
        self._latest_heartbeat = now

    def stop(self):
        """Receive stop signal and act accordingly.

        Subclasses implementing sourcing elements should override this method
        by first invoking their super class implementation and then running
        through steps specific to stopping their ongoing sample processing.

        """
        self._state = PIPE_STATE_STOPPED

    def connect_to_next_element(self, next_element=None):
        """Connect this element to the next element in the pipe.

        Subclasses should not override this method.

        """
        assert next_element
        assert isinstance(next_element, PipeElement)
        self._next_element = next_element

    def receive_next_sample(self, **sample):
        """Receive next sample from a connected previous element if applicable.

        All pipeline elements except for the first (sourcing) element
        in the pipeline will depend on this method to feed them with new
        samples to process.

        Subclasses should not override this method.

        :Parameters:
        ----------
        **sample : dict
            A dict of (key, value) pairs that represent the sample.
            It is left to specialized implementations of PipeElement to specify
            their in/out sample formats and enforce compatibility with
            adjacent connected pipe elements.

        """
        self.heartbeat()
        for processed_sample in self.process_sample(**sample):
            if self._next_element:
                if (processed_sample):
                    self._next_element.receive_next_sample(**processed_sample)
                else:
                    self._next_element.receive_next_sample()
                self.heartbeat()

    def process_sample(self, **sample) -> Iterable[dict]:
        """Override and implement as generator.

        Invoked by receive_next_sample() when the previous element
        (or pipeline source) feeds another data input sample.

        Implementing subclasses should process input samples and yield
        output samples for the next element in the pipeline.

        :Parameters:
        ----------
        **sample : dict
            A dict of (key, value) pairs that represent the sample.
            It is left to specialized implementations of PipeElement to specify
            their in/out sample formats and enforce compatibility with
            adjacent connected pipe elements.

        :Returns:
        ----------
        processed_sample: Iterable[dict]
            Generates processed samples to be passed on
            to the next pipeline element.

        """
        yield sample


class HealthChecker(PipeElement):
    """Monitor overall pipeline throughput health.

    Attaches at the end of a pipeline to monitor its health status
    based on received output samples and their frequency.
    """

    def __init__(self, health_status_callback=None, **kwargs):
        """Create instance given health status callback.

        The health status call back will be invoked each time
        the sample_process method is invoked.

        :Parameters:
        ----------
        health_status_callback : function
            Method that is expected to measure the overall pipeline throughput
            health.
        """
        super().__init__(**kwargs)
        assert health_status_callback
        self._health_status_callback = health_status_callback

    def process_sample(self, **sample):
        """Call health callback and pass on sample as is."""
        log.debug('%s received sample from the connected '
                  'preceding pipe element.',
                  self.__class__.__name__
                  )
        self._health_status_callback()
        yield sample
