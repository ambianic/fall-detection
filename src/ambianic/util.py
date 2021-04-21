"""Service classes for OS interaction and multithreading."""
from threading import Thread, Event
import logging
import traceback
import time
from abc import abstractmethod

log = logging.getLogger(__name__)


class ManagedService:
    """A service contract with lifecycle methods."""

    @abstractmethod
    def start(self, **kwargs):
        """Start and run until finished or asked to stop()."""

    @abstractmethod
    def stop(self):
        """Exit start() method as soon as possible.

        Delay to exit may result in forceful process termination.
        """

    @abstractmethod
    def healthcheck(self):
        """Report vital health information.

        :Returns:
        -------
        touple (time, string)
            (latest_heartbeat_timestamp, status_code)
            latest_heartbeat_timestamp is in the format of time.monotonic().
            status_code should have a semantic mapping to the service health.

        """
        return time.monotonic(), 'OK'

    def heal(self):
        """Inspect and repair the state of the job.

        This method is normally invoked from a management service when
        a long running job doesn't seem healthy.
        Most likely because healthstate() returns bad reports:
        such as outdated heartbeat timestamp or a bad status code.
        The method should try to bring the job back to a healthy state
        as soon as possible to prevent forceful termination.
        """


class ThreadedJob(Thread, ManagedService):
    """A job that runs in its own python thread.

    Jobs managed by Threaded Job must implement all ManagedService methods.
    """

    # Reminder: even though multiple processes can work well for pipelines,
    # since they are mostly independent,
    # Google Coral does not allow access to it from different processes yet.
    # Consider moving access to coral in a separate process that can serve
    # an inference task queue from multiple pipelines.

    def __init__(self, job=None):
        """Inititalize with a ManagedService.

        :Parameters:
        ----------
        job : ManagedService
            The underlying service wrapped in this thread.
        """
        Thread.__init__(self, daemon=True)
        assert isinstance(job, ManagedService)
        self.job = job
        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        # self.shutdown_flag = threading.Event()
        # ... Other thread setup code here ...
        self._stop_requested = Event()

    def run(self):
        log.info('Thread #%s started with job: %s',
                 self.ident,
                 self.job.__class__.__name__)
        self.job.start()
        log.info('Thread #%s for job %s stopped',
                 self.ident,
                 self.job.__class__.__name__)

    def stop(self):
        log.debug('Thread #%s for job %s is signalled to stop'
                  'Passing request to job.',
                  self.ident,
                  self.job.__class__.__name__)
        self._stop_requested.set()
        self.job.stop()

    def heal(self):
        log.debug('Thread #%s for job %s is signalled to heal.'
                  'Passing request to job.',
                  self.ident,
                  self.job.__class__.__name__)
        self.job.heal()
        log.debug('Thread #%s for job %s completed heal request.',
                  self.ident,
                  self.job.__class__.__name__)

    def healthcheck(self):
        log.debug('Thread #%s for job %s healthcheck requested.'
                  'Passing request to job.',
                  self.ident,
                  self.job.__class__.__name__)
        health_status = self.job.healthcheck()
        return health_status


class ServiceExit(Exception):
    """Main program exit signal.

    Custom exception which is used to trigger clean exit
    of all running threads and the main program.

    Method for controlled multi-threaded Python app exit
    suggested by George Notaras:
    https://www.g-loaded.eu/2016/11/24/how-to-terminate-running-python-threads-using-signals/
    """


def stacktrace():
    """Get stack trace as a multi line string.

    :Returns:
    -------
    string
        Milti-line string of current stack trace.

    """
    formatted_lines = traceback.format_exc().splitlines()
    strace = 'Runtime stack trace:\n %s' + '\n'.join(formatted_lines)
    return strace
