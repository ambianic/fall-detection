"""Main Ambianic OS executable."""
import os
import ambianic
import ambianic.server
import signal


def main():
    """Start the main app executable in the hosting OS environment."""
    env_work_dir = ambianic.get_work_dir()
    ambianic.server_instance = ambianic.server.AmbianicServer(
        work_dir=env_work_dir
    )
    # run with a little lower priority
    # to avoid delaying docker container from syncing with OS resources
    # such as log files
    os.nice(1)
    # start main server
    ambianic.server_instance.start()


def stop():
    """Stop a running app executable in the hosting OS environment."""
    ambianic.server_instance.stop()


def _service_shutdown(signum=None, frame=None):
    """Wrap system exit signal into an application exception.

    The ServiceExit exception will be caught in main process threads
    and handled gracefully.
    """
    print('Caught system shutdown signal (Ctrl+C or similar). '
          'Signal code: ', signum)
    raise ambianic.util.ServiceExit


def _register_sys_handlers():  # pragma: no cover
    """Register system exit handler for graceful shutdown."""
    signal.signal(signal.SIGTERM, _service_shutdown)
    signal.signal(signal.SIGINT, _service_shutdown)


if __name__ == '__main__':  # pragma: no cover
    _register_sys_handlers()
    main()
