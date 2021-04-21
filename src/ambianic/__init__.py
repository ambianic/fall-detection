import os
from dynaconf import Dynaconf
from dynaconf.utils.boxing import DynaBox
from typing import Union
import importlib_metadata as metadata

DEFAULT_WORK_DIR: str = '/workspace'
DEFAULT_DATA_DIR: str = './data'

DEFAULT_CONFIG_FILE: str = 'config.yaml'
DEFAULT_SECRETS_FILE: str = 'secrets.yaml'

__CONFIG_FILE: str = None
__SECRETS_FILE: str = None
__version__: str = metadata.version('ambianic-edge')


def get_config_file() -> str:
    return __CONFIG_FILE


def get_secrets_file() -> str:
    if __SECRETS_FILE:
        return __SECRETS_FILE
    return os.path.join(get_work_dir(), DEFAULT_SECRETS_FILE)


def __merge_secrets(config: Union[Dynaconf, DynaBox], src_config: Dynaconf = None):
    if src_config is None:
        src_config = config
    for key, val in config.items():
        if isinstance(val, dict):
            __merge_secrets(val, src_config)
            continue
        # NOTE value must be an exact match to avoid interfering
        # with other templates
        if (
            isinstance(val, str) and
                (val[0:2] == "${" and val[-1] == "}")
        ):
            ref_key = val[2:-1]
            ref_val = src_config.get(ref_key, None)
            if ref_val is not None:
                config[key] = ref_val
            continue


def __init_config() -> Dynaconf:
    config = Dynaconf(
        settings_files=[get_config_file(), get_secrets_file()],
        # secrets=[],
        merge=True,
        environments=False,
    )
    __merge_secrets(config)
    return config


def load_config(filename: str, clean: bool = False) -> Dynaconf:
    if clean:
        config.clean()
    if filename:
        config.load_file(path=[filename, get_secrets_file()])
        __merge_secrets(config)
        global __CONFIG_FILE
        __CONFIG_FILE = filename
    return config


def get_work_dir() -> str:
    """Retrieve the ambianic working directory"""
    env_work_dir = os.environ.get('AMBIANIC_DIR', os.getcwd())
    if not env_work_dir:
        env_work_dir = DEFAULT_WORK_DIR
    return env_work_dir


# initialization
server_instance = None

__CONFIG_FILE = os.path.join(get_work_dir(), DEFAULT_CONFIG_FILE)
config: Dynaconf = __init_config()
