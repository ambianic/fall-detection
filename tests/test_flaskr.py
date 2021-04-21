import os
import json
import tempfile
import pytest
from ambianic.webapp import flaskr
from ambianic import config, __version__
import logging

log = logging.getLogger(__name__)


def reset_config():
    config.reload()


def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    reset_config()


def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method."""
    reset_config()


@pytest.fixture
def client():

    app = flaskr.create_app()

    app.config['TESTING'] = True

    with app.test_client() as fclient:
        # with app.app_context():
        #     flaskr.init_db()
        yield fclient


def test_hello(client):
    rv = client.get('/')
    assert b'Ambianic Edge!' in rv.data


def test_health_check(client):
    rv = client.get('/healthcheck')
    assert b'is running' in rv.data


# broken?
# def test_pipelines(client):
#     rv = client.get('/pipelines')
#     log.debug("data %s", rv)
#     # assert b'is running' in rv.data

def test_status(client):
    rv = client.get('/api/status')
    assert (json.loads(rv.data)["status"], json.loads(
        rv.data)["version"]) == ("OK", __version__)


def test_get_timeline(client):
    rv = client.get('/api/timeline')
    assert json.loads(rv.data)["status"] == "success"


def test_get_samples(client):
    rv = client.get('/api/samples')
    assert json.loads(rv.data)["status"] == "success"


def test_add_samples(client):
    rv = client.post('/api/samples', json={
        'title': None,
        'author': None,
        'read': None,
    })
    data = json.loads(rv.data)
    assert data["status"] == "success"


def test_update_samples(client):
    rv = client.post('/api/samples', json={
        'title': None,
        'author': None,
        'read': None,
    })
    log.debug("%s", json.loads(rv.data))
    sample_id = json.loads(rv.data)["sample_id"]
    rv = client.put('/api/samples/' + sample_id, json={
        'title': None,
        'author': None,
        'read': None,
    })
    assert json.loads(rv.data)["status"] == "success"


def test_delete_samples(client):
    rv = client.post('/api/samples', json={
        'title': None,
        'author': None,
        'read': None,
    })
    log.debug("%s", json.loads(rv.data))
    sample_id = json.loads(rv.data)["sample_id"]
    rv = client.delete('/api/samples/' + sample_id)
    assert json.loads(rv.data)["status"] == "success"


def test_get_config(client):
    rv = client.get('/api/config')
    data = json.loads(rv.data)
    assert data is not None


def test_save_source(client):
    rv = client.put('/api/config/source/test1', json={
        "uri": "test",
        "type": "video",
        "live": True
    })
    data = json.loads(rv.data)
    assert data
    assert data["id"] == "test1"


def test_delete_source(client):
    rv = client.put('/api/config/source/test1', json={
        "uri": "test",
        "type": "video",
        "live": True
    })
    data = json.loads(rv.data)
    assert data["id"] == "test1"
    rv = client.delete('/api/config/source/test1')
    data = json.loads(rv.data)
    assert data["status"] == "success"


def test_ping(client):
    rv = client.get('/api/ping')
    assert b'pong' in rv.data


def test_static(client):
    rv = client.get('/static/ppview.css')
    log.debug("%s", rv.data)
    assert rv.data is not None
