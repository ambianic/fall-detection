"""Flask based Web services."""
import os
import logging
import time
from pathlib import Path
import flask
from flask import Flask, request, jsonify, json
from flask_cors import CORS
from flask.logging import default_handler
from requests import get
from werkzeug.serving import make_server
from werkzeug.exceptions import HTTPException
from ambianic import config, DEFAULT_DATA_DIR, __version__
from ambianic.util import ServiceExit, ThreadedJob, ManagedService
from ambianic.webapp.server import samples, config_sources

log = logging.getLogger(__name__)

# configuration
DEBUG = True


class FlaskJob(ManagedService):
    """Flask based managed web service."""

    def __init__(self, config):
        """Create Flask based web service."""
        self.config = config
        data_dir = None
        if config:
            data_dir = config.get('data_dir', None)
        if not data_dir:
            data_dir = DEFAULT_DATA_DIR
        self.srv = None
        app = create_app(data_dir=data_dir)
        ip_address = '0.0.0.0'
        port = 8778
        log.info('starting flask web server on %s:%d', ip_address, port)
        self.srv = make_server(ip_address, port, app)
        ctx = app.app_context()
        ctx.push()
        with app.app_context():
            flask.current_app.data_dir = data_dir
        self.flask_stopped = True
        log.debug('Flask process created')

    def start(self, **kwargs):
        """Start service."""
        log.debug('Flask starting main loop')
        self.flask_stopped = False
        try:
            self.srv.serve_forever()
        except ServiceExit:
            log.info('Service exit requested')
        self.flask_stopped = True
        log.debug('Flask ended main loop')

    def stop(self):
        """Stop service."""
        if not self.flask_stopped:
            log.debug('Flask stopping main loop')
            self.srv.shutdown()
            log.debug('Flask main loop ended')

    def healthcheck(self):
        """Report health status."""
        return time.monotonic(), 'OK'


class FlaskServer(ManagedService):
    """ Thin wrapper around Flask constructs.

    Allows controlled start and stop of the web app server
    in a separate process.

    Parameters
    ----------
    config : yaml
        reference to the yaml configuration file

    """

    def __init__(self, config):
        self.config = config
        self.flask_job = None

    def start(self, **kwargs):
        log.info('Flask server job starting...')
        f = FlaskJob(self.config)
        self.flask_job = ThreadedJob(f)
        self.flask_job.start()
        log.info('Flask server job started')

    def healthcheck(self):
        # Note: Implement actual health check for Flask
        # See if the /healthcheck URL returns a 200 quickly
        return time.monotonic(), True

    def heal(self):
        """Heal the server.

        TODO: Keep an eye for potential scenarios that cause this server to
         become unresponsive.
        """

    def stop(self):
        if self.flask_job:
            log.info('Flask server job stopping...')
            self.flask_job.stop()
            self.flask_job.join()
            log.info('Flask server job stopped.')


def create_app(data_dir=None):
    log.debug('Creating Flask app...')

    # if Ambianic is in INFO or DEBUG mode, pass that info on to Flask
    if log.level <= logging.INFO:
        os.environ['FLASK_ENV'] = 'development'

    # create and configure the web app
    # set the project root directory as the static folder, you can set others.
    app = Flask(__name__, instance_relative_config=True)
    app.logger.removeHandler(default_handler)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # enable CORS for development
    CORS(app, resources={r'/*': {'origins': '*'}})

    # [Sitemap]
    # sitemap definitions follow

    # a simple page that says hello
    @app.route('/')
    def hello():
        return 'Ambianic Edge! Helpful AI for home and business automation.'

    # healthcheck page available to docker-compose
    # and other health monitoring tools
    @app.route('/healthcheck')
    def health_check():
        return 'Ambianic Edge is running in a cheerful healthy state!'

    # live view of ambianic pipelines
    @app.route('/pipelines')
    def view_pipelines():
        return flask.render_template('pipelines.html')

    # healthcheck page available to docker-compose
    # and other health monitoring tools
    @app.route('/api/status')
    def get_status():
        response_object = {'status': 'OK', 'version': __version__}
        resp = jsonify(response_object)
        return resp

    @app.route('/api/timeline', methods=['GET'])
    @app.route('/api/timeline.json', methods=['GET'])
    def get_timeline():
        response_object = {'status': 'success'}
        req_page = request.args.get('page', default=1, type=int)
        log.debug('Requested timeline events page" %d', req_page)
        nonlocal data_dir
        resp = samples.get_timeline(page=req_page, data_dir=data_dir)
        response_object['timeline'] = resp
        log.debug('Returning %d timeline events', len(resp))
        # log.debug('Returning samples: %s ', response_object)
        resp = jsonify(response_object)
        return resp

    @app.route('/api/samples', methods=['GET', 'POST'])
    def get_samples():
        response_object = {'status': 'success'}
        if request.method == 'POST':
            post_data = request.get_json()
            new_sample = {
                'title': post_data.get('title'),
                'author': post_data.get('author'),
                'read': post_data.get('read')
            }
            samples.add_sample(new_sample)
            response_object['message'] = 'Sample added!'
            response_object['sample_id'] = new_sample["id"]
            log.debug('Sample added: %s ', new_sample)
        else:
            req_page = request.args.get('page', default=1, type=int)
            resp = samples.get_samples(page=req_page)
            response_object['samples'] = resp
            log.debug('Returning %d samples', len(resp))
        # log.debug('Returning samples: %s ', response_object)
        resp = jsonify(response_object)
        return resp

    @app.route('/api/samples/<sample_id>', methods=['PUT', 'DELETE'])
    def update_sample(sample_id):
        response_object = {'status': 'success'}
        if request.method == 'PUT':
            post_data = request.get_json()
            sample = {
                'id': sample_id,
                'title': post_data.get('title'),
                'author': post_data.get('author'),
                'read': post_data.get('read')
            }
            log.debug('update_sample %s', sample)
            samples.update_sample(sample)
            response_object['message'] = 'Sample updated!'
        if request.method == 'DELETE':
            samples.delete_sample(sample_id)
            response_object['message'] = 'Sample removed!'
        return jsonify(response_object)

    @app.route('/api/config', methods=['GET'])
    def get_config():
        return jsonify(config.as_dict())

    @app.route(
        '/api/config/source/<source_id>',
        methods=['GET', 'PUT', 'DELETE']
    )
    def handle_config_source(source_id):

        if request.method == 'DELETE':
            config_sources.remove(source_id)
            return jsonify({'status': 'success'})

        if request.method == 'PUT':
            source = request.get_json()
            config_sources.save(source_id, source)

        return jsonify(config_sources.get(source_id))

    # sanity check route
    @app.route('/api/ping', methods=['GET'])
    def ping():
        response_object = 'pong'
        return jsonify(response_object)

    @app.route('/static/<path:path>')
    def static_file(path):
        return flask.send_from_directory('static', path)

    @app.route('/api/data/<path:path>')
    def data_file(path):
        data_path = Path(DEFAULT_DATA_DIR).resolve()
        log.info('Serving static data file from: %r', data_path / path)
        return flask.send_from_directory(data_path, path)

    @app.route('/client', defaults={'path': 'index.html'})
    @app.route('/client/', defaults={'path': 'index.html'})
    @app.route('/client/<path:path>')
    def client_file(path):
        if log.level <= logging.DEBUG:  # development mode
            hostname = flask.request.host.split(':')[0]
            base_uri = 'http://{host}:1234/'.format(host=hostname)
            return get(f'{base_uri}{path}').content
        # production mode
        return flask.send_from_directory('client/dist', path)

    @app.errorhandler(Exception)
    def handle_exception(e: Exception):
        """Return JSON instead of HTML for HTTP errors."""

        # start with the correct headers and status code from the error
        if isinstance(e, HTTPException):
            response = e.get_response()
            response.content_type = "application/json"
            # replace the body with JSON
            response.data = json.dumps({
                "code": e.code,
                "error": e.description,
            })
            return response

        # generic error handler
        log.error("Request failed")
        log.exception(e)

        return jsonify(
            code=500,
            error="Request failed"
        ), 500

#    @app.route('/', defaults={'path': 'index.html'})
#    @app.route('/<path:path>')
#    def client_all(path):
#        return flask.send_from_directory('client/dist', path)

    log.debug('Flask url map: %s', str(app.url_map))
    log.debug('Flask config map: %s', str(app.config))
    log.debug('Flask running in %s mode',
              'development' if app.config['DEBUG'] else 'production')

    log.debug('Flask app created.')
    return app
