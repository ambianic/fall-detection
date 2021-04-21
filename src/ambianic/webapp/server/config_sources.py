
from werkzeug.exceptions import NotFound, BadRequest
from ambianic import config
import logging

log = logging.getLogger(__name__)

source_model = {
    "id": str,
    "uri": str,
    "type": str,
    "live": bool
}

source_types = ["video", "audio", "image"]


def validate(source_id, source):
    """Validate input object"""

    if not isinstance(source, dict):
        raise BadRequest(
            "Source should be a valid dictionary of pipeline source objects"
        )

    source["id"] = source_id
    source_keys = source.keys()

    for prop in source_model:
        if prop not in source_keys:
            raise BadRequest(f"missing property {prop}")
        if not isinstance(source[prop], source_model[prop]):
            raise BadRequest(f"invalid type for {prop}")

    if source["type"] not in source_types:
        raise BadRequest(f"type should be one of {source_types}")

    return source


def get(source_id):
    """Retrieve a source by id"""
    log.info("Get source_id=%s", source_id)

    if not source_id:
        raise BadRequest("source id is empy")

    if not isinstance(source_id, str):
        raise BadRequest("source id should be a string")

    source = config.sources[source_id]
    
    if source is None:
        raise NotFound("source not found")

    return source


def remove(source_id):
    """Remove source by id"""
    log.info("Removing source_id=%s", source_id)
    get(source_id)
    del config.sources[source_id]


def save(source_id, source):
    """Save source configuration information"""
    log.info("Saving source_id=%s", source_id)
    source = validate(source_id, source)
    config.sources[source["id"]] = source
    return config.sources[source["id"]]
