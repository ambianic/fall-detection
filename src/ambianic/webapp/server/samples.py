"""REST API for timeline events from pipeline samples."""
import logging
import uuid
from datetime import datetime
from pathlib import Path
import os
import json
import yaml

from ambianic import DEFAULT_DATA_DIR

log = logging.getLogger()

SAMPLES = [
    {
        'file': '20190913-063945-json.txt',
        'id': '2a34987234324324',
        "datetime": "2019-09-13T16:32:34.704797",
        "image": "20190913-163234-image.jpg",
        "inference_result": [
            {
                "category": "person",
                "confidence": 0.98046875,
                "box": {
                    "xmin": 0.5251423732654468,
                    "ymin": 0.0021262094378471375,
                    "xmax": 0.9498447340887946,
                    "ymax": 0.23079824447631836
                }
            }
            ]
    },
    {
        'file': '20190913-064945-json.txt',
        'id': '3ea34987234345424',
        "datetime": "2019-09-15T06:38:30.019847",
        "image": "20190915-063830-image.jpg",
        "inference_result": []
    },
    {
        'file': '20190913-064945-json.txt',
        'id': '2c349bb74234324324',
        "datetime": "2019-09-12T15:20:47.550151",
        "image": "20190912-152047-image.jpg",
        "inference_result": [
            {
                "category": "person",
                "confidence": 0.9921875,
                "box": {
                    "xmin": 0.349978506565094,
                    "ymin": 0.09689526346356389,
                    "xmax": 0.5911635756492615,
                    "ymax": 0.40339951629117893
                }
            }
        ]
    },
]


def get_samples(before_datetime=None, page=1):
    """Get stored pipeline samples.

    Parameters
    ----------
    before_datetime : date time in ISO 8601 compatible format,
        YYYY-MM-DDTHH:MM:SS. For example '2002-12-25 00:00:00-06:39'.
        It uses python's standard function datetime.fromisoformat().
        If not provided, the function will start with the most recent available
        sample.
    page : positive integer
        Paginates samples in batches of 5. Defaults to page=1.

    Returns
    -------
    dictionary
        Returns a dictionary of previously saved pipeline samples.

    """
    parsed_datetime = None
    assert isinstance(page, int)
    assert page > 0
    page_size = 5
    if before_datetime:
        try:
            parsed_datetime = datetime.fromisoformat(before_datetime)
            log.debug('Fetching samples saved before %s',
                      parsed_datetime)
        except ValueError as e:
            log.warning('Unable to parse before_datetime parameter: %s. '
                        ' Error: %s', before_datetime, str(e))
    page_start_position = (page-1)*page_size
    page_end_position = page_start_position + page_size
    if not parsed_datetime:
        log.debug('Fetching most recent saved samples')
    log.debug('Fetching samples page %d. Page size %d. '
              'Sample index range [%d:%d]. ',
              page, page_size, page_start_position, page_end_position)
    p = Path(os.path.join(DEFAULT_DATA_DIR, 'detections/front-door/faces'))
    log.debug('Samples path: %s', p.resolve())
    files = list(p.glob("*-json.txt"))
    log.debug('Fetched %d file names.', len(files))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    samples = []
    for json_file in files[page_start_position:page_end_position]:
        with open(json_file) as f:
            sample = json.load(f)
            sample['id'] = uuid.uuid4().hex
            sample['file'] = str(json_file)
            samples.append(sample)
    # lines = map(str, files)
    # log.debug('File names follow:\n %s', "\n".join(lines))
    return samples


def remove_timeline(file_path):
    try:
        os.remove(file_path)
    except Exception:
        logging.exception("Error removing %s" % file_path)

def get_timeline(before_datetime=None, page=1, data_dir=None):
    """Get stored pipeline timeline events.

    :Parameters:
    ----------
    before_datetime : date time in ISO 8601 compatible format,
        YYYY-MM-DDTHH:MM:SS. For example '2002-12-25 00:00:00-06:39'.
        It uses python's standard function datetime.fromisoformat().
        If not provided, the function will start with the most recent available
        sample.
    page : positive integer
        Paginates samples in batches of 5. Defaults to page=1.

    :Returns:
    -------
    list: json
        Returns a list of previously saved pipeline events.

    """

    if data_dir is None or not os.path.exists(data_dir):
        log.warning("data_dir is not valid: %s", data_dir)
        return []

    parsed_datetime = None
    assert isinstance(page, int)
    assert page > 0
    page_size = 5
    if before_datetime:
        try:
            parsed_datetime = datetime.fromisoformat(before_datetime)
            log.debug('Fetching samples saved before %s',
                      parsed_datetime)
        except ValueError as e:
            log.warning('Unable to parse before_datetime parameter: %s. '
                        ' Error: %s', before_datetime, str(e))
    page_start_position = (page-1)*page_size
    page_end_position = page_start_position + page_size

    if not parsed_datetime:
        log.debug('Fetching most recent saved samples')
    log.debug('Fetching samples page %d. Page size %d. '
              'Sample index range [%d:%d]. ',
              page, page_size, page_start_position, page_end_position)

    files = list(Path(data_dir).glob('./timeline-event-log.yaml*'))
    files = sorted(files, reverse=False)

    page_count = 1
    events_queue = []

    # load the event history, older first
    for file_path in files:
        with file_path.open() as pf:

            try:
                timeline_events = yaml.safe_load(pf)
                timeline_events += events_queue
            except (
                yaml.reader.ReaderError,
                yaml.scanner.ScannerError,
                yaml.composer.ComposerError,
                yaml.constructor.ConstructorError
            ):
                log.exception("Detected unreadable timeline, removing %s" % file_path)
                remove_timeline(file_path)
                continue

            events_queue = []
            events_len = len(timeline_events)
            if events_len < page_end_position:

                pages_mod = events_len % page_size
                if pages_mod > 0:
                    events_queue = timeline_events[0:pages_mod]
                    page_start_position += pages_mod
                    page_end_position += pages_mod

                page_start_position -= events_len
                page_end_position -= events_len
                page_count += 1

            else:

                if page_start_position >= events_len:
                    return []

                # events are appended to the file as they arrive
                # we need to read in reverse order to get the latest one first

                return \
                    timeline_events[-1*page_start_position - 1:
                                    -1*page_end_position - 1:
                                    -1]

    page_count += 1

    if page_count < page:
        return []

    # return the remaining queue if there are no more files to process
    return events_queue


def add_sample(new_sample=None):
    assert new_sample
    log.debug('add_sample new_sample 0 %s', new_sample)
    new_sample['id'] = uuid.uuid4().hex
    log.debug('add_sample 1 %s', new_sample)
    SAMPLES.append(new_sample)


def update_sample(edited_sample=None):
    assert edited_sample
    for i, old_sample in enumerate(SAMPLES):
        old_sample = SAMPLES[i]
        if old_sample['id'] == edited_sample['id']:
            SAMPLES[i] = edited_sample
            return True
    log.debug('sample not found %s', edited_sample)
    return False


def delete_sample(sample_id):
    sample = None
    for sample in SAMPLES:
        if sample['id'] == sample_id:
            SAMPLES.remove(sample)
            log.debug('sample deleted %s', sample)
            return True
    log.debug('sample not found %s', sample)
    return False
