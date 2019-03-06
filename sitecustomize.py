"""
Log specified Python imports to a file, remote machine or stderr.

What and how import are logged are controlled by the following environment
variables:

    * PYIMPORT_LOGFILE : Filename to log imports.
    * PYIMPORT_HOST : Host to send UDP logging messages.
    * PYIMPORT_PORT : Port to send UDP logging messages.
    * PYIMPORT_LIBLIST : A comma separated list of import names to log.
    * PYIMPORT_MSGFORMAT : Format of logging message, {username} and
        {import_name} will be expanded.
    * PYIMPORT_DEBUG : Set to 1 to print imports to stderr.

"""

import getpass
import logging
import os
import sys
from logging import FileHandler, StreamHandler
from logging.handlers import DatagramHandler

IS_DEBUG = os.environ.get('PYIMPORT_DEBUG')
DEBUG = bool(IS_DEBUG)

LOGFILE = os.environ.get('PYIMPORT_LOGFILE')

HOST = os.environ.get('PYIMPORT_HOST')
PORT = os.environ.get('PYIMPORT_PORT')

LIBLIST = os.environ.get('PYIMPORT_LIBLIST', '')
MODULES_TO_LOG = LIBLIST.split(',')

MSG_FORMAT = os.environ.get('PYIMPORT_MSGFORMAT', '{import_name}')

USERNAME = getpass.getuser()


class ImportLogger(object):
    """ Log imports to strerr, a file or datagram. """

    def __init__(self):
        logger = logging.getLogger('importlogger')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        if DEBUG:
            handler = StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if LOGFILE is not None:
            handler = FileHandler(filename=LOGFILE)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if HOST is not None and PORT is not None:
            handler = DatagramHandler(host=HOST, port=int(PORT))
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        self.logger = logger

    # used in python>=3.4
    def find_spec(self, fullname, path=None, *args, **kwargs):
        if path is None:
            if fullname in MODULES_TO_LOG:
                import_name = fullname
                message = MSG_FORMAT.format(
                    import_name=import_name,
                    username=USERNAME,
                )
                self.logger.info(message)
        return None

    # find module defined for <=3.4
    find_module = find_spec


# register the import logger
sys.meta_path.insert(0, ImportLogger())
