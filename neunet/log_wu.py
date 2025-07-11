import logging
import sys
from typing import Optional, Callable


def setup_logging(logfile: Optional[str] = None) -> None:
    # set logging to be printed in terminal during run
    stream_handler = logging.StreamHandler(sys.stdout)
    if logfile:
        # also send logs to a file
        handlers_tuple = (stream_handler, logging.FileHandler(logfile))
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s [%(filename)s::%(funcName)s -- Line %(lineno)d]",
            datefmt="%Y/%m/%d %H:%M:%S",
            handlers=handlers_tuple,
        )
    else:
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s [%(filename)s::%(funcName)s -- Line %(lineno)d]",
            datefmt="%Y/%m/%d %H:%M:%S",
            handlers=[stream_handler],
        )


def run_with_logging(fnc: Callable, *args, logfile=None) -> None:
    """

    Args:
        fnc: function you wish to call with logging
        *args: arguments you wish to pass to the function
        logfile: optional save path for the log file

    Returns: None

    Runs the provide function with logging.  This included the added functionality of logging any Error message in
    the event of an exception.

    """
    try:
        fnc(*args)
    except Exception as e:
        logging.exception(e)
    finally:
        if logfile:
            logging.info(f"Log available at {logfile}")
