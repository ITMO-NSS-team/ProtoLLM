import logging
import warnings
from contextlib import contextmanager
from typing import Optional, Generator


class ParsingLogger:
    def __init__(self, silent_errors: bool = False, name: Optional[str] = None):
        name = name or __name__
        self._logger = logging.getLogger(name)
        self._logs: dict[str, list[str]] = {}
        self._silent_errors = silent_errors

    @property
    def logger(self):
        return self._logger

    @property
    def logs(self):
        return self._logs

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    @contextmanager
    def parsing_info_handler(self, file_name: str) -> Generator[None, None, None]:
        try:
            try:
                with warnings.catch_warnings(record=True) as record:
                    warnings.simplefilter("default")
                    yield
            finally:
                for warning in record:
                    warn_msg = f"{warning.message} (in {file_name})"
                    self.warning(warn_msg)
        except Exception as error:
            self.logs[file_name] = self.logs.get(file_name, [])
            self.logs[file_name].append(str(error))
            err_msg = f"{error} (in {file_name})"
            self.error(err_msg)
            if not self._silent_errors:
                raise
