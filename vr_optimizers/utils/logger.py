import logging
import coloredlogs


class Logger(object):

    log_format = '[%(asctime)s] (%(process)s) {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
    log_level = None

    @classmethod
    def setup_logging(cls, loglevel='INFO', logfile=""):
        cls.registered_loggers = dict()
        cls.log_level = loglevel
        numeric_level = getattr(logging, loglevel.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % loglevel)
        if logfile:
            logging.basicConfig(handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
                                level=numeric_level,
                                format=cls.log_format,
                                datefmt='%Y-%m-%d %H:%M:%S',)
        else:
            logging.basicConfig(level=numeric_level,
                                format=cls.log_format,
                                datefmt='%Y-%m-%d %H:%M:%S',)

    @classmethod
    def get(cls, logger_name='default'):
        if logger_name in cls.registered_loggers:
            return cls.registered_loggers[logger_name]
        else:
            return cls(logger_name)

    def __init__(self, logger_name='default'):
        if logger_name in self.registered_loggers:
            raise ValueError(f"Logger {logger_name} already exists. Call with Logger.get(\"{logger_name}\")")
        else:
            self.name = logger_name
            self.logger = logging.getLogger(self.name)
            self.registered_loggers[self.name] = self.logger
            coloredlogs.install(
                level=self.log_level,
                logger=self.logger,
                fmt=self.log_format,
                datefmt='%Y-%m-%d %H:%M:%S')

    def log(self, loglevel, msg):
        loglevel = loglevel.upper()
        if loglevel == 'DEBUG':
            self.logger.debug(msg)
        elif loglevel == 'INFO':
            self.logger.info(msg)
        elif loglevel == 'WARNING':
            self.logger.warning(msg)
        elif loglevel == 'ERROR':
            self.logger.error(msg)
        elif loglevel == 'CRITICAL':
            self.logger.critical(msg)

    def debug(self, msg):
        self.log('debug', msg)

    def info(self, msg):
        self.log('info', msg)

    def warning(self, msg):
        self.log('warning', msg)

    def error(self, msg):
        self.log('error', msg)

    def critical(self, msg):
        self.log('critical', msg)
