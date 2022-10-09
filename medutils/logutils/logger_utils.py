import os
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler


class MedLogger:
    def __init__(self, log_name, log_dir):
        self.log_name = log_name
        self.log_dir = log_dir

    def init_logger(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if self.log_name not in Logger.manager.loggerDict:
            logger = logging.getLogger(self.log_name)
            logger.setLevel(logging.DEBUG)
            handler = TimedRotatingFileHandler(filename=os.path.join(self.log_dir, "%s.log" % self.log_name), when='D',
                                               backupCount=30)
            datefmt = '%Y-%m-%d %H:%M:%S'
            format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
            formatter = logging.Formatter(format_str, datefmt)
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            logger.addHandler(console)

            handler = TimedRotatingFileHandler(filename=os.path.join(self.log_dir, "ERROR.log"), when='D',
                                               backupCount=30)
            datefmt = '%Y-%m-%d %H:%M:%S'
            format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
            formatter = logging.Formatter(format_str, datefmt)
            handler.setFormatter(formatter)
            handler.setLevel(logging.ERROR)
            logger.addHandler(handler)
        logger = logging.getLogger(self.log_name)
        return logger


if __name__ == "__main__":
    log = MedLogger('test', 'd:/tets').init_logger()
    log.info("test")
    log.error("test2")
    log.debug("debug")
    log.warning("warning")
