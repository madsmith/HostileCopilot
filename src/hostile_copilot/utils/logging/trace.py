import logging

logging.TRACE = logging.DEBUG // 2
logging.addLevelName(logging.TRACE, "TRACE")

class TraceLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        self.log(logging.TRACE, msg, *args, **kwargs)