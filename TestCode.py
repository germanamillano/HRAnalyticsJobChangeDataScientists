from hranalyticsjobchangedatascientists.hranalyticsjobchangedatascientists.logs.MyLogger import \
    MyLogger

logger = MyLogger.__call__().get_logger()
logger.info("Hello, Logger")
logger.debug("bug occured")
