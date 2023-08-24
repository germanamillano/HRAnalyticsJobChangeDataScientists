import os

# from hranalyticsjobchangedatascientists.hranalyticsjobchangedatascientists.logs.MyLogger import MyLogger
# logger = MyLogger.__call__().get_logger()

# Function for check if the file are in path


def data_exists(path: str, filename: str) -> bool:
    # logger.info("Checking path an file")

    if os.path.isfile(path + filename):
        return True
    else:
        return False

# Function for check if the model file are in path


def trained_model_exist(path: str, filename: str) -> bool:
    # logger.info("Checking path an modelfile")

    if os.path.isfile(path + filename):
        return True
    else:
        return False
