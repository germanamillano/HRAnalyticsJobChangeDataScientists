
import pytest
import tests_funciones as test_f

TRAINED_MODEL_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/models/'
TRAINED_DATA_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/data/'
PIPELINE_NAME = 'decision_tree'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'
DATA_TRAIN_CSV = "aug_train.csv"
DATA_TEST_CSV = "aug_test.csv"


def test_train_csv_file_existence():
    """
    Test case to check if the data csv file exists.
    """
    # logger.info("Test case to check if the data csv file exists")

    file_exists = test_f.trained_model_exist(TRAINED_DATA_DIR, DATA_TRAIN_CSV)

    assert file_exists is True, f"The train CSV file at does not exist."


def test_test_csv_file_existence():
    """
    Test case to check if the data csv file exists.
    """
    # logger.info("Test case to check if the data csv file exists")

    file_exists = test_f.trained_model_exist(TRAINED_DATA_DIR, DATA_TEST_CSV)

    assert file_exists is True, f"The test CSV file at does not exist."


def test_trained_model_existence():
    """
    Test case to check if the trained model exists.
    """
    # logger.info("Test case to check if the trained model exists")

    # file_exists = os.path.isfile(save_path)
    file_exists = test_f.data_exists(TRAINED_MODEL_DIR, PIPELINE_SAVE_FILE)

    assert file_exists is True, f"The trained model does not exist."


if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])
