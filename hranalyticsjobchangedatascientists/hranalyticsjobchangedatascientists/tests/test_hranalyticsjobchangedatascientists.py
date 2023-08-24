#!/usr/bin/env python

"""Tests for `hranalyticsjobchangedatascientists` package."""

import pytest
import tests_funciones as test_f

TRAINED_MODEL_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/models/'
TRAINED_DATA_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/data/'
PIPELINE_NAME = 'decision_tree'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'
DATA_TRAIN_CSV = "aug_train.csv"
DATA_TEST_CSV = "aug_test.csv"


def return_dir(filetype: str):
    match filetype:
        case "test":
            return [(TRAINED_DATA_DIR, DATA_TEST_CSV)]

        case "train":
            return [(TRAINED_DATA_DIR, DATA_TRAIN_CSV)]

        case "model":
            return [(TRAINED_MODEL_DIR, PIPELINE_SAVE_FILE)]

        case "all":
            return [(TRAINED_MODEL_DIR, PIPELINE_SAVE_FILE), (TRAINED_DATA_DIR,
                                                              DATA_TEST_CSV), (TRAINED_DATA_DIR, DATA_TRAIN_CSV)]


@pytest.mark.parametrize('path, filename', return_dir("test"))
def test_test_data_exists(path, filename):
    assert test_f.data_exists(path, filename) is True


@pytest.mark.parametrize('path, filename', return_dir("train"))
def test_train_data_exists(path, filename):
    assert test_f.data_exists(path, filename) is True


@pytest.mark.parametrize('path, filename', return_dir("model"))
def test_model_data_exists(path, filename):
    assert test_f.data_exists(path, filename) is True


@pytest.mark.parametrize('path, filename', return_dir("all"))
def test_files_data_exists(path, filename):
    assert test_f.data_exists(path, filename) is True


if __name__ == "__main__":
    # Run the test function using Pytest
    pytest.main([__file__])
