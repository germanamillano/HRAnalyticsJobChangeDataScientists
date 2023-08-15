"""Main module."""

# SETUP

import joblib
from load.load_data import DataRetriever
from sklearn.metrics import accuracy_score, roc_auc_score
from train.train_data import hranalyticsjobDataPipeline

# CONSTANTS
DATASETS_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/data/'
DATASETS_TRAIN = 'aug_train.csv'
DATASETS_TEST = 'aug_test.csv'

SEED_MODEL = 404

NUMERICAL_VARS = [
    'city',
    'gender',
    'relevent_experience',
    'enrolled_university',
    'education_level',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job',
    'training_hours']
CATEGORICAL_VARS = []

NUMERICAL_VARS_WITH_NA = [
    'gender',
    'enrolled_university',
    'education_level',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job']
CATEGORICAL_VARS_WITH_NA = []

SELECTED_FEATURES = [
    'city',
    'gender',
    'relevent_experience',
    'enrolled_university',
    'education_level',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job',
    'training_hours']

TRAINED_MODEL_DIR = 'hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/models/'
PIPELINE_NAME = 'decision_tree'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output.pkl'

# MAIN

if __name__ == "__main__":
    print(DATASETS_DIR + DATASETS_TRAIN)

    # Functions/Data retrieval
    data_retriever = DataRetriever(DATASETS_TRAIN, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    X_train, X_valid, y_train, y_valid = data_retriever.retrieve_train_test_split()

    # Custom Transformers/Pipeline
    hranalyticsjob_Data_Pipeline = hranalyticsjobDataPipeline(
        seed_model=SEED_MODEL,
        numerical_vars=NUMERICAL_VARS,
        categorical_vars_with_na=CATEGORICAL_VARS_WITH_NA,
        numerical_vars_with_na=NUMERICAL_VARS_WITH_NA,
        categorical_vars=CATEGORICAL_VARS,
        selected_features=SELECTED_FEATURES)

    decision_tree_model = hranalyticsjob_Data_Pipeline.fit_decision_tree(
        X_train, y_train)

    X_valid = hranalyticsjob_Data_Pipeline.PIPELINE.fit_transform(X_valid)
    y_pred = decision_tree_model.predict(X_valid)

    # Make predictions
    class_pred = decision_tree_model.predict(X_valid)
    proba_pred = decision_tree_model.predict_proba(X_valid)[:, 1]
    print(f'test roc-auc : {roc_auc_score(y_valid, proba_pred)}')
    print(f'test accuracy: {accuracy_score(y_valid, class_pred)}')

    # Persistence/Szve model
    save_path = TRAINED_MODEL_DIR + PIPELINE_SAVE_FILE
    joblib.dump(decision_tree_model, save_path)
    print(f"Model saved in {save_path}")
