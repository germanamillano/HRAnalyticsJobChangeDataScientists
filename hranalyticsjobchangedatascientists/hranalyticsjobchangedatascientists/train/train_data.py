from logs.MyLogger import MyLogger
from preprocess.preprocess_data import (CategoricalImputer, FeatureSelector,
                                        NumericalImputer)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

logger = MyLogger.__call__().get_logger()


class hranalyticsjobDataPipeline:
    """
    A class representing data processing and modeling pipeline.

    Attributes:
        NUMERICAL_VARS (list): A list of numerical variables in the dataset.
        CATEGORICAL_VARS_WITH_NA (list): A list of categorical variables with missing values.
        NUMERICAL_VARS_WITH_NA (list): A list of numerical variables with missing values.
        CATEGORICAL_VARS (list): A list of categorical variables in the dataset.
        SEED_MODEL (int): A seed value for reproducibility.

    Methods:
        create_pipeline(): Create and return the Titanic data processing pipeline.
    """

    # Class Builder
    def __init__(self, seed_model, numerical_vars, categorical_vars_with_na,
                 numerical_vars_with_na, categorical_vars, selected_features):
        self.SEED_MODEL = seed_model
        self.NUMERICAL_VARS = numerical_vars
        self.CATEGORICAL_VARS_WITH_NA = categorical_vars_with_na
        self.NUMERICAL_VARS_WITH_NA = numerical_vars_with_na
        self.CATEGORICAL_VARS = categorical_vars
        self.SEED_MODEL = seed_model
        self.SELECTED_FEATURES = selected_features

    # Create and return the Titanic data processing pipeline.

    def create_pipeline(self):
        """
        Create and return the data processing pipeline.

        Returns:
            Pipeline: A scikit-learn pipeline for data processing and modeling.
        """

        logger.debug(
            "[create_pipeline(self)] - Create and return the data processing pipeline")

        self.PIPELINE = Pipeline(
            [
                ('categorical_imputer', CategoricalImputer(
                    variables=self.CATEGORICAL_VARS_WITH_NA)), ('median_imputation', NumericalImputer(
                        variables=self.NUMERICAL_VARS_WITH_NA)), ('feature_selector', FeatureSelector(
                            self.SELECTED_FEATURES)), ])

        return self.PIPELINE

    # Fit a Logistic Regression model using the predefined data preprocessing
    # pipeline
    def fit_decision_tree(self, X_train, y_train):
        """
        Fit a Logistic Regression model using the predefined data preprocessing pipeline.

        Parameters:
        - X_train (pandas.DataFrame or numpy.ndarray): The training input data.
        - y_train (pandas.Series or numpy.ndarray): The target values for training.

        Returns:
        - logistic_regression_model (LogisticRegression): The fitted Logistic Regression model.
        """
        logger.debug(
            "[fit_decision_tree(self, X_train, y_train)] - Fit a Logistic Regression model using the predefined data preprocessing pipeline")

        tree_clf = DecisionTreeClassifier(random_state=17)

        pipeline = self.create_pipeline()
        pipeline.fit(X_train, y_train)

        X_train = pipeline.transform(X_train)
        tree_clf.fit(X_train, y_train)

        print('Train:', tree_clf.score(X_train, y_train))

        return tree_clf

    # Apply the data preprocessing pipeline on the test data
    def transform_test_data(self, X_test):
        """
        Apply the data preprocessing pipeline on the test data.

        Parameters:
        - X_test (pandas.DataFrame or numpy.ndarray): The test input data.

        Returns:
        - transformed_data (pandas.DataFrame or numpy.ndarray): The preprocessed test data.
        """
        logger.debug(
            "[transform_test_data(self, X_test)] - Apply the data preprocessing pipeline on the test data")

        pipeline = self.create_pipeline()
        return pipeline.transform(X_test)
