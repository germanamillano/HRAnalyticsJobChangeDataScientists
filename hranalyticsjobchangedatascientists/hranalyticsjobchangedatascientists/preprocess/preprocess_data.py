
import pandas as pd
from logs.MyLogger import MyLogger
from sklearn.base import BaseEstimator, TransformerMixin

logger = MyLogger.__call__().get_logger()


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to impute missing values in categorical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to impute missing values for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to impute missing values for.

    Methods:
        fit(X, y=None):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Imputes missing values in the specified categorical variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    imputer = CategoricalImputer(variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('imputer', imputer),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    # Class Builder

    def __init__(self, variables=None):
        """
        Initialize the CategoricalImputer transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to impute missing values for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        logger.debug("Initialize the CategoricalImputer transformer")

        self.variables = [variables] if not isinstance(
            variables, list) else variables

    # This method does not perform any actual training or fitting, as imputation is based on data.
    # It returns the transformer instance itself.
    def fit(self, X, y=None):
        """
        This method does not perform any actual training or fitting, as imputation is based on data.
        It returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed. Not used in this method.
            y (pd.Series or np.array, optional): Target variable. Not used in this method.

        Returns:
            self (CategoricalImputer): The transformer instance.
        """
        return self

    # Imputes missing values in the specified categorical variables and
    # returns the modified DataFrame.
    def transform(self, X):
        """
        Imputes missing values in the specified categorical variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with missing values imputed for the specified categorical variables.
        """
        logger.debug(
            "[transform(self, X)] - Imputes missing values in the specified categorical variables and returns the modified DataFrame")

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to impute missing values in numerical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to impute missing values for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to impute missing values for.
        median_dict_ (dict): Dictionary to store the median values for each specified numerical variable during fitting.

    Methods:
        fit(X, y=None):
            Calculates the median values for the specified numerical variables from the training data.
            It returns the transformer instance itself.

        transform(X):
            Imputes missing values in the specified numerical variables using the median values and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    imputer = NumericalImputer(variables=['age', 'income'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('imputer', imputer),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self, variables=None):
        """
        Initialize the NumericalImputer transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to impute missing values for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        logger.debug("Initialize the NumericalImputer transformer")

        self.variables = [variables] if not isinstance(
            variables, list) else variables

    def fit(self, X, y=None):
        """
        Calculates the median values for the specified numerical variables from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (NumericalImputer): The transformer instance.
        """
        self.median_dict = {}
        for var in self.variables:
            self.median_dict[var] = X[var].median()
        return self

    def transform(self, X):
        """
        Imputes missing values in the specified numerical variables using the median values and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with missing values imputed for the specified numerical variables.
        """
        logger.debug(
            "[transform(self, X)] - Imputes missing values in the specified numerical variables " +
            "using the median values and returns the modified DataFrame")

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(-1)
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform one-hot encoding for categorical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to perform one-hot encoding for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to perform one-hot encoding for.
        dummies (list): List of column names representing the one-hot encoded dummy variables.

    Methods:
        fit(X, y=None):
            Calculates the one-hot encoded dummy variable columns for the specified categorical variables from the training data.
            It returns the transformer instance itself.

        transform(X):
            Performs one-hot encoding for the specified categorical variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    encoder = OneHotEncoder(variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('encoder', encoder),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self, variables=None):
        """
        Initialize the OneHotEncoder transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to perform one-hot encoding for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.variables = [variables] if not isinstance(
            variables, list) else variables

    def fit(self, X, y=None):
        """
        Calculates the one-hot encoded dummy variable columns for the specified categorical variables from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (OneHotEncoder): The transformer instance.
        """
        self.dummies = pd.get_dummies(
            X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        """
        Performs one-hot encoding for the specified categorical variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with one-hot encoded dummy variables for the specified categorical variables.
        """
        X = X.copy()
        X = pd.concat(
            [X, pd.get_dummies(X[self.variables], drop_first=True)], axis=1)
        X.drop(self.variables, axis=1)

        # Adding missing dummies, if any
        missing_dummies = [var for var in self.dummies if var not in X.columns]
        if len(missing_dummies) != 0:
            for col in missing_dummies:
                X[col] = 0

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to select specific features (columns) from a DataFrame.

    Parameters:
        feature_names (list or array-like): List of column names to select as features from the input DataFrame.

    Methods:
        fit(X, y=None):
            Placeholder method that returns the transformer instance itself.

        transform(X):
            Selects and returns the specified features (columns) from the input DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Define the feature names to be selected
    selected_features = ['feature1', 'feature2', 'feature3']

    # Instantiate the custom transformer
    feature_selector = FeatureSelector(feature_names=selected_features)

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('feature_selector', feature_selector),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self, feature_names):
        """
        Initialize the FeatureSelector transformer.

        Parameters:
            feature_names (list or array-like): List of column names to select as features from the input DataFrame.
        """
        self.feature_names = feature_names

    def fit(self, X, y=None):
        """
        Placeholder method that returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (FeatureSelector): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Selects and returns the specified features (columns) from the input DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_selected (pd.DataFrame): DataFrame containing only the specified features (columns).
        """
        return X[self.feature_names]
