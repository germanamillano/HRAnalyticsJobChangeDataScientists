import pandas as pd
from logs.MyLogger import MyLogger
from sklearn.model_selection import train_test_split

logger = MyLogger.__call__().get_logger()


class DataRetriever:
    """
    A class for retrieving data from a CSV and processing it for further analysis.

    Parameters:
        data_name (str): The name from which the data will be loaded.
        data_path (str): The URL from which the data will be loaded.


    Example usage:
    ```
    URL = 'https://www.openml.org/data/get_csv/16826755/phpMYEkMl'
    data_retriever = DataRetriever(DATASETS_TRAIN, DATASETS_DIR)
    result = data_retriever.retrieve_data()
    print(result)
    ```
    """

    # Variables that should be deleted from the data set
    DROP_COLS = ['enrollee_id', 'city_development_index']

    # Target variable for the model
    TARGET = 'target'

    # Numerical values for each of the categorical variables:
    gender_encod = {'Male': 2,
                    'Female': 1,
                    'Other': 0}

    relevent_experience_encod = {'Has relevent experience': 1,
                                 'No relevent experience': 0}
    enrolled_university_encod = {'no_enrollment': 0,
                                 'Part time course': 1,
                                 'Full time course': 2}

    education_level_encod = {'Primary School': 0,
                             'High School': 1,
                             'Graduate': 2,
                             'Masters': 3,
                             'Phd': 4}

    major_discipline_encod = {'STEM': 5,
                              'Business Degree': 4,
                              'Arts': 3,
                              'Humanities': 2,
                              'No Major': 1,
                              'Other': 0}

    last_job_encod = {'never': 0,
                      '1': 1,
                      '2': 2,
                      '3': 3,
                      '4': 4,
                      '>4': 5}

    company_size_encod = {'<10': 0,
                          '10/49': 1,
                          '50-99': 2,
                          '100-500': 3,
                          '500-999': 4,
                          '1000-4999': 5,
                          '5000-9999': 6,
                          '10000+': 7}

    company_type_encod = {'Pvt Ltd': 5,
                          'Funded Startup': 4,
                          'Early Stage Startup': 3,
                          'Public Sector': 2,
                          'NGO': 1,
                          'Other': 0}

    experience_encod = {'<1': 0,
                        '1': 1,
                        '2': 1,
                        '3': 1,
                        '4': 1,
                        '5': 1,
                        '6': 2,
                        '7': 2,
                        '8': 2,
                        '9': 2,
                        '10': 2,
                        '11': 3,
                        '12': 3,
                        '13': 3,
                        '14': 3,
                        '15': 3,
                        '16': 4,
                        '17': 4,
                        '18': 4,
                        '19': 4,
                        '20': 4,
                        '>20': 5}

    # logger = MyLogger.__call__().get_logger()

    # Class Builder
    def __init__(self, data_name, data_path):
        self.data_name = data_name
        self.datasets_dir = data_path

    # Retrieves data from the specified URL, processes it, and stores it in a
    # CSV file
    def retrieve_data(self):
        """
        Retrieves data from the specified URL, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """
        logger.info(
            "[retrieve_data(self)] - Retrieves data from the specified URL, processes it, and stores it in a CSV file")

        # Loading data from specific URL
        self.data = pd.read_csv(self.datasets_dir + self.data_name)

        self.data['gender'] = self.data['gender'].map(self.gender_encod)
        self.data['relevent_experience'] = self.data['relevent_experience'].map(
            self.relevent_experience_encod)
        self.data['enrolled_university'] = self.data['enrolled_university'].map(
            self.enrolled_university_encod)
        self.data['education_level'] = self.data['education_level'].map(
            self.education_level_encod)
        self.data['major_discipline'] = self.data['major_discipline'].map(
            self.major_discipline_encod)
        self.data['experience'] = self.data['experience'].map(
            self.experience_encod)
        self.data['company_size'] = self.data['company_size'].map(
            self.company_size_encod)
        self.data['company_type'] = self.data['company_type'].map(
            self.company_type_encod)
        self.data['last_new_job'] = self.data['last_new_job'].map(
            self.last_job_encod)

        self.data['city'] = self.data['city'].apply(
            lambda x: int(x.replace('city_', '')))

        # Drop irrelevant columns
        self.data.drop(self.DROP_COLS, axis=1, inplace=True)

        return self.data

    # Function that returns a data set in 2 samples for Test and Validation
    def retrieve_train_test_split(self):
        """
        Function that returns a data set in 2 samples for Test and Validation.

        Parameters:

        Example usage:
        ```
        X_train, X_valid, y_train, y_valid = data_retriever.retrieve_train_test_split()
        ```
        """
        logger.info(
            "[retrieve_train_test_split(self)] - Function that returns a data set in 2 samples for Test and Validation.")

        y = self.data[self.TARGET]
        x = self.data.drop(self.TARGET, axis=1)

        return train_test_split(
            x,
            y,
            shuffle=True,
            random_state=1,
            test_size=0.3)
