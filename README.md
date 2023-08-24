# HRAnalyticsJobChangeDataScientists
Creation of the Repository for the Integrator Project: HR Analytics: Job Change of Data Scientists - Predict who will move to a new job -


### Context

A company which is active in Big Data and Data Science wants to hire data scientists among people who successfully pass some courses which conduct by the company. Many people signup for their training. Company wants to know which of these candidates are really wants to work for the company after training or looking for a new employment because it helps to reduce the cost and time as well as the quality of training or planning the courses and categorization of candidates. Information related to demographics, education, experience are in hands from candidates signup and enrollment.

This dataset designed to understand the factors that lead a person to leave current job for HR researches too. By model(s) that uses the current credentials,demographics,experience data you will predict the probability of a candidate to look for a new job or will work for the company, as well as interpreting affected factors on employee decision.

The whole data divided to train and test . Target isn't included in test but the test target values data file is in hands for related tasks. A sample submission correspond to enrollee_id of test set provided too with columns : enrollee _id , target

Note:

- The dataset is imbalanced.

- Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.

- Missing imputation can be a part of your pipeline as well.

URL: https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists?datasetId=1019790&language=Python&select=aug_train.csv

### Scope

With this project we seek to apply all the knowledge acquired in class, seeking to make an implementation of MLOps. Within the scope that we seek to apply is the following:

- Include virtual environments: Creation and implementation of an environment for Python.
- Continued use of GitHub: Continued use of Git with documented commits and incremental code.
- Unit tests: Generation of unit tests that allow to speed up the Test.
- Pre-commits: Generation of the Pre-commit process allowing alerts when rules of standards and good practices are not met
- Refactoring: Refactoring of a code to a folder structure, functionalities, classes and functions that allow to have a more reusable and scalable code.
- Linting and formatting: Automatic formatting through the use of Pre.commits that help to refactor code.
- Directory structure: Use of directory structures where we can have classes with main responsibilities.
- OOP (Classes, methods, transformers, pipelines): Use of object-oriented programming, and not code in a single notebook that cannot be reused and structured.


# Virtual Environments
Follow the instructions below to do the Environments.

### Instructions
1. Clone the project `https://github.com/germanamillano/HRAnalyticsJobChangeDataScientists` on your local computer.
2. Create a virtual environment with `Python 3.10.9`
    * Create venv
        ```
        python3.10 -m venv venv
        ```

    * Activate the virtual environment

        ```
        source venv/bin/activate
        ```

3. Install libraries
    Run the following command to install the other libraries.

    ```bash
    pip install -r '/Users/mac/Documents/GitHub/HRAnalyticsJobChangeDataScientists/requirements-310txt'
    ```
    Verify the installation with this command:
    ```bash
    pip freeze
    ```
    Output:

4. Install Pre-commit in Terminal
    ```
    pre-commit install   
    ```
    
3. Open the `hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists/hranalyticsjobchangedatascientists.py` notebook and click on `Run All`. 
    > **IMPORTANT!**  
    Do not forget to select the Python 3.10.9 kernel you have already created.

**Congrats, the notebook is running in a virtual environment with Python 3.10!**

## Continuous use of GitHub

* GitHub was used continuosly during the development of this project, increasing graduately the content of the repository.
  * [HRAnalyticsJobChangeDataScientists/commits](https://github.com/germanamillano/HRAnalyticsJobChangeDataScientists/commits/main)

# Pre-commits

* The Pre-commit functionality was implemented, within this functionality the following validations are being made:
  * isort
  * autoflake
  * autopep8
  * flake8
    * _Every one has its own hooks to represent specific checks on the code._
    * _The corresponding libraries are contained inside requirements-310.txt file. They may be installed but nothing will happen if .yaml file does not exist or is empty, or pre-commit has not been initialized on the project for the first time._
* The configuration file is: .pre-commit-config.yaml

# Setup pre-commits

* Open your terminal, navigate to the root directory of your project
* Install pre.commitPre-commit for the firs time use:
    ```bash
    pip install pre-commit
    ```
* After creating the .pre-commit-config.yaml file, initialize pre-commit for the project:
  ```bash
  pre-commit install
  ```
* With the pre-commit hooks installed, you can now make changes to your Python code. When you're ready to commit your changes, run the following command to trigger the pre-commit checks:
  ```bash
  git commit -m "add pre-commit file"
  ```
* If every check "passed", then you are ready to upload your changes to the repository at GitHub.
  ```bash
  git push
  ```
# Refactorization

* Folders with refactorized code is found in the following directory structure of this project ([repository](https://github.com/JDEQ413/mlops_project)).
  * api
  * docs
  * hranalyticsjobchangedatascientists
    * data
    * load
    * logs
    * models
    * predictor
    * preprocess
    * tests
    * train
  *

 * All the code separated in modules and classes can be executed in the terminal
   * Change the directory to "mlops_project" folder
   * If not active, activate virtual environment
     ```bash
     source venv/bin/activate
     ```
     * Windows cmd:
       ```bash
       venv310\scripts\activate.bat
       ```
   * Run the following:
     ```bash
     python mlops_project\mlops_project.py
     ```