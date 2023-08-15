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

### Other Documents

#### Enviroment Creation

HRAnalyticsJobChangeDataScientists/Environment.md

#### Data Documentation

hranalyticsjobchangedatascientists/data/INFO.md
