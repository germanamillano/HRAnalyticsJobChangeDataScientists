from pydantic import BaseModel


class hranalyticsjob(BaseModel):
    """
    Represents a passenger on the Titanic with various attributes.

    Attributes:
        crim (float): per capita crime rate by town
        zn (float): proportion of residential land zoned for lots over 25,000 sq.ft.
        indus (float): proportion of non-retail business acres per town
        chas (int): Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        nox (float): nitric oxides concentration (parts per 10 million) [parts/10M]
        rm (float): average number of rooms per dwelling
        age (float): proportion of owner-occupied units built prior to 1940
        dis (float): weighted distances to five Boston employment centres
        rad (int): index of accessibility to radial highways
        tax (float): full-value property-tax rate per $10,000 [$/10k]
        ptration (float): pupil-teacher ratio by town
        b (float): The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        lstat (float): % lower status of the population
    """

    city: int
    gender: float
    relevent_experience: int
    enrolled_university: float
    education_level: float
    major_discipline: float
    experience: float
    company_size: float
    company_type: float
    last_new_job: float
    training_hours: int
