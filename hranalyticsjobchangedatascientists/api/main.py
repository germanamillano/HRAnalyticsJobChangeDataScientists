
import os
import sys

from fastapi import FastAPI
from starlette.responses import JSONResponse

from hranalyticsjobchangedatascientists.api.model.model import hranalyticsjob
from hranalyticsjobchangedatascientists.hranalyticsjobchangedatascientists.predictor.predict import \
    ModelPredictor

# from hranalyticsjobchangedatascientists.hranalyticsjobchangedatascientists.logs.MyLogger import \
#     MyLogger
# from predictor.predict import \
#     ModelPredictor


# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

relative_path = "hranalyticsjobchangedatascientists/models"
model_path = os.path.join(os.path.abspath(parent_dir), relative_path)

app = FastAPI()

# logger = MyLogger.__call__().get_logger()


"""
PARAMETER VALUES
Values are required after de endpoint.
"""


@app.get('/', status_code=200)
async def healthcheck():
    # logger.info("API services active")
    return 'hr analytics job change decision tree is ready to go!'


@app.post('/predict')
def predictor(item: hranalyticsjob):
    # logger.info("The predictor was called for one prediction")
    predictor = ModelPredictor(model_path + "/decision_tree_output.pkl")
    X = [
        item.city,
        item.gender,
        item.relevent_experience,
        item.enrolled_university,
        item.education_level,
        item.major_discipline,
        item.experience,
        item.company_size,
        item.company_type,
        item.last_new_job,
        item.training_hours
    ]
    prediction = predictor.predict([X])

    # logger.info("The prediction was done")
    # logger.debug(f"Resultado predicción: {prediction}")

    return JSONResponse(f"Resultado predicción: {prediction}")
