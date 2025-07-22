from flask import Flask, request, render_template
import os
from src.utils import GetProcessorObj
from src.pipeline.model_handler import ModelRunHandle
from src.components.data_transformation import DataTransformation
from src.pipeline.predict_pipeline import PredictionSelectionData, PredictPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils import load_object




# NOTE: FUTURE PIPELINE UPGRADE --> feature_options should only need to be saved once so 
# get_raw_features should not be called multiple times
data_transformation=DataTransformation()
feature_options = data_transformation.get_raw_features()

# selected_feature = request.form.get('features')
# selected_feature = selected_feature.replace(" (numerical)", "").replace(" (categorical)", "")
target_feature_name = "math_score"
print(f"the selected feature is: {target_feature_name}")
selected_feature = target_feature_name

training_pipeline = TrainingPipeline()
results = training_pipeline.training(selected_feature)
print(f"the results are: {results}")


# NOTE: FUTURE PIPELINE UPGRADE --> need to know which model to use in predictions
# currently have to hardcode "target_feature_name"
model_path = os.path.join('artifacts', 'model_Linear Regression.joblib')
preproc_path = os.path.join('artifacts', 'pre_proc.joblib')
print(f"the model_path is: {model_path}")
print(f"the preproc_path is: {preproc_path}")
predict_pipeline = PredictPipeline(model_path, preproc_path)

model=load_object(file_path=model_path, unique_name=target_feature_name)
preprocessor=load_object(file_path=preproc_path, unique_name=target_feature_name)


import pandas as pd

data = {
    "gender": ["female"],
    "race_ethnicity": ["group A"],
    "parental_level_of_education": ["associate's degree"],
    "lunch": ["free/reduced"],
    "test_preparation_course": ["completed"],
    "reading_score": [87],
    "writing_score": [78]
}

features = pd.DataFrame(data)
print(f"the selected features are: {features}")
data_scaled=preprocessor.transform(features)
prediction=model.predict(data_scaled)
print(f"the sprediction is: {prediction}")

# Load preprocessor and get options
pre_processor_obj = GetProcessorObj(
    proc_obj_path=preproc_path,
    target_feature_name="math_score"
)
print(pre_processor_obj.feature_options)


# data = PredictionSelectionData(
#     gender="female",
#     race_ethnicity="group A",
#     parental_level_of_education="associate's degree",
#     lunch="free/reduced",
#     test_preparation_course="completed",
#     reading_score=87,
#     writing_score=78,
# )

# pred_df = data.get_data_as_data_frame()
# print(pred_df)
# results = predict_pipeline.predict(pred_df, 'math_score')
# print(results)

