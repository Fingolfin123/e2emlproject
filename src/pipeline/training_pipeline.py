import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        pass

    def training(self,feature):
        try:
            obj=ModelTrainer()
            training_report = obj.evaluate_features(feature)            
            # trainer_model_file_path='artifacts\model_Linear Regression.joblib'
            # pre_proc_obj_path='artifacts\pre_proc.joblib'
            # model=load_object(file_path=trainer_model_file_path, unique_name=target_feature_name)
            # preprocessor=load_object(file_path=pre_proc_obj_path, unique_name=target_feature_name)
            # data_scaled=preprocessor.transform(features)
            # prediction=model.predict(data_scaled)
            # print(prediction)
            
            return training_report
        
        except Exception as e:
            CustomException(e,sys)

class TrainingSelectionData:
    def __init__(
        self,
        features: str,
        ):
        self.features=features

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "features": [self.features],            
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e,sys)

