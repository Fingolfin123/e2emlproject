import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features, target_feature_name):
        try:
            trainer_model_file_path='artificats\model.joblib'
            pre_proc_obj_path='artificats\pre_proc.joblib'
            
            
            model=load_object(file_path=trainer_model_file_path, unique_name=target_feature_name)
            preprocessor=load_object(file_path=pre_proc_obj_path, unique_name=target_feature_name)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)

            return prediction
        
        except Exception as e:
            CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preapartion_course: str,
        reading_score: int,
        writing_score: int
        ):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preapartion_course=test_preapartion_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preapartion_course": [self.test_preapartion_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],                
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            CustomException(e,sys)