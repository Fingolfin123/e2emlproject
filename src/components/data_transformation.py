import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformConfig:
    pre_proc_obj_path: str=os.path.join('artifacts',"pre_proc.joblib")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformConfig()
        self.data_ingestion=DataIngestion()
        
    
    def get_ingest_data(self):
        # NOTE: PIPELINE FEATURE START
        # if (
        #     not os.path.exists(self.data_ingestion.ingestion_config.raw_data_path) or
        #     not os.path.exists(self.data_ingestion.ingestion_config.train_data_path) or
        #     not os.path.exists(self.data_ingestion.ingestion_config.test_data_path)
        #     ):
        #     print("Raw, Training or Test set not found, regenerating from input source.")
        #     data_ingestor = DataIngestion()
        #     data_ingestor.initiate_data_ingestion()
        # NOTE: PIPELINE FEATURE END
        data_ingestor = DataIngestion()
        data_ingestor.initiate_data_ingestion()        
            
        df_raw = pd.read_csv(self.data_ingestion.ingestion_config.raw_data_path)
        df_train = pd.read_csv(self.data_ingestion.ingestion_config.train_data_path)
        df_test = pd.read_csv(self.data_ingestion.ingestion_config.test_data_path)

        return df_raw,df_train,df_test

    def combine_input_target_arrays(self,input_array,target_array):
        combined_array = np.c_[
            input_array, np.array(target_array)
        ]
        return combined_array

    def split_features(self,df):
        categorical_features = df.select_dtypes(include="object").columns
        numeric_features = df.select_dtypes(exclude="object").columns        

        return categorical_features, numeric_features

    def split_input_X_and_target_y(self,df,target_feature_name):

        # Seperate Model Input (X) and predicted values (y)
        X = df.drop(columns=[target_feature_name],axis=1)
        y = df[target_feature_name]

        return X,y
    
    def get_transformer_obj(self,df,target_feature_name):
        '''
         This function perfroms data transformation and creates preprocessng obbject
        '''
        logging.info('Obtaining Processing Object')
        try:
            # Separate Categorical and Numerical Features
            X,y = self.split_input_X_and_target_y(df, target_feature_name) # remove target field
            categorical_features, numeric_features = self.split_features(X)

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), # replace nulls median val
                    ("scalar",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), # replace nulls mode
                    ("OneHotEncoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )  
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )    
            logging.info(f'Categorical Features: {list(categorical_features)}')
            logging.info(f'Numerical Features: {list(numeric_features)}')
            return preprocessor          
                
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, target_feature_name=None):
        logging.info('Transforming Model Input')
        try:
            # Get ingestion data
            df_raw,df_train,df_test = self.get_ingest_data()

            # Separate input matrix and predicted output vector
            X_train,y_train = self.split_input_X_and_target_y(df_train, target_feature_name)
            X_test,y_test = self.split_input_X_and_target_y(df_test, target_feature_name)

            # Get preprocessir object to fit model for numerical and categorical features
            preprocessor_obj = self.get_transformer_obj(df_raw, target_feature_name)
            X_train_feature = preprocessor_obj.fit_transform(X_train)
            X_test_feature = preprocessor_obj.transform(X_test)

            # Separate input matrix and mredicted output vector into train and test sets
            train_arr = self.combine_input_target_arrays(X_train_feature,y_train)
            test_arr = self.combine_input_target_arrays(X_test_feature,y_test)
            
            logging.info("Input Transformations Completed")

            save_object(
                file_path=self.transformation_config.pre_proc_obj_path,
                unique_name=target_feature_name,
                obj=preprocessor_obj,
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.pre_proc_obj_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataTransformation()
    obj.initiate_data_transformation('math_score')
