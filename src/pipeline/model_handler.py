import os
import sys
# # General App Functionality:
# 1. Select Data Source
# 2. Select Field(s) to Train
# 3. Create Predictions




# # This needs to do the following things:
# 1. Handle Data Source
#     - Ingestion
#         - Dynamic Source Types
#     - Exploration
#     - Posting Exploration Results
#         - Shape
#         - Size
#         - Features
# 2. Handle Data Transformation
#     - Determine "best" model for selected features
#     - Save Model Evaluation Report
#     - Save Top Performing Data Transformation Model
#     - Post Model Evaluation Report
# 3. Handle Predictions
# 4. Handle Stored Models
#     - When a model is trained on a given data set for a given feature, archive data set and model
#     - Create an archive listing of dataset and trained model
#     - Retrieve 

# Pro version of this:
# 1 Data Ingestion (e.g., Apache Kafka, Amazon Kinesis)
# 2 Data Preprocessing (e.g., pandas, NumPy)
# 3 Feature Engineering and Selection (e.g., Scikit-learn, Feature Tools)
# 4 Model Training (e.g., TensorFlow, PyTorch)
# 5 Model Evaluation (e.g., Scikit-learn, MLflow)
# 6 Model Deployment (e.g., TensorFlow Serving, TFX)
# 7 Monitoring and Maintenance (e.g., Prometheus, Grafana)

class ModelRunHandle:
    def __init__(self, proc_obj_path: str, target_feature_name: str):
        # self.processor_obj=load_object(file_path=proc_obj_path, unique_name=target_feature_name)       
        pass 


    def get_selected_feature(self, feature:str) -> str:
        self.selected_feature = feature
        