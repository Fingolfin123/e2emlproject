import os
import sys
import dill
import joblib

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, unique_name, obj):
    try:
        logging.info('Saving Processing Object')
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Split filename and insert unique name before the extension
        base, ext = os.path.splitext(os.path.basename(file_path))
        modified_filename = f"{base}_{unique_name}{ext}"
        modified_path = os.path.join(dir_path, modified_filename)

        # Use joblib if it's a .joblib file, otherwise use dill
        if ext.lower() == ".joblib":
            joblib.dump(obj, modified_path)
        else:
            with open(modified_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)