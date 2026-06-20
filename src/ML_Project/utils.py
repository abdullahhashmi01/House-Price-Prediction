import os 
import sys
import pandas as pd
import numpy as np
import pickle

from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException



def read_data(file_path):

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data read successfully from {file_path}")
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):

    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating {model_name}")
            params = param_grid.get(model_name, {})

            if params:
                    from sklearn.model_selection import RandomizedSearchCV
                    grid_search = RandomizedSearchCV(estimator=model, param_distributions=params,
                                cv=3, n_jobs=-1, random_state=42)
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5

            report[model_name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            trained_models[model_name] = best_model
            logging.info(f"{model_name} -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)