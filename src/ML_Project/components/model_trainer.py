import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
from src.ML_Project.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Splitting data into train and test for model training")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "CatBoost": CatBoostRegressor(random_state=42, verbose=False),
                "LightGBM": LGBMRegressor(random_state=42, verbosity=-1),
            }

            param_grid = {
                "Linear Regression": {},
                "Random Forest": {
                    "n_estimators": [60, 100],
                    "max_depth": [10, 15],
                },
                "XGBoost": {
                    "n_estimators": [200, 300],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [4, 6],
                },
                "CatBoost": {
                    "iterations": [300, 500],
                    "learning_rate": [0.05, 0.1],
                    "depth": [4, 6],
                },
                "LightGBM": {
                    "n_estimators": [200, 300],
                    "learning_rate": [0.05, 0.1],
                    "num_leaves": [31, 63],
                },
            }

            logging.info("Evaluating all candidate models with hyperparameter tuning")
            report, trained_models = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param_grid=param_grid
            )

            # report[model_name] = {"R2": ..., "MAE": ..., "RMSE": ...}
            best_model_name = max(report, key=lambda name: report[name]["R2"])
            best_model_score = report[best_model_name]["R2"]
            best_model = trained_models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No model achieved a satisfactory R2 score (> 0.6)", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Saved best model ({best_model_name}) at {self.model_trainer_config.trained_model_file_path}")

            return report

        except Exception as e:
            raise CustomException(e, sys)