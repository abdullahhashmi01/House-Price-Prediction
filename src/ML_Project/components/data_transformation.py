import os
import sys
import re
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.ML_Project.logger import logging
from src.ML_Project.exception import CustomException
from src.ML_Project.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ---------- Parsing helpers (raw text -> numeric features) ----------

    @staticmethod
    def parse_amount(x):
        """Convert '42 Lac' / '1.40 Cr' style strings into rupee values."""
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if 'Call' in x:
            return np.nan
        m = re.match(r'([\d.]+)\s*(Lac|Cr)', x)
        if not m:
            return np.nan
        val, unit = float(m.group(1)), m.group(2)
        return val * 1e5 if unit == 'Lac' else val * 1e7

    @staticmethod
    def parse_sqft(x):
        if pd.isna(x):
            return np.nan
        m = re.match(r'([\d.]+)', str(x))
        return float(m.group(1)) if m else np.nan

    @staticmethod
    def parse_floor(x):
        if pd.isna(x):
            return np.nan, np.nan
        m = re.match(r'(Ground|\d+)\s*out of\s*(\d+)', str(x))
        if not m:
            return np.nan, np.nan
        floor = 0 if m.group(1) == 'Ground' else float(m.group(1))
        return floor, float(m.group(2))

    @staticmethod
    def parse_parking(x):
        if pd.isna(x):
            return np.nan
        nums = re.findall(r'(\d+)', str(x))
        return sum(float(n) for n in nums) if nums else np.nan

    # ---------- Feature engineering ----------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn raw scraped columns into clean numeric / categorical features."""
        try:
            df = df.copy()

            df['total_price'] = df['Amount(in rupees)'].apply(self.parse_amount)

            df['carpet_sqft'] = df['Carpet Area'].apply(self.parse_sqft)
            df['super_sqft'] = df['Super Area'].apply(self.parse_sqft)
            df['area_sqft'] = df['carpet_sqft'].fillna(df['super_sqft'])

            df['BHK'] = df['Title'].str.extract(r'(\d+)\s*BHK').astype(float)

            floors = df['Floor'].apply(self.parse_floor)
            df['floor_num'] = floors.apply(lambda t: t[0])
            df['total_floors'] = floors.apply(lambda t: t[1])

            for c in ['Bathroom', 'Balcony']:
                df[c] = df[c].replace('> 10', '11').astype(float)

            df['car_parking_num'] = df['Car Parking'].apply(self.parse_parking)

            # Filter unrealistic / unparseable rows
            df = df[(df['total_price'] > 1e5) & (df['total_price'] < 2e8)]
            df = df[(df['area_sqft'] >= 100) & (df['area_sqft'] <= 10000)]
            df = df.dropna(subset=['total_price', 'area_sqft', 'BHK'])

            # Group rare locations into 'other' to keep one-hot encoding manageable
            top_loc = df['location'].value_counts().head(30).index
            df['location_grp'] = np.where(df['location'].isin(top_loc), df['location'], 'other')

            # Target: log(price) to tame the right-skew
            df['log_price'] = np.log1p(df['total_price'])

            logging.info(f"Feature engineering complete. Shape after cleaning: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ---------- Preprocessing pipeline (imputation + encoding) ----------

    def get_data_transformer_object(self):
        try:
            num_features = ['area_sqft', 'BHK', 'floor_num', 'total_floors',
                             'Bathroom', 'Balcony', 'car_parking_num']
            cat_features = ['location_grp', 'Status', 'Transaction', 'Furnishing', 'facing']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median'))
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_features),
                ('cat_pipeline', cat_pipeline, cat_features)
            ])

            logging.info("Preprocessing object (ColumnTransformer) created successfully")
            return preprocessor, num_features, cat_features

        except Exception as e:
            raise CustomException(e, sys)

    # ---------- Main entry point ----------

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation")

            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)

            preprocessor, num_features, cat_features = self.get_data_transformer_object()
            input_features = num_features + cat_features
            target_column = 'log_price'

            X_train = train_df[input_features]
            y_train = train_df[target_column]

            X_test = test_df[input_features]
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing data")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Saved preprocessing object")

            return (
                X_train_arr,
                y_train.to_numpy(),
                X_test_arr,
                y_test.to_numpy(),
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)