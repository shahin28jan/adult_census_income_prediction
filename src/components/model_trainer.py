# Basuc Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model=LogisticRegression()
            #you are training your model here
            model.fit(X_train,y_train)
            model.score(X_train,y_train)
            y_pred=model.predict(X_test)
            accuracy_score(y_test,y_pred)
            confusion_matrix(y_test,y_pred)

           

            logging.info(f'Model Report : {model_report}')


            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model
            
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)