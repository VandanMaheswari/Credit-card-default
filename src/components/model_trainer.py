# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score
from src.exception import CustomException
from src.logger import logging


from src.utils import save_object
# from src.utils import evaluate_model

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
        try :
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            
            # models = {
            #     'LogisticRegression': LogisticRegression(),
             
            # }
            
            
            # model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            
            # print(model_report)
            # print('\n====================================================================================\n')
            # logging.info(f'Model Report : {model_report}')
            
            
            # best_model_score = max(sorted(model_report.values()))
            
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            # best_model = models[best_model_name]
            
            # print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            # print('\n====================================================================================\n')
            # logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            
            model = LogisticRegression()
            
            #you are training your model here
            model.fit(X_train,y_train)
            
            train_score = model.score(X_train,y_train) 
            # our train model score 
            
        
            y_pred=model.predict(X_test)
            
            
            # test accuracy
            test_score = accuracy_score(y_test,y_pred)
            
            roc_auc = roc_auc_score(y_test, y_pred)
            # roc-auc score as well
            
            precision = precision_score(y_test, y_pred)
            # precision score
            
            recall = recall_score(y_test, y_pred)
            # recall_score
            
            conf_matrix = confusion_matrix(y_test,y_pred)
            
            
            print('Model Name : LinearRegression')
            print(f'train Score : {train_score}')
            print(f'test Score : {test_score}')
            print(f'roc_auc_score : {roc_auc}')
            print(f'precision score : {precision}')
            print(f'recall score : {recall}')
            print(f'confusion matrix :\n {conf_matrix}')
            
            
            
            
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=model
            )
          
          
            
        except Exception as e :
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)            