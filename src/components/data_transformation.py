import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object





@dataclass
class DataTransformationConfig:
       preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
       
       


class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
        
    def get_data_transformation_object(self):
        try :
               
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            preprocessor=Pipeline(
                steps=[
                ('imputer',SimpleImputer(missing_values = np.nan,
                        strategy ='mean')),
                ('scaler',StandardScaler())

                ]

            )
            
            
            return preprocessor

            logging.info('Pipeline Completed')      
        
        
        except Exception as e:
            logging.info("error in data transformation")
            raise CustomException(e,sys)
        
        
        
        
        
        
        
        
    def initaite_data_transformation(self,train_path,test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            df = pd.concat([train_df, test_df])
            
            logging.info('Read the data is completed')
            
            
            df.rename(columns={"PAY_0" : "PAY_1" , "default payment next month" : "DEFAULT"},inplace = True)
            df.drop(["ID"],axis=1, inplace=True)
            
            
            df['avg_default'] = round(df.iloc[:,5:11].sum(axis=1)/6,3)
            # average default history
            df['avg_bill_amt'] = round(df.iloc[:, 11:17].sum(axis=1) / 6,3)
            # average bill amount
            df['avg_pay_amt'] = round(df.iloc[:, 18:24].sum(axis=1) / 6,3)
            # average payment amount
            
            df = df.drop(['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','MARRIAGE'],axis = 1)
            
            
            train_df,test_df = train_test_split(df,test_size=0.30,random_state=42)
            
            
            

            logging.info('Obtaining preprocessing object')
            
            
            preprocessing_obj = self.get_data_transformation_object()
            
            
            target_column_name = 'DEFAULT'
            drop_columns = [target_column_name]
            
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            # x_train , y_train

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            # X_test, y_test
            

            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # in this line we have to convert target_feature_train_df in the array from and input_feature_train_arr is already a array
            # because when we do fit_tranfrom or transfrom it will return in form of array 
            # and we are doing nothing just at first we divide the data into train and test and that data we use in this
            # and then we divide data into x:- independant and y:- dependant for both train and test then
            # fit_tranfrom for train and transform for test data of x only means input features then 
            # combine them in a form of array in which train and test array contrain independant tranformed data and y data(input feature
            # and traget feautre)
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                # file ka path or processor pas kia 

            )
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                # transformed train and test data as well as file path for pickle file returned
            )

        
        except Exception as e :
            logging.info("exception occured in this intiate data tranformation")
            raise CustomException(e,sys)    
    
    
        