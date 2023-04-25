import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd





class PredictPipeline:
    
    def __init__(self):
        pass
    
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join(r"F:\study material\Data Science\iNeuron-FullStack-DataSciene-Assignments\modular coding assignment\Credit-card-default\src\pipeline\artifacts\preprocessor.pkl")
          
            model_path=os.path.join(r"F:\study material\Data Science\iNeuron-FullStack-DataSciene-Assignments\modular coding assignment\Credit-card-default\src\pipeline\artifacts\model.pkl")
      
            # we are giving path as os.path.join because this formate will run in both linux instead of doing /path/ this type
            
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            data_scaled=preprocessor.transform(features)
            # consider it as a xtest data as we only tranform it
            
            pred=model.predict(data_scaled)
            return pred
        
        
        
        
        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
        
        
class CustomData:
    
    def __init__(self,LIMIT_BAL:float,SEX:float,EDUCATION:float,AGE:float,avg_default:float,avg_bill_amt:float,avg_pay_amt:float):
        
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.AGE = AGE
        self.avg_default = avg_default
        self.avg_bill_amt = avg_bill_amt
        self.avg_pay_amt = avg_pay_amt
        
        
        
    def get_data_as_dataframe(self):
        try :
            custom_data_input_dict = {
                'LIMIT_BAL':[self.LIMIT_BAL],
                'SEX':[self.SEX],
                'EDUCATION':[self.EDUCATION],
                'AGE':[self.AGE],
                'avg_default':[self.avg_default],
                'avg_bill_amt':[self.avg_bill_amt],
                'avg_pay_amt':[self.avg_pay_amt]
                
            }
        
            df = pd.DataFrame(custom_data_input_dict)  
            logging.info('Dataframe Gathered')
            return df   
        
           
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
            