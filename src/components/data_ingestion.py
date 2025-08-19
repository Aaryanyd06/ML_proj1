import os,sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data : str = os.path.join('artefacts', "train.csv")
    test_data : str = os.path.join('artefacts', "test.csv")
    raw_data : str = os.path.join('artefacts', "raw.csv")

class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def data_ingestion(self):
        logging.info("Entered data ingestion class")
        try:
            df = pd.read_csv('notebook/stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data, header=True, index=False)
            
            test_set.to_csv(self.ingestion_config.test_data, header=True, index=False)

            logging.info("Ingestion complete")

            return(
                self.ingestion_config.train_data,
                self.ingestion_config.test_data
            )
        except Exception as e: 
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=Dataingestion()
    train_data, test_data=obj.data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))


