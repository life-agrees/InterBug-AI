import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys

# Add the project root to sys.path if needed
# Uncomment if Sei_Module is not in your PYTHONPATH
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the Sei_Module
from Sei_Module.data_cleaning import clean_data
from Sei_Module.data_preprocess import preprocessing 
from Sei_Module.testing import framework
from Sei_Module.model import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_freshness(file_path, hours=24):
    """Check if file exists and was modified within given hours"""
    try:
        if not os.path.exists(file_path):
            return False
            
        file_time = datetime.fromtimestamp(
            os.path.getmtime(file_path), 
            tz=timezone.utc
        )
        now = datetime.now(timezone.utc)
        return (now - file_time) < timedelta(hours=hours)
        
    except Exception as e:
        logger.warning(f"Freshness check failed for {file_path}: {str(e)}")
        return False

def main():
    try:
        logger.info("Starting Interbug AI Pipeline")
        
        # Path configuration
        RAW_DATA_PATH = 'combined_datas.csv'
        CLEANED_DATA_PATH = 'official_sei_data3.csv'
        PREPROCESSED_DATA_PATH = 'preprocessed_features.csv'
        TRAINING_DATA_PATH = 'training_data.csv'
        
        # Check if raw data exists
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")
        
        # Data cleaning flow
        if not os.path.exists(CLEANED_DATA_PATH) or not check_file_freshness(CLEANED_DATA_PATH):
            logger.info("Cleaning data")
            clean_data()
            
            # Verify cleaned data was created
            if not os.path.exists(CLEANED_DATA_PATH):
                raise FileNotFoundError(f"Cleaned data file was not created: {CLEANED_DATA_PATH}")
        else:
            logger.info(f"Using existing cleaned data from {CLEANED_DATA_PATH}")
            
        # Load cleaned data
        cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
        logger.info(f"Loaded cleaned data with {len(cleaned_data)} rows")
        
        # Processing pipeline
        if not os.path.exists(PREPROCESSED_DATA_PATH) or not check_file_freshness(PREPROCESSED_DATA_PATH):
            logger.info("Preprocessing data")
            processed_data = preprocessing(cleaned_data)
            if processed_data.empty:
                raise ValueError("Preprocessing returned empty dataset")
                
            processed_data.to_csv(PREPROCESSED_DATA_PATH, index=False)
            logger.info(f"Saved preprocessed data to {PREPROCESSED_DATA_PATH}")
        else:
            logger.info(f"Using existing preprocessed data from {PREPROCESSED_DATA_PATH}")
            
        # Load preprocessed data
        processed_data = pd.read_csv(PREPROCESSED_DATA_PATH)
        logger.info(f"Loaded preprocessed data with {len(processed_data)} rows and {len(processed_data.columns)} features")
            
        # Bug detection
        if not os.path.exists(TRAINING_DATA_PATH) or not check_file_freshness(TRAINING_DATA_PATH):
            logger.info("Detecting bugs")
            final_data = framework.detect_bugs(processed_data)
            if final_data.empty:
                raise ValueError("Bug detection returned empty dataset")
                
            # Verify required columns
            required_cols = ['transaction_mismatch', 'fee_ratio', 
                            'success_rate', 'bug_detected']
            missing_cols = [col for col in required_cols if col not in final_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
                
            # Save training data
            final_data.to_csv(TRAINING_DATA_PATH, index=False)
            logger.info(f"Saved training data to {TRAINING_DATA_PATH} with {final_data['bug_detected'].sum()} bugs")
        else:
            logger.info(f"Using existing training data from {TRAINING_DATA_PATH}")
        
        # Model training
        if not os.path.exists("interbug_model.pkl") or not check_file_freshness("interbug_model.pkl"):
            logger.info("Training model")
            train_model()
            
            # Verify model was created
            if not os.path.exists("interbug_model.pkl"):
                raise FileNotFoundError("Model file was not created")
                
            logger.info("Model training completed successfully")
        else:
            logger.info("Using existing trained model")
            
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()