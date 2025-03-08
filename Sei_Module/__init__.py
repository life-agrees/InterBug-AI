# Export core functions
from .data_collection.fetch_data import fetch_sei_data
from .data_cleaning.clean_data import clean_and_save_data
from .data_preprocess.preprocessing import preprocess_data
from .testing.framework import detect_bugs
from .model.train_model import train_model

__all__ = [
    'fetch_sei_data',
    'clean_and_save_data',
    'preprocess_data',
    'detect_bugs',
    'train_model'
]