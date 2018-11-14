import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))               # Root directory of the project
RAW_DATA_DIRECTORY = '/Downloaded_Data/Raw_Data/'                   # Raw-scraped data from the Fitbit API
PROCESSED_DATA_DIRECTORY = '/Downloaded_Data/Preprocessed_Data/'    # Preprocessed up/down sampled data
TRAINING_DATA_DIRECTORY = '/Downloaded_Data/Training_Data/'         # Training data for the neural network
FILE_EXT = '.xlsx'                                                  # File format to save/load data
CLIENT_ID = '22D56P'                                                # Fitbit API Client ID
CLIENT_SECRET = '9aefa27740d00cd57d1b06beb43992ac'                  # Fitbit API Client Secret
START_DATE = '2018-01-30'   # '2016-09-28'                          # Date to start scraping data
