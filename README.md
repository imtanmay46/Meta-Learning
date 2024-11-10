# Meta-Learning

THIS REPOSITORY ONLY STORES FEW OF THE GOOD RUNS AS BACKUP.

THE LATEST COMMIT, IS IDENTICAL TO THE ONE IN THE SUBMISSION, WITH A CORR ON META-TESTING DATA AS 0.018

META-LEARNING 

ASSIGNMENT-1: NumerAI

TANMAY SINGH
CSAI
CLASS OF '25


INSTRUCTIONS TO RUN & DESCRIPTION OF FILE CONTENTS:

1. Run the pre-installs.py file to download/resolve all the dependency issues. This will also create two directories, 'data' (used to download & store the data from NumerAI using an API call)
   & the 'saved_models' directory that will download the pickle files of the trained models from the publicly accessible google drive link using gdown & store them (to be used in validation & prediction files).
2. Run the models.py present inside the Models directory to initialize the models to be used in the training files (6 experts & the meta-model).
3. Run the train.ipynb or train.py to train the models & overwrite the downloaded pickles. If google drive pickle files are to be used after this, re-run pre-installs.
4. Run the validation.ipynb or validation.py to evaluate the trained/saved models on the meta-testing set (validation parquet).
5. Run the predict.ipynb or predict.py to generate live predictions. This will create a directory 'predictions' that will store the live predictions. Ensure to run the pre-installs.py script before running these files, but comment out the call to download saved models if you want to generate predictions on the newly trained models. Do not comment out if you want the predictions to be generated on the saved models on the google drive.
