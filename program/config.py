# DATA 
import os
print("new conf")
DATA_FOLDER ='/opt/ml/input/data/training'
#"../work_dir/input/v2-plan-seedlings-dataset/"  
#'/opt/ml/input/v2-plan-seedlings-dataset'

# MODEL FITTING 
IMAGE_SIZE = 150 
BATCH_SIZE = 4 
EPOCHS = 1 

# MODEL PERSISTING 
MODEL_PATH = "/opt/ml/model/cnn_model.h5"
PIPELINE_PATH = "/opt/ml/model/cnn_pipe.pkl"  
CLASSES_PATH = "/opt/ml/model/classes.pkl"
ENCODER_PATH = '/opt/ml/model/encoder.pkl'

