
import sagemaker
from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()

def run_training(test=True):

    if test:
        tf_estimator = TensorFlow(entry_point='pipeline.py', 
                            #role=role,
                            instance_count=1, 
                            instance_type='local',
                            framework_version='1.12', 
                            py_version='py3',
                            script_mode=True,
                            hyperparameters={'epochs': 1}
                            )
    else:
        f_estimator = TensorFlow(entry_point='mnist-train-cnn.py', 
                          #role=role,
                          instance_count=1, 
                          instance_type='ml.p3.2xlarge',
                          framework_version='1.12', 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters={
                              'epochs': 10,
                              'batch-size': 32,
                              'learning-rate': 0.001}
                         )
    


if __name__ == '__main__':
    run_training()