import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from pathlib import Path
project_root=Path(__file__).resolve().parent
data_dir=project_root / 'data' /'UCI-HAR Dataset'

# load dataset

x_train = np.loadtxt(data_dir/ "train" / "X_train.txt")
y_train = np.loadtxt(data_dir/ "train" / "y_train.txt")

x_test = np.loadtxt(data_dir/ "test" / "X_test.txt")
y_test = np.loadtxt(data_dir/ "test" / "y_test.txt")

#tracking in ML flow and training

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("UCI-HAR-CLASSIFICATION")
architecture=[(100,50),(50,25),(64,32,16)]


with mlflow.start_run(run_name='ARCHITECTURE') as parent:
    
    
        for arch in architecture:
            print(f'running the code with architecture={arch}')
        
            with mlflow.start_run(run_name=f'architecture-{arch}',nested=True):
                
              
                mlflow.log_param('hidden layer ',str(arch))
                
                mlp=MLPClassifier(  hidden_layer_sizes=arch, 
                max_iter=100, 
                alpha=1e-4,
                solver='adam', 
                verbose=False, 
                random_state=42)
                
                mlp.fit(x_train,y_train)
                
                mlflow.sklearn.log_model(sk_model=mlp,name='UCI-HAR CLASSIFIER',serialization_format='cloudpickle')#used gpt here
                
                train_pred=mlp.predict(x_train)
                
                train_acc=accuracy_score(y_train,train_pred)
                
                y_pred=mlp.predict(x_test)
                
                test_acc=accuracy_score(y_test,y_pred)
                
                for epoch,loss in enumerate(mlp.loss_curve_):
                    mlflow.log_metric(f'training loss',loss,step=epoch)#plot
                
                mlflow.log_metric('train accuracy',train_acc)
                
                mlflow.log_metric('accuracy',test_acc)
                mlflow.sklearn.log_model(sk_model=mlp,name='UCI-HAR CLASSIFIER',
                                        serialization_format='cloudpickle')#used gpt here