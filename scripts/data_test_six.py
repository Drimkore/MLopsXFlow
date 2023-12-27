import sys
import io
import os
import yaml
import pickle
import json
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("testing_model")

input_file_test = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage4", "test.csv")
input_file_model = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","models", "model.pkl")
output_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","metrics", "evaluation.json")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","metrics"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["train"]
p_max_depth = params["max_depth"]
p_max_features = params["max_features"]
p_min_samples_leaf = params["min_samples_leaf"]

df_gps = pd.read_csv(input_file_test)
print(df_gps)
x_test = df_gps.drop(labels=["Rating"], axis = 1)
y_test = df_gps['Rating']

metr= 0
with open(input_file_model, "rb") as ff: 
    unpickler = pickle.Unpickler(ff)
    tree = unpickler.load()
    scr = tree.score(x_test, y_test)
    metr = scr

with open(output_file, "w") as f:
    json.dump({"score":metr}, f)


with mlflow.start_run():
    mlflow.log_metric("score", metr)
    with open(input_file_model, "rb") as ff: 
        unpickler = pickle.Unpickler(ff)
        tree = unpickler.load()
        model_info = mlflow.sklearn.log_model(tree,
                            artifact_path="sklearn-model",
                            registered_model_name="tree",
                            input_example=x_test)        
        model = mlflow.sklearn.load_model(model_info.model_uri)
        pred = model.predict(x_test)
        signature = infer_signature(x_test, pred)
        mlflow.sklearn.log_model(tree,
                            artifact_path="sklearn-model",
                            registered_model_name="tree",
                            signature = signature,
                            input_example=x_test)  

        mlflow.log_artifact(local_path="/home/art/gitprj/MLopsXFlow/scripts/data_test_six.py",
                            artifact_path="testing_model code")
        
    mlflow.end_run()


