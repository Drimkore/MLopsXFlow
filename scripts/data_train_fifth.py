import sys
import io
import os
import yaml
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import mlflow
from mlflow.tracking import MlflowClient



mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("training_mod")

input_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage4", "train.csv")
output_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","models", "model.pkl")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","models"), exist_ok=True)

params = {}
df_gps = pd.read_csv(input_file)
print(df_gps)
x_train= df_gps.drop(labels=["Rating"], axis = 1)
y_train = df_gps['Rating']


max_depth_list = [5,10,15]
max_features_list = [15,19,23]
min_samples_leaf_list = [1]

def run_exp(p_max_depth, p_max_features, p_min_samples_leaf):
    params["max_depth"] = p_max_depth
    params["max_features"] = p_max_features
    params["min_samples_leaf"] = p_min_samples_leaf
    with mlflow.start_run():
        mlflow.set_experiment_tag("version","train")
        mlflow.log_params(params)
        tree = DecisionTreeClassifier(max_depth=p_max_depth,max_features=p_max_features,min_samples_leaf=p_min_samples_leaf)
        mlflow.sklearn.log_model(tree,
                            artifact_path="sklearn-model",
                            registered_model_name="tree")
        mlflow.log_artifact(local_path="/home/art/gitprj/MLopsXFlow/scripts/data_train_fifth.py",
                            artifact_path="training_mod code")
        tree.fit(x_train, y_train)
        mlflow.log_metric("score", tree.score(x_train, y_train))
        mlflow.end_run()

for depth in max_depth_list:
    for features in max_features_list:
        for leaf in min_samples_leaf_list:
            run_exp(depth, features, leaf)

with mlflow.start_run():
    mlflow.set_experiment_tag("version","best")
    df_metr = mlflow.search_runs(experiment_names=["training_mod"], order_by=["metrics.score DESC"])
    print(df_metr[["metrics.score", "params.max_depth","params.max_features","params.min_samples_leaf", "run_id"]])
    params["max_depth"] = int(df_metr[["params.max_depth"]].head(1).values[0][0])
    params["max_features"] = int(df_metr[["params.max_features"]].head(1).values[0][0])
    params["min_samples_leaf"] = int(df_metr[["params.min_samples_leaf"]].head(1).values[0][0])
    mlflow.log_params(params)
    tree_f = DecisionTreeClassifier(max_depth=int(df_metr[["params.max_depth"]].head(1).values[0][0]),max_features=int(df_metr[["params.max_features"]].head(1).values[0][0]),min_samples_leaf=int(df_metr[["params.min_samples_leaf"]].head(1).values[0][0]))
    tree_f.fit(x_train, y_train) 
    mlflow.sklearn.log_model(tree_f,
                            artifact_path="sklearn-model",
                            registered_model_name="tree")
    mlflow.log_artifact(local_path="/home/art/gitprj/MLopsXFlow/scripts/data_train_fifth.py",
                            artifact_path="training_mod code")
    mlflow.log_metric("score", tree_f.score(x_train, y_train))   
    with open(output_file, "wb") as f:
        pickle.dump(tree_f, f)        
    mlflow.end_run()

