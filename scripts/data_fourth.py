import sys
import io
import os
import yaml
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


input_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage3", "googleplaystore.csv")
output_file_test = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage4", "test.csv")
output_file_train = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage4", "train.csv")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage4"), exist_ok=True)

df_gps = pd.read_csv(input_file)
print(df_gps)

x = df_gps.drop(labels=["Rating"], axis = 1)
y = df_gps['Rating']
params = yaml.safe_load(open("params.yaml"))["split"]
p_split_ratio = params["split_ratio"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=p_split_ratio, random_state=42)

pd.concat([x_train, y_train], axis=1).to_csv(output_file_train, index=None)
pd.concat([x_test, y_test], axis=1).to_csv(output_file_test, index=None)

