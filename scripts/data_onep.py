import sys
import io
import os
import pandas as pd
from sklearn import preprocessing
import mlflow
from mlflow.tracking import MlflowClient


dataset_path = os.path.join("/home","art","gitprj","MLopsXFlow","datasets") 
input_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "raw", "googleplaystore.csv")
output_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage1", "googleplaystore.csv")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage1"), exist_ok=True)
print(input_file)
df_gps = pd.read_csv(input_file)
print(df_gps)

with mlflow.start_run():
    rem = df_gps['Category']!='1.9'
    df_gps = df_gps.loc[rem]

    df_gps['Size'] = [str(float(i.replace("M", ""))*1024) if "M" in i else i for i in df_gps['Size']]
    df_gps['Size'] = df_gps['Size'].apply(lambda x: str(x).replace('k', ''))
    df_gps['Size'] = df_gps['Size'].apply(lambda x: str(x).replace('Varies with device', '0'))
    df_gps['Size'] = df_gps['Size'].apply(lambda x: str(x).replace("+",""))
    df_gps['Size'] = df_gps['Size'].apply(lambda x: str(x).replace(",",""))
    df_gps['Size'] = df_gps['Size'].astype(float)
    #df_gps['Size'].dtype


    df_gps["Installs"] = df_gps["Installs"].apply(lambda x: str(x).replace("+",""))
    df_gps["Installs"] = df_gps["Installs"].apply(lambda x: str(x).replace(",",""))
    df_gps["Installs"] = df_gps["Installs"].astype(float)
    #df_gps.Installs.dtype


    df_gps["Price"] = df_gps["Price"].apply(lambda x: str(x).replace("$", ""))
    df_gps["Price"] = df_gps["Price"].astype(float)
    #df_gps["Price"].dtype

    df_gps_type = pd.get_dummies(df_gps['Type'])
    df_gps = pd.concat([df_gps, df_gps_type], axis=1)

    le = preprocessing.LabelEncoder()
    df_gps['Content Rating'] = le.fit_transform(df_gps['Content Rating'])
    df_gps['Genres'] = le.fit_transform(df_gps['Genres'])

    df_gps_cat = pd.get_dummies(df_gps['Category'])
    df_gps = pd.concat([df_gps, df_gps_cat], axis=1)

    df_gps = df_gps.drop(labels = ['App','Category','Type','Last Updated','Current Ver','Android Ver'], axis=1)
    df_gps["Reviews"] = df_gps["Reviews"].astype(float)

    df_gps = df_gps.fillna(0)

    mlflow.log_artifact(local_path="/home/art/gitprj/MLopsXFlow/scripts/data_onep.py",
                        artifact_path="data_onep code")
    mlflow.end_run()

df_gps.to_csv(output_file)
