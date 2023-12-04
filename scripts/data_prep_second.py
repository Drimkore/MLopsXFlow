import sys
import io
import os
import pandas as pd
from sklearn import preprocessing


input_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage1", "googleplaystore.csv")
output_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage2", "googleplaystore.csv")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage2"), exist_ok=True)

df_gps = pd.read_csv(input_file)
print(df_gps)


norm = preprocessing.MinMaxScaler()

df_gps['Reviews'] = norm.fit_transform(df_gps[['Reviews']])
df_gps['Size'] = norm.fit_transform(df_gps[['Size']])
df_gps['Installs'] = norm.fit_transform(df_gps[['Installs']])
df_gps['Price'] = norm.fit_transform(df_gps[['Price']])

df_gps.to_csv(output_file)

