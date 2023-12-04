import sys
import io
import os
import pandas as pd


input_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage2", "googleplaystore.csv")
output_file = os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage3", "googleplaystore.csv")
os.makedirs(os.path.join("/home","art","gitprj","MLopsXFlow","datasets","data", "stage3"), exist_ok=True)

df_gps = pd.read_csv(input_file)
print(df_gps)

df_gps['Rating'] = [round(i) for i in df_gps['Rating']]

df_gps.to_csv(output_file)
