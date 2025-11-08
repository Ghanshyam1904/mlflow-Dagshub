import pandas as pd
import numpy as np
import os

data = {"Name":["Alice","Bob","Charlie"],
        "Age":[22,22,22],
        "city":["India","Russia","Japan"]}

df = pd.DataFrame(data)

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

file_path = os.path.join(data_dir, 'data.csv')
data = df.to_csv(file_path,index=False)
print(data)