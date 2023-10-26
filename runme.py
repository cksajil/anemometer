import os
import pandas as pd

DATA_PATH = "dataset"
DATA_FILE = "data.csv"

data = pd.read_csv(os.path.join(DATA_PATH, DATA_FILE))

print(data)