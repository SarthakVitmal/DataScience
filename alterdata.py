import pandas as pd

data = pd.read_csv('Dataset.csv')

data["Market Share Gaming GPU (%)"] = 2 * data["Market Share Gaming GPU (%)"]

data.to_csv('Dataset.csv', index=False)