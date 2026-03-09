import pandas as pd

data = pd.read_csv("dataset/hypertension_dataset.csv")

for col in ["Smoking_Status", "Physical_Activity_Level", "Family_History"]:
    data[col] = data[col].astype("category")
    print(col, dict(enumerate(data[col].cat.categories)))