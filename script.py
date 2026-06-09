import pandas as pd
df = pd.read_csv("114/results.csv")
df = df[df["date"] >= "2021-01-01"]
df = df[df["tournament"]== "Morocco, Capital of African Football"]
print(df)
