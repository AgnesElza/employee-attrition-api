import pandas as pd, json
df = pd.read_csv("data/reference_sample.csv")
row = df.iloc[0].to_dict()
with open("sample.json","w") as f:
    json.dump({"features": row}, f, indent=2)
print("Wrote sample.json with", len(row), "features")
