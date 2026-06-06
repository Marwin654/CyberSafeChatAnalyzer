import pandas as pd

df = pd.read_csv("datasets/spam/sms.csv")

print(df.head())
print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())