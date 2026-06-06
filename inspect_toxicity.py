import pandas as pd

df = pd.read_csv("datasets/toxicity/train.csv")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())