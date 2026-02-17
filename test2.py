import pandas as pd
df = pd.read_parquet("data_raw/PType.parquet")
print(df["Parent Category"].unique())
print(df["P Type"].unique())
