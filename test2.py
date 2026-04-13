import pandas as pd
df = pd.read_parquet("data_agg/tab4_industry.parquet")
print(df.columns)
print(df.head())