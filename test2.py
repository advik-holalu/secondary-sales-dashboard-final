import pandas as pd

file_path = "data_raw/BusinessOverview.xlsx"

sheet_name = "Blinkit Secondary"   # check one first
df = pd.read_excel(file_path, sheet_name=sheet_name)

print("Columns in Blinkit Secondary:")
for col in df.columns:
    print("-", col)