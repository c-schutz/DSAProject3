import pandas as pd

movies_file = "movies_metadata.csv"
credits_file = "credits.csv"
movies_df = pd.read_csv(movies_file)
print(movies_df.duplicated().sum())
print(movies_df.isna().sum())