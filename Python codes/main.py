import os
import pandas as pd

def group_data(df):
    groups = df.groupby(by=["sector_code", "trading_date"])
    df["trans_type"] = df["transaction_type"].map(equiv)
    df["money_flow"] = df["trans_type"]*df["price"]*df["volume"]
    grouped_df = groups["money_flow"].sum().unstack(-1)
    return grouped_df



equiv = {10:1, 20:-1}
df = pd.read_csv(f"{os.pardir}/Data/test_before_2000.csv")
df = group_data(df)
print(df)