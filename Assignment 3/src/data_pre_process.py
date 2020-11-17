import pandas as pd
import numpy as np

#data cleaning functions
def fill_values(x, f_type):

    if f_type == "mean":
        val = np.nanmean(x)
    elif f_type == "mode":
        val = max(x, key=x.tolist().count)
    elif f_type == "median":
        val = np.nanmedian(x)
    return val

def ask(feature):

    replacement_type = {}
    replacement_type["Community"] = "mode"
    replacement_type["Age"] = "median"
    replacement_type["Residence"] = "mode"
    replacement_type["BP"] = "mean"
    replacement_type["HB"] = "mean"
    replacement_type["Delivery phase"] = "mode"
    replacement_type["Weight"] = "mean"

    return replacement_type[feature]


def preprocess_data(df):
    df.drop(["Education"], axis = 1, inplace=True) #education column is dropped because all values are equal and thus variance is zero. This independent varaible wont have effect on dependent varaiable
    for feature in df.columns:
        if df[feature].isnull().any():
            replacing_value = fill_values(df[feature].values, ask(feature))
            df[feature].fillna(replacing_value, inplace=True)
    df.to_csv("LBW_cleaned.csv", index=False)

df = pd.read_csv("LBW_Dataset.csv")
preprocess_data(df)
