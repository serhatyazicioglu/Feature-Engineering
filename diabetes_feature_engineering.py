"""
Pregnancies = Hamile kalma sayısı
Glucose = Glikoz
Blood Pressure = Kan basıncı
Skin Thickness = Deri kalınlığı
Insulin = İnsülin
BMI (Body Mass Index) = Beden kitle endeksi
Diabetes Pedigree Function = Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
Age = Yaş
Outcome = Diyabet olup olmadığı bilgisi
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from helpers.eda import grab_col_names

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("datasets/diabetes.csv")

df.info()
df.isnull().sum()

df.describe().T

df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = \
    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)


def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.25)
    quartile3 = dataframe[col_name].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


for col in num_cols:
    if col != "Glucose":
        replace_with_thresholds(df, col)

# MISSING VALUES
df.isnull().sum()


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


na_name = missing_values_table(df, na_name=True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_name)

df.pivot_table(df, index=["Outcome"])

for col in df.columns:
    df.loc[(df["Outcome"] == 0) & (df[col].isnull()), col] = df[df["Outcome"] == 0][col].median()
    df.loc[(df["Outcome"] == 1) & (df[col].isnull()), col] = df[df["Outcome"] == 1][col].median()

# FEATURE ENGINEERING

df.loc[(df["Age"] < 18), "NEW_AGE_CAT"] = "Young"
df.loc[(df["Age"] > 18) & (df["Age"] < 56), "NEW_AGE_CAT"] = "Mature"
df.loc[(df["Age"] > 56), "NEW_AGE_CAT"] = "Old"

df.loc[(df["BMI"] < 18.5), "NEW_BMI_CAT"] = "Underweight"
df.loc[(df["BMI"] > 18.5) & (df["BMI"] < 25), "NEW_BMI_CAT"] = "Normal"
df.loc[(df["BMI"] > 25) & (df["BMI"] < 30), "NEW_BMI_CAT"] = "Overweight"
df.loc[(df["BMI"] > 30) & (df["BMI"] < 40), "NEW_BMI_CAT"] = "Obese"
df.loc[(df["BMI"] > 40), "NEW_BMI_CAT"] = "	Severe Obese"

df.loc[(df["Glucose"] < 70), "NEW_GLUCOSE_CAT"] = "Low"
df.loc[(df["Glucose"] > 70) & (df["Glucose"] < 99), "NEW_GLUCOSE_CAT"] = "Normal"
df.loc[(df["Glucose"] > 99) & (df["Glucose"] < 126), "NEW_GLUCOSE_CAT"] = "Secret"
df.loc[(df["Glucose"] > 126) & (df["Glucose"] < 200), "NEW_GLUCOSE_CAT"] = "High"

df.loc[(df["BloodPressure"] < 79), "NEW_BLOODPRESSURE_CAT"] = "Normal"
df.loc[(df["BloodPressure"] > 79) & (df["BloodPressure"] < 89), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S1"
df.loc[(df["BloodPressure"] > 89) & (df["BloodPressure"] < 123), "NEW_BLOODPRESSURE_CAT"] = "Hypertension_S2"


def set_insulin(row):
    if 16 <= row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_CAT"] = df.apply(set_insulin, axis=1)

df.columns = [col.upper() for col in df.columns]

df.drop(["BLOODPRESSURE", "GLUCOSE", "AGE", "BMI", "INSULIN"], inplace=True, axis=1)


# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col].astype(str))
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)


# ONE-HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols, drop_first=True)

# FEATURE SCALING

transformer = MinMaxScaler().fit(df[["DIABETESPEDIGREEFUNCTION"]])
df["DIABETESPEDIGREEFUNCTION"] = transformer.transform(df[["DIABETESPEDIGREEFUNCTION"]])
