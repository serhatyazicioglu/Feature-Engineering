"""
AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
Hits: 1986-1987 sezonundaki isabet sayısı
HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
Runs: 1986-1987 sezonunda takımına kaç sayı kazandırdı
RBI: Bir vurucunun vuruş yaptıgında kaç tane oyuncuya koşu yaptırdığı.
Walks: Karşı oyuncuya kaç defa hata yaptırdığı
Years: Oyuncunun major liginde kaç sene oynadığı
CAtBat: Oyuncunun kariyeri boyunca kaç kez topa vurduğu
CHits: Oyuncunun kariyeri boyunca kaç kez isabetli vuruş yaptığı
CHmRun: Oyucunun kariyeri boyunca kaç kez en değerli vuruşu yaptığı
CRuns: Oyuncunun kariyeri boyunca takımına kaç tane sayı kazandırdığı
CRBI: Oyuncunun kariyeri boyunca kaç tane oyuncuya koşu yaptırdığı
CWalks: Oyuncun kariyeri boyunca karşı oyuncuya kaç kez hata yaptırdığı
League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from helpers.eda import grab_col_names
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("datasets/hitters.csv")

df.isnull().sum()
df.describe().T

df.corr() > 0.9

check_df(df)
# OUTLIERS

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.10)
    quartile3 = dataframe[col_name].quantile(0.90)
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
    replace_with_thresholds(df, col)

# MISSING VALUES

df.isnull().values.any()
df.isnull().sum()

msno.bar(df)
plt.show()

df["Salary"].fillna(df["Salary"].mean(), inplace=True)

# FEATURE ENGINEERING

df["player_season_success"] = (df["AtBat"] * 4 / 100 + df["Hits"] * 10 / 100 + df["HmRun"] * 12 / 100 + df[
    "Runs"] * 12 / 100 + df["RBI"] * 10 / 100 + df["Walks"] * 12 / 100 + df[
                                   "Assists"] * 10 / 100 + df["PutOuts"] * 10 / 100 - df["Errors"] * 20 / 100)

df["career_AtBat"] = df["CAtBat"] / df["Years"]
df["career_Hits"] = df["CHits"] / df["Years"]
df["career_HmRun"] = df["CHmRun"] / df["Years"]
df["career_Runs"] = df["CRuns"] / df["Years"]
df["career_RBI"] = df["CRBI"] / df["Years"]
df["career_Walks"] = cwalks = df["CWalks"] / df["Years"]

df["player_general_success"] = (
        df["career_AtBat"] * 10 / 100 + df["career_Hits"] * 15 / 100 + df["career_HmRun"] * 20 / 100
        + df["career_Runs"] * 20 / 100
        + df["career_RBI"] * 20 / 100 + df["career_Walks"] * 15 / 100)

df["Is_Success"] = df["player_season_success"].apply(lambda x: 1 if x > df["player_season_success"].mean() else 0)
df.groupby("player_general_success").agg({"Salary": "mean"})

df.drop(["CAtBat", "CHits", "CHmRun", "CRuns", "CRBI"], inplace=True, axis=1)

df.columns = [col.upper() for col in df.columns]


# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

# FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler(feature_range=(0, 10)).fit(df[["PLAYER_GENERAL_SUCCESS"]])
df["PLAYER_GENERAL_SUCCESS"] = transformer.transform(df[["PLAYER_GENERAL_SUCCESS"]])

transformer = MinMaxScaler(feature_range=(0, 10)).fit(df[["PLAYER_SEASON_SUCCESS"]])
df["PLAYER_SEASON_SUCCESS"] = transformer.transform(df[["PLAYER_SEASON_SUCCESS"]])
