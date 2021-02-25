import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from helpers.eda import grab_col_names

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("datasets/titanic.csv")

"""
Survived: hayatta kalma(0 = Hayır, 1 =Evet)
Pclass: bilet sınıfı(1 = 1., 2 = 2., 3 = 3.)
Sex: cinsiyet
Sibsp: Titanik’teki kardeş/eş sayısı
Parch: Titanik’teki ebeveynlerin/çocukların sayısı
Ticket: bilet numarası
Fare: ücret
Cabin: kabin numarası
Embarked: biniş limanı
"""

# Outliers
df.describe().T

sns.boxplot(x="Age", data=df)
plt.show()

sns.boxplot(x="Fare", data=df)
plt.show()


# OUTLIERS
def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
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


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Fare", index=True)


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


replace_with_thresholds(df, "Fare")

check_outlier(df, "Fare")

# Missing Values

df.isnull().values.any()
df.isnull().sum()

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

msno.matrix(df)
plt.show()


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


missing_vs_target(df, "Survived", na_name)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df["Embarked"].value_counts()

df.groupby(["Embarked", "Sex"]).agg({"Survived": ["mean", "count"]})

# Feature Engineering
df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype("int")
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_WORD_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df["Age"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["Age"] >= 18) & (df["Age"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 56), "NEW_AGE_CAT"] = "old"

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_AGE_CAT"] = "youngmale"
df.loc[(df['Sex'] == 'male') & (
        (df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_AGE_CAT"] = "oldmale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_AGE_CAT"] = "youngfemale"
df.loc[(df['Sex'] == 'female') & (
        (df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_AGE_CAT"] = "oldfemale"

titanic["NEW_ECONOMY"] = pd.qcut(titanic['FARE'], 3, labels=["BRONZE", "SILVER", "GOLD"])

df.columns = [col.upper() for col in df.columns]

df.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]

for col in binary_cols:
    df = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

df = one_hot_encoder(df, ohe_cols)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 10)).fit(df[["AGE"]])
df["AGE"] = scaler.transform(df[["AGE"]])
df["AGE"].describe().T

df.to_pickle("titanic.pkl")

dff = pd.read_pickle("titanic.pkl")
dff.head()

from helpers.data_prep import *
