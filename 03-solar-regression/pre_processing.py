import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class SolarData:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pre_process(pd.read_csv(csv_path), debug=False)


def d_print(debug, st):
    if debug:
        print(st)


def filter_dataframe_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


# Work with 24h
def day_part(hour):
    if hour in [4, 5]:
        return "dawn"
    elif hour in [6, 7]:
        return "early morning"
    elif hour in [8, 9, 10]:
        return "late morning"
    elif hour in [11, 12, 13]:
        return "noon"
    elif hour in [14, 15, 16]:
        return "afternoon"
    elif hour in [17, 18, 19]:
        return "evening"
    elif hour in [20, 21, 22]:
        return "night"
    elif hour in [23, 24, 1, 2, 3]:
        return "midnight"


# Work With a dataframe
def extract_datetime_feature(df, date_column, remove=True, is_timestamp=True):
    new_df = df.copy()
    if is_timestamp:
        new_df[date_column] = pd.to_datetime(new_df[date_column], format='%m/%d/%y %I:%M %p')
    new_df["year"] = new_df[date_column].dt.year
    new_df["month"] = new_df[date_column].dt.month
    new_df["day"] = new_df[date_column].dt.day
    new_df["hour"] = new_df[date_column].dt.hour
    new_df["day_of_week"] = new_df[date_column].dt.day_of_week
    new_df["day_of_year"] = new_df[date_column].dt.day_of_year
    new_df['day_part'] = new_df['hour'].apply(day_part)
    new_df['is_year_start'] = new_df[date_column].dt.is_year_start
    new_df['is_quarter_start'] = new_df[date_column].dt.is_quarter_start
    new_df['is_month_start'] = new_df[date_column].dt.is_month_start
    new_df['is_month_end'] = new_df[date_column].dt.is_month_end
    new_df['is_weekend'] = np.where(new_df['day_of_week'].isin([5, 6]), 1, 0)
    # df["season"] = df["Date"].dt
    if remove:
        new_df.drop(columns="Date")
    return new_df


def augment_temperature(df, replace=False, columns=None):
    new_df = df.copy()
    if columns is None:
        columns = []

    for col in columns:
        new_col = col.replace("C", "F", 1)
        new_df[new_col] = new_df.apply(lambda x: (9 / 5) * x[col] + 32, axis=1)
        if replace:
            new_df.drop(columns=col)
        return new_df


def get_outliers_from_feature(df, feature_from, feature_to, feature_value, bound='mid'):
    assert bound in ('mid', 'upper'), "Invalid side '{}'".format(bound)
    lower_bound = 0
    upper_bound = 0
    q1 = df[df[feature_from] == feature_value][feature_to].quantile(0.25)
    q3 = df[df[feature_from] == feature_value][feature_to].quantile(0.75)
    iqr = q3 - q1
    if bound == 'mid':
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    elif bound == "upper":
        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr
    if lower_bound != 0 and upper_bound != 0:
        return df[df[feature_from] == feature_value].loc[(df[feature_to] <= lower_bound) |
                                                         (df[feature_to] >= upper_bound)]
    else:
        print("failed to remove outliers")
        return df


# MANUAL CLEANING PART
def hand_cleaning(df, join_colum, bad_path="", debug=True):
    # Remove all empty values
    clean_df = df[(df["P (kW)"] != 0.0) & (df["Ta (C)"] != 0.0) & (df["Tm (C)"] != 0.0) & (df["I15 (W/m2)"] != 0) & (
            df["I3 (W/m2)"] != 0.0)]
    # Remove bad row (T15 and T3 without P (kW))
    bad_pow = df[(df["P (kW)"] == 0.0) & (df["Ta (C)"] != 0.0) & (df["Tm (C)"] != 0.0) & (df["I15 (W/m2)"] != 0) & (
            df["I3 (W/m2)"] != 0.0)]
    row_removed = bad_pow.shape[0]
    clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, bad_pow[join_colum])
    # Load hand cleaned data
    if os.path.isfile(bad_path):
        bad_data = pd.read_csv(bad_path)
        clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, bad_data[join_colum])
        row_removed = row_removed + bad_data.shape[0]
    d_print(debug, f"{row_removed} total row row removed")

    clean_df = clean_df[(df["hour"] == 21) & (clean_df["P (kW)"] < 80) | (clean_df["hour"] != 21)]
    clean_df = clean_df[(df["hour"] == 20) & (clean_df["P (kW)"] < 100) | (clean_df["hour"] != 20)]
    clean_df = clean_df[(df["hour"] == 19) & (clean_df["P (kW)"] < 220) | (clean_df["hour"] != 19)]
    clean_df = clean_df[(df["hour"] == 18) & (clean_df["P (kW)"] < 400) | (clean_df["hour"] != 18)]
    clean_df = clean_df[(df["hour"] == 17) & (clean_df["P (kW)"] < 400) | (clean_df["hour"] != 17)]
    clean_df = clean_df[(df["hour"] == 6) & (clean_df["P (kW)"] < 700) | (clean_df["hour"] != 6)]
    clean_df = clean_df[(df["hour"] == 5) & (clean_df["P (kW)"] < 500) | (clean_df["hour"] != 5)]
    clean_df = clean_df[(df["hour"] == 4) & (clean_df["P (kW)"] < 100) | (clean_df["hour"] != 4)]
    clean_df = clean_df[(df["hour"] == 3) & (clean_df["P (kW)"] < 80) | (clean_df["hour"] != 3)]
    clean_df = clean_df[(df["hour"] == 4) & (clean_df["I3 (W/m2)"] < 200) | (clean_df["hour"] != 4)]
    # TODO add or remove more
    if debug:
        d_print(debug, "Before hand cleaning")
        df.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "After hand cleaning")
        clean_df.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
    return clean_df


def get_outliers(df, feature):
    q1 = df[feature].quantile(0.45)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    return df.loc[(df[feature] < lower_bound) | (df[feature] > upper_bound)]


def pre_process(csv_path="./DBs/train.txt", rounding=0, debug=True):
    if os.path.exists(csv_path) is False:
        raise FileNotFoundError

    solar_data = pd.read_csv(csv_path)

    # FEATURES ENGINEERING
    d_print(debug, "BEGIN FEATURE ENGINEERING...")
    dataframe = extract_datetime_feature(df=solar_data, date_column='Date', remove=False, is_timestamp=True)
    d_print(debug, "datatime extracted!")
    d_print(debug, "END FEATURE ENGINEERING")

    # CLEANING
    d_print(debug, "BEGIN CLEANING ...")
    dataframe = hand_cleaning(df=dataframe, join_colum="Date", bad_path='./DBs/SolarPark/bad_data.csv', debug=debug)
    d_print(debug, "had cleaned!")

    # AUGMENTATION
    d_print(debug, "BEGIN AUGMENTATION ...")
    dataframe = augment_temperature(df=dataframe, replace=False, columns=["Ta (C)", "Tm (C)"])
    d_print(debug, "temperature augmented!")
    dataframe = dataframe.replace(2013, 2012)
    d_print(debug, "year fused!")
    d_print(debug, "END AUGMENTATION")

    pure_data = dataframe.copy()
    # Remove outliers
    for val in set(dataframe["hour"]):
        outliers = get_outliers_from_feature(df=pure_data, feature_from="hour", feature_to="P (kW)",
                                             feature_value=val, bound="mid")
        if not outliers.empty:
            pure_data = filter_dataframe_rows_by_values(df=pure_data, col="Date", values=outliers["Date"])
    d_print(debug, "removed outliers ...")
    if debug:
        d_print(debug, "Before outliers cleaning")
        dataframe.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "After outliers cleaning")
        pure_data.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
    d_print(debug, "END CLEANING")

    # NORMALIZATION
    d_print(debug, "BEGIN NORMALIZATION ...")
    # Rounding data
    if rounding != 0:
        dataframe.round(rounding)
        print(f"Rounded data by {rounding}!")
    # TODO add standard o min_max_scaler
    d_print(debug, "END NORMALIZATION ...")
    return dataframe, pure_data
