import math
import os
import time
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
debug = True

class SolarData:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pre_process(pd.read_csv(csv_path), debug=False)


def d_print(debug, st):
    if debug:
        print(st)


def get_dataset_features(df, y=None, useless_column=None):
    col_y = pd.DataFrame()
    df_feature = df.copy()
    if useless_column is None:
        useless_column = []
    if check_dataframe_column(df_feature, y):
        col_y = df_feature.pop(y)
    for col in useless_column:
        if check_dataframe_column(df_feature, col):
            df_feature = df_feature.drop(columns=col)
    return df_feature.values, col_y.values


def check_dataframe_column(df, col):
    if col in df:
        return True
    else:
        print("Column", col, "does not exist in the DataFrame.")
        return False


def filter_dataframe_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


# Work with 24h
def day_part(hour):
    if hour in [4, 5]:
        return 1  # "dawn"
    elif hour in [6, 7]:
        return 2  # "early morning"
    elif hour in [8, 9, 10]:
        return 3  # "late morning"
    elif hour in [11, 12, 13]:
        return 4  # "noon"
    elif hour in [14, 15, 16]:
        return 5  # "afternoon"
    elif hour in [17, 18, 19]:
        return 6  # "evening"
    elif hour in [20, 21, 22]:
        return 7  # "night"
    elif hour in [23, 24, 1, 2, 3]:
        return 8  # "midnight"


def get_features(df, y=None, useless_column=None):
    col_y = None
    df_feature = df.copy()
    if useless_column is None:
        useless_column = []
    if check_dataframe_column(df_feature, y):
        col_y = df_feature.pop(y)
        for col in useless_column:
            if check_dataframe_column(df_feature, col):
                df_feature = df_feature.drop(columns=col)
    return df_feature, col_y


# Work With a dataframe
def extract_datetime_feature(df, date_column, remove=True, is_timestamp=True):
    new_df = df.copy()
    if is_timestamp:
        new_df[date_column] = pd.to_datetime(new_df[date_column], format='%m/%d/%y %I:%M %p')
    new_df["year"] = new_df[date_column].dt.year
    new_df["month"] = new_df[date_column].dt.month
    new_df["day"] = new_df[date_column].dt.day
    new_df["hour"] = new_df[date_column].dt.hour
    new_df["minute"] = new_df[date_column].dt.minute
    new_df["day_of_week"] = new_df[date_column].dt.day_of_week.astype(int)
    new_df["day_of_year"] = new_df[date_column].dt.day_of_year.astype(int)
    new_df['day_part'] = new_df['hour'].apply(day_part)
    new_df['is_year_start'] = new_df[date_column].dt.is_year_start.astype(int)
    new_df['is_quarter_start'] = new_df[date_column].dt.is_quarter_start.astype(int)
    new_df['is_month_start'] = new_df[date_column].dt.is_month_start.astype(int)
    new_df['is_month_end'] = new_df[date_column].dt.is_month_end.astype(int)
    new_df['is_weekend'] = np.where(new_df['day_of_week'].isin([5, 6]), 1, 0)
    # df["season"] = df["Date"].dt
    if remove:
        new_df.drop(columns="Date")
    return new_df


def augment_cycle_variable(df, replace=False, columns=None):
    new_df = df.copy()
    if columns is None:
        columns = []
    for col in columns:
        if check_dataframe_column(df, col):
            new_col_sin = col + "__sin"
            new_col_cos = col + "__cos"
            values = set(new_df[col])
            num_values = len(set(new_df[col]))
            if 0 in values:
                new_df[new_col_sin] = np.sin(new_df[col] * (2. * np.pi / num_values))
                new_df[new_col_cos] = np.cos(new_df[col] * (2. * np.pi / num_values))
            # shift by one
            else:
                new_df[new_col_sin] = np.sin((new_df[col] - 1) * (2. * np.pi / num_values))
                new_df[new_col_cos] = np.cos((new_df[col] - 1) * (2. * np.pi / num_values))
            if replace:
                new_df = new_df.drop(columns=col)
    return new_df


def augment_celsius(df, replace=False, columns=None):
    new_df = df.copy()
    if columns is None:
        columns = []

    for col in columns:
        if check_dataframe_column(df, col):
            new_col = col.replace("C", "F")
            if replace:
                new_df[col] = new_df.apply(lambda x: (9 / 5) * x[col] + 32, axis=1)
            else:
                new_df[new_col] = new_df.apply(lambda x: (9 / 5) * x[col] + 32, axis=1)
        return new_df


def get_outliers_from_feature(df, feature_from, feature_to, threshold='mid', secure=True):
    assert threshold in ('mid', 'upper'), "Invalid side '{}'".format(threshold)
    lower_bound = 0
    upper_bound = 0
    data_outliers = pd.DataFrame()
    pure_data = df.copy()
    for val in set(df[feature_from]):
        q1 = df[df[feature_from] == val][feature_to].quantile(0.25)
        q3 = df[df[feature_from] == val][feature_to].quantile(0.75)
        iqr = q3 - q1
        if threshold == 'mid':
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        elif threshold == "upper":
            lower_bound = q1 - 3.0 * iqr
            upper_bound = q3 + 3.0 * iqr
        if lower_bound == 0 or upper_bound == 0 and secure:
            print(f"prevent feature {feature_from} to remove zeros outliers (change secure to remove)")
            outliers = pd.DataFrame()
        else:
            outliers = df[df[feature_from] == val].loc[(df[feature_to] <= lower_bound) |
                                                       (df[feature_to] >= upper_bound)]
        if not outliers.empty:
            pure_data = filter_dataframe_rows_by_values(df=pure_data, col=feature_from, values=outliers[feature_from])

        data_outliers = data_outliers.append(outliers, ignore_index=True)
    d_print(debug, f"{data_outliers.shape[0]} outliers rows removed")
    return pure_data, data_outliers


# MANUAL CLEANING PART
def hand_cleaning(df, join_colum, bad_path="", debug=True):
    clean_df = df.copy()
    zeros = clean_df[(clean_df["P (kW)"] == 0.0) & (clean_df["Ta (C)"] == 0.0) & (clean_df["Tm (C)"] == 0.0) &
                     (clean_df["I15 (W/m2)"] == 0) & (clean_df["I3 (W/m2)"] == 0.0) &
                     (clean_df["hour"] < 21) & (clean_df["hour"] >= 4)]
    d_print(debug, f"{zeros.shape[0]} mid zeros rows removed")
    # Remove bad row (T15 and T3 without P (kW))
    inconsistent = clean_df[(df["P (kW)"] == 0.0) & (clean_df["Ta (C)"] != 0.0) & (clean_df["Tm (C)"] != 0.0) & (
            clean_df["I15 (W/m2)"] != 0) & (
                                    clean_df["I3 (W/m2)"] != 0.0)]
    d_print(debug, f"{inconsistent.shape[0]} inconsistent rows removed")

    clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, zeros[join_colum])
    clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, inconsistent[join_colum])
    # Load hand cleaned data
    if os.path.isfile(bad_path):
        bad_data = pd.read_csv(bad_path)
        clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, bad_data[join_colum])
        d_print(debug, f"{inconsistent.shape[0]} bad_data rows removed")
    else:
        bad_data = pd.DataFrame()

    drop_df = clean_df[((clean_df["hour"] == 21) & (clean_df["P (kW)"] > 150) |
                        (clean_df["hour"] == 20) & (clean_df["P (kW)"] > 300) |
                        (clean_df["hour"] == 19) & (clean_df["P (kW)"] > 400) |
                        (clean_df["hour"] == 18) & (clean_df["P (kW)"] > 400) |
                        (clean_df["hour"] == 17) & (clean_df["P (kW)"] > 400) |
                        # (clean_df["hour"] == 6) & (clean_df["P (kW)"] > 800) |
                        # (clean_df["hour"] == 5) & (clean_df["P (kW)"] > 600) |
                        # (clean_df["hour"] == 4) & (clean_df["P (kW)"] > 700) |
                        (clean_df["hour"] == 3) & (clean_df["P (kW)"] > 150))]
                        # (clean_df["hour"] == 4) & (clean_df["I3 (W/m2)"] > 600))

    d_print(debug, f"{drop_df.shape[0]} drop_table rows removed")
    clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, drop_df[join_colum])
    data_removed = pd.concat([zeros, inconsistent, bad_data], axis=0, ignore_index=True)
    clean_df = filter_dataframe_rows_by_values(clean_df, join_colum, data_removed[join_colum])
    if debug:
        d_print(debug, "Before hand cleaning")
        df.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "After hand cleaning")
        clean_df.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
    return clean_df, data_removed


def pre_process(csv_path="./DBs/train.txt", rounding=0, d=True, extension=True):
    global debug
    debug = d

    if os.path.exists(csv_path) is False:
        raise FileNotFoundError

    solar_data = pd.read_csv(csv_path)

    # FEATURES ENGINEERING
    d_print(debug, "BEGIN FEATURE ENGINEERING...")
    dataframe = extract_datetime_feature(df=solar_data, date_column='Date', remove=False, is_timestamp=True)
    d_print(debug, "datatime extracted!")
    d_print(debug, "END FEATURE ENGINEERING")

    # AUGMENTATION
    d_print(debug, "BEGIN AUGMENTATION ...")
    dataframe = augment_celsius(df=dataframe, replace=False, columns=["Ta (C)", "Tm (C)"])
    d_print(debug, "temperature augmented!")
    dataframe = augment_cycle_variable(df=dataframe, replace=False, columns=["Time Frame", "month", "day",
                                                                             "hour", "minute", "day_of_year",
                                                                             "day_of_week"])
    d_print(debug, "cycles augmented!")
    # CLEANING
    d_print(debug, "BEGIN CLEANING ...")
    dataframe, data_removed = hand_cleaning(df=dataframe, join_colum="Date", bad_path='./DBs/SolarPark/bad_data2.csv',
                                            debug=debug)
    d_print(debug, "hand cleaned!")
    dataframe = dataframe.replace(2013, 2012)
    d_print(debug, "year fused!")
    d_print(debug, "END AUGMENTATION")

    outliers = pd.DataFrame()
    for feature in ["hour__sin", "hour__cos", "Time Frame__sin", "Time Frame__cos", "day__sin", "day__cos"]:
        pure_data, outliers_removed = get_outliers_from_feature(df=dataframe, feature_from=feature,
                                                                feature_to="P (kW)", threshold="upper")
        outliers = pd.concat([outliers, outliers_removed], axis=0, ignore_index=True)
    if debug:
        d_print(debug, "Before outliers cleaning")
        dataframe.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "After outliers cleaning")
        pure_data.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
    if extension:
        extended_frame = dataframe.copy()

        # EXTENTION
        d_print(debug, "BEGIN EXTENSION ...")
        print(f"{data_removed.shape[0]} data_remove ROWS!")
        augmented_data = pd.DataFrame()
        progress_bar = tqdm(range(data_removed.shape[0]+pd.concat([data_removed, outliers], axis=0,
                                                                  ignore_index=True).shape[0]))


        # standard frame
        for idx, removed in data_removed.iterrows():
            for col in ["P (kW)", "I3 (W/m2)", "I15 (W/m2)"]:
                q1 = extended_frame[extended_frame['Time Frame__sin'] == removed['Time Frame__sin']][col].quantile(0.25)
                q3 = extended_frame[extended_frame['Time Frame__sin'] == removed['Time Frame__sin']][col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.0 * iqr
                upper_bound = q3 + 1.0 * iqr
                if math.isnan(lower_bound) | math.isnan(lower_bound):
                    new_val = 0
                else:
                    new_val = round(np.random.uniform(0, upper_bound, size=10).mean(), 2)
                removed[col] = new_val
                augmented_data = augmented_data.append(removed, ignore_index=True)
            if debug:
                progress_bar.update(1)
        d_print(debug, f"{augmented_data.shape[0]} rows extended to dataframe!")
        extended_frame = pd.concat([extended_frame, augmented_data], axis=0, ignore_index=True)

        # pure dataframe
        data_removed = pd.concat([data_removed, outliers_removed, outliers], axis=0, ignore_index=True)
        pure_extended = pure_data.copy()


        for idx, removed in data_removed.iterrows():
            for col in ["P (kW)", "I3 (W/m2)", "I15 (W/m2)"]:
                q1 = pure_extended[pure_extended['I15 (W/m2)'] == removed['I15 (W/m2)']][col].quantile(0.25)
                q3 = pure_extended[pure_extended['I15 (W/m2)'] == removed['I15 (W/m2)']][col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.0 * iqr
                upper_bound = q3 + 1.0 * iqr
                if math.isnan(lower_bound) | math.isnan(lower_bound):
                    new_val = 0
                else:
                    new_val = round(np.random.uniform(0, upper_bound, size=10).mean(), 2)
                removed[col] = new_val
                augmented_data = augmented_data.append(removed, ignore_index=True)
            if debug:
                progress_bar.update(1)
        d_print(debug, f"{augmented_data.shape[0]} rows extended to dataframe!")
        pure_extended = pd.concat([pure_extended, augmented_data], axis=0, ignore_index=True)

        d_print(debug, "END EXTENSION")
    if debug & extension:
        d_print(debug, "NORMAL - Before extensions")
        dataframe.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "NORMAL - After extensions")
        extended_frame.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "PURE - Before extensions")
        pure_data.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
        plt.show()
        d_print(debug, "PURE - After extensions")
        pure_extended.boxplot(column='P (kW)', by='Time Frame', figsize=(5, 5))
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
    if extension:
        return extended_frame, pure_extended
    else:
        return dataframe, pure_data


class BestParams:
    _regressor = None
    _t_size = 0
    _split = 0
    type = None

    def __init__(self, type):
        self.type = type

    def update(self, bestparams, t_size, split):
        print("Best score replaced: ")
        print("New: ", bestparams.best_score_)
        if self._regressor != None:
            print("Old: ", self._regressor.best_score_)
        self._regressor = bestparams
        self._t_size = t_size
        self._split = split

    def isbetter(self, score):
        if self._regressor is None: return True
        # print(score.cv_results_['mean_test_score'], self._bestparams['cv_results_'])
        return score.cv_results_['mean_test_score'] > self._regressor.cv_results_['mean_test_score']

    def printVals(self):
        print("Regressor type: ", self.type)
        print("_bestscore: ", self._regressor.best_score_)
        print("_bestparams: ", self._regressor.best_params_)
        print("_t_size: ", self._t_size)
        print("_split: ", self._split)

    def isBetterRegressor(self, newRegressor):
        if self._regressor is None: return True
        # print(score.cv_results_['mean_test_score'], self._bestparams['cv_results_'])
        return newRegressor.best_score_ < self._regressor.best_score_


def getOptimalRegressor(model_params, train_x, train_y, test_size=[0.2], n_split=[3, 4, 5]):
    bestparams = BestParams("Simple regressor")

    result = None
    for data_test_size in test_size:
        for split in n_split:
            cross_val = ShuffleSplit(n_splits=split, test_size=data_test_size, random_state=42)

            for model_name, mp in model_params.items():
                grid = GridSearchCV(estimator=mp['model'],
                                    param_grid=mp['params'],
                                    cv=cross_val,
                                    verbose=2,
                                    return_train_score=False)

                grid.fit(train_x, train_y)

                if bestparams.isbetter(grid):
                    bestparams.update(grid, data_test_size, split)

    return bestparams


def getVotingRegressor(model_params, train_x, train_y, test_size=[0.2], n_split=[3, 4, 5]):
    bestparams = BestParams("Voting Regressor")

    votingRegressorModel = VotingRegressor([
        ('rf', RandomForestRegressor()),
        ('lr', LinearRegression(**model_params['LinearRegressor']['params']))
    ])

    params = {}
    for param, value in model_params['RandomForestRegressor']['params'].items():
        params['rf__' + param] = value

    for param, value in model_params['LinearRegressor']['params'].items():
        params['lr__' + param] = value

    for data_test_size in test_size:
        for split in n_split:
            cross_val = ShuffleSplit(n_splits=split, test_size=data_test_size, random_state=42)

            grid = GridSearchCV(estimator=votingRegressorModel,
                                param_grid=params,
                                cv=cross_val,
                                verbose=2,
                                return_train_score=False)

            grid.fit(train_x, train_y)

        if bestparams.isbetter(grid):
            bestparams.update(grid, data_test_size, split)

    return bestparams


def getBestRegressor(model_params, train_x, train_y, test_size=[0.2], n_split=[3, 4, 5], useVotingRegressor=False):
    # Use GridSearchCV with RandomForest and LinearRegressor
    # Output -> il migliore regressore fra questi due
    regressor = getOptimalRegressor(model_params, train_x, train_y, test_size, n_split)

    # Use GridSearchCV with VotingRegressor.
    votingRegressor = None
    if useVotingRegressor:
        votingRegressor = getVotingRegressor(model_params, train_x, train_y, test_size, n_split)

    # Print results
    regressor.printVals()
    if votingRegressor is not None:
        votingRegressor.printVals()

    if regressor.isBetterRegressor(votingRegressor._regressor):
        return regressor
    else:
        return votingRegressor
