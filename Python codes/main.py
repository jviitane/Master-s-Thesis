##
from random import shuffle
from sktime.classification.all import TimeSeriesForestClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier

# from sktime.classification.deep_learning.cnn import CNNClassifier

import tensorflow as tf
import numpy as np
from itertools import combinations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow import reduce_prod
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
import time
from datetime import timedelta
import datetime
import os
from typing import List
import seaborn as sb
import pandas as pd
import warnings
import sklearn.metrics
from tensorflow.keras import utils
import numpy as np
# import data
import tensorflow as tf
import models
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

print("Packages imported")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

## Function working area
def cnn(max_nodes=16, nb_layers=1, dropout=0, activation='relu',
        output_nodes=4, output_activation="softmax", kernel_size=(3, 3)):
    model = Sequential()
    for n in range(nb_layers):
        model.add(Conv2D(max_nodes / (2 ** n),
                         kernel_size,
                         activation=activation,
                         kernel_initializer='he_uniform',
                         padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(padding='same'))

    # Output layer of the model
    model.add(Flatten())
    model.add(Dense(max_nodes, activation=activation, kernel_initializer='he_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes, activation=output_activation))
    return model

def train_model(model, trainX, trainY, testX):
    start = time.time()
    name = type(model).__name__
    # print(f"Category: {category} | Model: {name}:")
    if not name == "LSTMFCNClassifier":
        trainX_ml = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]), order='F')
        testX_ml = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]), order='F')
    else:
        trainX_ml = trainX.copy()
        testX_ml = testX.copy()



    model.fit(trainX_ml, trainY)
    predY = model.predict(testX_ml)
    predY_proba = model.predict_proba(testX_ml)

    return predY, predY_proba


def lstm(max_nodes=16, nb_layers=1, dropout=0, activation='relu',
         output_nodes=4, output_activation="softmax"):
    model = Sequential()
    model.add(LSTM(units=max_nodes, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=max_nodes // 2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(output_nodes, activation=output_activation))

    return model


def nn_compile(model, category, max_nodes, nb_layers, dropout, activation, output_nodes, output_activation, kernel_size,
               opt, loss_func, metrics, epochs, batch_size, callback,
               trainX, trainY, validX, validY, testX, testY):
    if model not in ["cnn", "mlp", "lstm"]:
        print(f"Wrong model ({model})")
        return False
    if category not in range(17):
        print(f"Wrong category ({category})")
        return False

    print("Category", category)
    tf.keras.backend.clear_session()
    if model == "cnn":
        model = cnn(max_nodes, nb_layers, dropout, activation, output_nodes, output_activation, kernel_size)
        trainX = trainX[:, :, np.newaxis]
        validX = validX[:, :, np.newaxis]
        testX = testX[:, :, np.newaxis]
    elif model == "mlp":
        model = models.mlp(max_nodes, nb_layers, dropout, activation, output_nodes, output_activation)
    elif model == "lstm":
        model = lstm(max_nodes, nb_layers, dropout, activation, output_nodes, output_activation)

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2], trainX.shape[1]))
        validX = np.reshape(validX, (validX.shape[0], validX.shape[2], validX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[2], testX.shape[1]))

    model.compile(optimizer=opt, loss=loss_func, metrics=metrics)

    # %% Without data augmentation
    history = model.fit(trainX, trainY,
                        epochs=epochs,
                        validation_data=(validX, validY),
                        batch_size=batch_size,
                        callbacks=[callback],
                        verbose=2)

    prediction = model.predict(testX, verbose=0)
    print("Finished category", category)
    print()
    return prediction


def traditional_ml(category, model, trainX, trainY, testX, testY):
    start = time.time()
    name = type(model).__name__
    # print(f"Category: {category} | Model: {name}:")
    if not name == "LSTMFCNClassifier":
        trainX_ml = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
        testX_ml = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))
    else:
        trainX_ml = trainX.copy()
        testX_ml = testX.copy()
    if len(trainY.shape) < 2:
        trainY_ml = trainY
    else:
        trainY_ml = np.argmax(trainY, axis=1)

    '''
    trainX_ml = np.reshape(trainX, (trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
    testX_ml = np.reshape(testX, (testX.shape[0], testX.shape[1] * testX.shape[2]))
    trainY_ml = np.argmax(trainY[:, category], axis=1)
    testY_ml = np.argmax(testY[:, category], axis=1)
    '''

    model.fit(trainX_ml, trainY_ml)
    end = time.time()
    training_time = end - start
    start = time.time()
    predY = model.predict_proba(testX_ml)

    end = time.time()
    # print(f"Training time / batch {training_time:.2f} s")

    return predY


def prev_model(Yfull, Ytest, output_window):
    prevY = np.roll(Yfull, output_window, axis=0)[-len(Ytest):]
    if len(Yfull.shape) < 3: # IF not onehot-encoding
        prevY_proba = utils.to_categorical(np.expand_dims(prevY, axis=-1), num_classes=np.max(Ytest) + 1)
    else:
        prevY_proba = prevY
    return prevY, prevY_proba


def naive_model(Y):
    if len(Y.shape) < 3: # IF not onehot-encoding
        counts = np.apply_along_axis(np.bincount, axis=0, arr=Y)
        naiveY = np.full(Y.shape, np.argmax(counts, axis=0))
        naiveY_proba = utils.to_categorical(np.expand_dims(naiveY, axis=-1), num_classes=len(counts))

    else: # IF onehot-encoding
        counts = np.apply_along_axis(np.bincount, axis=0, arr=np.argmax(Y, axis=-1))
        naiveY = np.full(Y.shape[:-1], np.argmax(counts, axis=0))
        naiveY = utils.to_categorical(naiveY, num_classes=len(counts))
        naiveY_proba = naiveY

    return naiveY, naiveY_proba


def get_metrics(Ytrue, Ypred, Ypred_proba):
    if Ypred_proba.shape[-1] > 2:
        avg = 'weighted'
        y_score = Ypred_proba
        y_true = utils.to_categorical(Ytrue)
    else:
        avg = 'binary'
        y_score = Ypred_proba[:, -1]
        y_true = Ytrue
    metric_dict = {}
    metric_dict["accuracy"] = metrics.accuracy_score(Ytrue, Ypred)
    metric_dict["f_score"] = metrics.f1_score(Ytrue, Ypred, average=avg)
    metric_dict["recall"] = metrics.recall_score(Ytrue, Ypred, average=avg)
    metric_dict["precision"] = metrics.precision_score(Ytrue, Ypred, average=avg)
    metric_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_score)

    return metric_dict


def stock_data(stock_name="nokia", volatility_len=30, price_change_intervals=[]):
    index = pd.read_csv(f"{os.getcwd()}/Data/index_stock.csv", low_memory=False, sep=";")
    index["Date"] = pd.to_datetime(index["Date"], infer_datetime_format=True, yearfirst=False, dayfirst=True).dt.date
    index = index.sort_values("Date")
    index["return"] = index["Close"].pct_change(1, fill_method=None)

    df = pd.read_csv(f"{os.getcwd()}/Data/{stock_name}_stock2.csv", low_memory=False, sep=";", header=1, decimal=",")
    if df.shape[1] < 12:
        df = pd.read_csv(f"{os.getcwd()}/Data/{stock_name}_stock2.csv", low_memory=False, sep=";", header=0,
                         decimal=",")

    #df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, yearfirst=True, dayfirst=False).dt.date
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, yearfirst=False, dayfirst=True).dt.date
    df = df.sort_values("Date")
    df = df[df['Closing price'].notna()]

    df["price_change"] = df["Closing price"].pct_change(1, fill_method=None)
    #df["intraday_volatility"] = 0.5 * (np.log(df["High price"]) - np.log(df["Low price"])) ** 2 \
     #                           - ((2 * np.log(2) - 1) * (np.log(df["Closing price"]) - np.log(df["Opening price"])) ** 2)

    for timeframe in price_change_intervals:
        df[f"{timeframe}d_return"] = df["Closing price"].copy().pct_change(timeframe,
                                                                         fill_method=None)  # .shift(-timeframe)
        index[f"{timeframe}d_return"] = index["Close"].copy().pct_change(timeframe,
                                                                               fill_method=None)  # .shift(-timeframe)
        comb = df.merge(right=index, on="Date")
        df[f"{timeframe}d_excess_return"] = comb[f"{timeframe}d_return_x"] - comb[f"{timeframe}d_return_y"]

    df["volume_change"] = df["Total volume"].pct_change(1, fill_method=None)
    df[f"{volatility_len}_volatility"] = (df["price_change"].rolling(volatility_len).std() * (252 ** 0.5))
    df["night_change"] = (df["Opening price"].copy().shift(-1) / df["Closing price"]) - 1
    df["day_change"] = (df["Closing price"] / df["Opening price"]) - 1
    # f["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, yearfirst=True)

    return df


def calulate_net_scaled_volume(df: pd.DataFrame, date_column="registration_date", groupby="categories"):
    groups = df.groupby(by=[groupby, date_column, "transaction_type"])
    grouped_df = groups["volume"].sum().unstack(-1)
    grouped_df = grouped_df.fillna(0)
    #grouped_df = grouped_df.reset_index()
    grouped_df = grouped_df.rename(columns={10: "buy", 20: "sell"})
    grouped_df["net_volume"] = grouped_df["buy"] - grouped_df["sell"]
    grouped_df["total_volume"] = grouped_df["buy"] + grouped_df["sell"]
    grouped_df["net_scaled_volume"] = grouped_df["net_volume"] / grouped_df["total_volume"]

    '''
    groups2 = imbalances.groupby(by=[date_column])
    a = groups2["total_volume"].sum().reset_index().rename(columns={"total_volume": "day_volume"})
    imbalances = imbalances.merge(a, on=date_column)
    imbalances["volume_share"] = imbalances["total_volume"] / imbalances["day_volume"]
    '''
    sectors = df[groupby].unique()
    net_scaled_volumes = pd.DataFrame(index=df[date_column].unique())
    net_scaled_volumes.index.name = date_column

    for sector in sectors:
        a = grouped_df.loc[sector]
        net_scaled_volumes[f"{sector}_net_scaled_volume1"] = a["net_scaled_volume"]
    net_scaled_volumes = net_scaled_volumes.reset_index().fillna(0).sort_values(by=date_column)

    net_scaled_volumes[date_column] = pd.to_datetime(net_scaled_volumes[date_column], infer_datetime_format=True,
                                        yearfirst=True).dt.date
    #dates = grouped_df[date_column].unique()
    #categories = grouped_df[groupby].unique()
    #idx = pd.MultiIndex.from_product((dates, categories), names=[date_column, groupby])
    #grouped_df = grouped_df.set_index([date_column, groupby]).reindex(idx, fill_value=0)
    #grouped_df = grouped_df.reset_index().sort_values(by=date_column)
    # imbalances = imbalances.pivot_table(index="categories", columns='trading_date', values='net_index')
    # imbalances.to_csv('input_data.csv', index=False)
    return net_scaled_volumes

def calulate_cross_flows(df: pd.DataFrame, date_column="trading_date", groupby="categories"):


    household_groups = df.loc[df['sector'] == "Households"].groupby(by=[date_column, "price", "transaction_type"])
    household_volumes = household_groups["volume"].sum().unstack(-1)
    household_volumes = household_volumes.fillna(0)
    #household_volumes = household_volumes.reset_index()
    household_volumes = household_volumes.rename(columns={10: "buy", 20: "sell"})
    household_volumes["buy_household"] = 0
    household_volumes["sell_household"] = 0

    total_groups = df.groupby(by=[date_column, "price", "transaction_type"])
    total_volumes = total_groups["volume"].sum().unstack(-1)
    total_volumes = total_volumes.fillna(0)
    #total_volumes = total_volumes.reset_index()
    total_volumes = total_volumes.rename(columns={10: "buy", 20: "sell"})

    flows = pd.DataFrame(index=df[date_column].unique())
    flows.index.name = date_column

    category_groups = df.groupby(by=[groupby, date_column, "price", "transaction_type"])
    category_volumes = category_groups["volume"].sum().unstack(-1)
    category_volumes = category_volumes.fillna(0)
    #category_volumes = category_volumes.reset_index(0)
    category_volumes = category_volumes.rename(columns={10: "buy", 20: "sell"})

        #household_volumes = household_volumes.set_index(["trading_date", 'price'])
        #total_volumes = total_volumes.set_index(["trading_date", 'price'])

        #Estimate how much households have bought/sold from/to each other based on how much they've been buying/selling
        # from total volume. At each price and date.

    for sector in df[groupby].unique():
        a = category_volumes.loc[sector]
        a["buy_household"] = a["buy"] * (household_volumes["sell"] / total_volumes["sell"])
        a["sell_household"] = a["sell"] * (household_volumes["buy"] / total_volumes["buy"])
        a = a.reset_index().fillna(0).drop("price", axis="columns")

        a = a.groupby(by=[date_column]).sum()
        flows[f"{sector}_buy_household_share"] = a["buy_household"]/a["buy"]
        flows[f"{sector}_sell_household_share"] = a["sell_household"]/a["sell"]
        #flows[f"{sector}_total_household_share"] = (a["sell_household"]+a["buy_household"])\
        #                                             /(a["sell"]+a["buy"])
    flows = flows.reset_index().fillna(0).sort_values(by=date_column)



    flows[date_column] = pd.to_datetime(flows[date_column], infer_datetime_format=True,
                                                    yearfirst=True).dt.date
    return flows
def calculate_group_activity(df: pd.DataFrame, date_column="trading_date", groupby="categories"):
    total_groups = df.groupby(by=[groupby])
    total_counts = total_groups["owner_id"].nunique()

    daily_groups = df.groupby(by=[groupby, date_column])
    daily_counts = daily_groups["owner_id"].nunique()

    daily_percentages = daily_counts/total_counts

    group_activity = pd.DataFrame(index=df[date_column].unique())
    group_activity.index.name = date_column

    for sector in df[groupby].unique():
        group_activity[f"{sector}_activity"] = daily_percentages.loc[sector]
    group_activity = group_activity.reset_index().fillna(0).sort_values(by=date_column)



    group_activity[date_column] = pd.to_datetime(group_activity[date_column], infer_datetime_format=True,
                                                    yearfirst=True).dt.date
    return group_activity

def calulate_volume_contributions(df: pd.DataFrame, date_column="trading_date", groupby="categories"):
    total_groups = df.groupby(by=[date_column,"transaction_type"])
    total_volumes = total_groups["volume"].sum().unstack(-1)
    total_volumes = total_volumes.fillna(0)
    #total_volumes = total_volumes.reset_index()
    total_volumes = total_volumes.rename(columns={10: "buy", 20: "sell"})

    category_groups = df.groupby(by=[groupby, date_column, "transaction_type"])
    category_volumes = category_groups["volume"].sum().unstack(-1)
    category_volumes = category_volumes.fillna(0)
    category_volumes = category_volumes.rename(columns={10: "buy", 20: "sell"})


    volume_contributions = pd.DataFrame(index=df[date_column].unique())
    volume_contributions.index.name = date_column
        #household_volumes = household_volumes.set_index(["trading_date", 'price'])
        #total_volumes = total_volumes.set_index(["trading_date", 'price'])

        #Estimate how much households have bought/sold from/to each other based on how much they've been buying/selling
        # from total volume. At each price and date.

    for sector in df[groupby].unique():
        a = category_volumes.loc[sector]
        volume_contributions[f"{sector}_buy_volume_share"] = a["buy"]/total_volumes["buy"]
        volume_contributions[f"{sector}_sell_volume_share"] = a["sell"]/total_volumes["sell"]
        #volume_contributions[f"{sector}_total_volume_share"] = (a["sell"]+a["buy"])\
                                                     #/(total_volumes["sell"]+total_volumes["buy"])
    volume_contributions = volume_contributions.reset_index().fillna(0).sort_values(by=date_column)



    volume_contributions[date_column] = pd.to_datetime(volume_contributions[date_column], infer_datetime_format=True,
                                                    yearfirst=True).dt.date
    return volume_contributions

'''
def calulate_net_flows(df: pd.DataFrame, date_column="trading_date"):

    household_groups = df.loc[df['sector'] == "Households"].groupby(by=["trading_date", "price", "transaction_type"])
    household_volumes = household_groups["volume"].sum().unstack(-1)
    #household_volumes = household_volumes.fillna(0)
    #household_volumes = household_volumes.reset_index()
    household_volumes = household_volumes.rename(columns={10: "buy", 20: "sell"})
    household_volumes["buy_household"] = 0
    household_volumes["sell_household"] = 0

    total_groups = df.groupby(by=["trading_date", "price", "transaction_type"])
    total_volumes = total_groups["volume"].sum().unstack(-1)
    #total_volumes = total_volumes.fillna(0)
    #total_volumes = total_volumes.reset_index()
    total_volumes = total_volumes.rename(columns={10: "buy", 20: "sell"})

    #household_volumes = household_volumes.set_index(["trading_date", 'price'])
    #total_volumes = total_volumes.set_index(["trading_date", 'price'])

    #Estimate how much households have bought/sold from/to each other based on how much they've been buying/selling
    # from total volume. At each price and date.
    household_volumes["buy_household"] = household_volumes["buy"] * (household_volumes["sell"] / total_volumes["sell"])
    household_volumes["sell_household"] = household_volumes["sell"] * (household_volumes["buy"] / total_volumes["buy"])
    household_volumes = household_volumes.reset_index().fillna(0).drop("price", axis="columns")

    household_volumes = household_volumes.groupby(by=["trading_date"]).sum()
    household_volumes["buy_household_share"] = household_volumes["buy_household"]/household_volumes["buy"]
    household_volumes["sell_household_share"] = household_volumes["sell_household"]/household_volumes["sell"]
    household_volumes["total_household_share"] = (household_volumes["sell_household"]+household_volumes["buy_household"])\
                                                 /(household_volumes["sell"]+household_volumes["buy"])
    household_volumes = household_volumes.reset_index().fillna(0)



    household_volumes[date_column] = pd.to_datetime(household_volumes[date_column], infer_datetime_format=True,
                                                    yearfirst=True).dt.date
    return household_volumes
'''

def calulate_net_scaled_volume2(df: pd.DataFrame, date_column="registration_date", groupby="categories"):
    groups = df.groupby(by=[groupby, "owner_id", date_column, "transaction_type"])
    grouped_df = groups["volume"].sum().unstack(-1)
    grouped_df = grouped_df.fillna(0)
    #grouped_df = grouped_df.reset_index()
    grouped_df = grouped_df.rename(columns={10: "buy", 20: "sell"})
    grouped_df["net_volume"] = grouped_df["buy"] - grouped_df["sell"]
    grouped_df["net_scaled_volume"] = grouped_df["net_volume"] / (grouped_df["buy"] + grouped_df["sell"])
    groups2 = grouped_df.groupby(by=[groupby, date_column])
    grouped_df = groups2["net_scaled_volume"].mean()
    sectors = df[groupby].unique()
    net_scaled_volumes = pd.DataFrame(index=df[date_column].unique())
    net_scaled_volumes.index.name = date_column

    for sector in sectors:
        a = grouped_df.loc[sector]
        net_scaled_volumes[f"{sector}_net_scaled_volume2"] = a
    net_scaled_volumes = net_scaled_volumes.reset_index().fillna(0).sort_values(by=date_column)

    net_scaled_volumes[date_column] = pd.to_datetime(net_scaled_volumes[date_column], infer_datetime_format=True,
                                        yearfirst=True).dt.date

    return net_scaled_volumes
def calulate_net_scaled_volume_variances(df: pd.DataFrame, date_column="registration_date", groupby="categories"):
    groups = df.groupby(by=[groupby, "owner_id", date_column, "transaction_type"])
    grouped_df = groups["volume"].sum().unstack(-1)
    grouped_df = grouped_df.fillna(0)
    #grouped_df = grouped_df.reset_index()
    grouped_df = grouped_df.rename(columns={10: "buy", 20: "sell"})
    grouped_df["net_volume"] = grouped_df["buy"] - grouped_df["sell"]
    grouped_df["net_scaled_volume"] = grouped_df["net_volume"] / (grouped_df["buy"] + grouped_df["sell"])
    groups2 = grouped_df.groupby(by=[groupby, date_column])
    grouped_df = groups2["net_scaled_volume"].std()
    sectors = df[groupby].unique()
    net_scaled_volumes = pd.DataFrame(index=df[date_column].unique())
    net_scaled_volumes.index.name = date_column

    for sector in sectors:
        a = grouped_df.loc[sector]
        net_scaled_volumes[f"{sector}_nsv_std"] = a
        net_scaled_volumes[f"{sector}_nsv_var"] = a**2
    net_scaled_volumes = net_scaled_volumes.reset_index().fillna(0).sort_values(by=date_column)

    net_scaled_volumes[date_column] = pd.to_datetime(net_scaled_volumes[date_column], infer_datetime_format=True,
                                        yearfirst=True).dt.date
    return net_scaled_volumes


def categorize(df: pd.DataFrame, on: List[str]):
    categories = pd.Categorical(zip(*[df[field] for field in on]))  # type: ignore
    category_labels = {
        code: name for code, name in set(zip(categories.codes, categories))}

    df['categories'] = [" ".join(s) for s in categories]
    return category_labels


def create_input_data(df_X, df_Y, input_window, output_window, threshold, neural_net, lead_lag, stock_compare=True):
    y = df_Y.copy().to_numpy().T
    x = df_X.T.copy()
    windows = x.rolling(input_window, axis=0)
    x = np.array([window.to_numpy().T for window in windows if window.to_numpy().shape[0] == input_window])
    # x = x[1:]  # OVERLAPPING PERIODS
    y = y[input_window - 1:]  # Start predicted values after the first window

    # Shift data one window length more.
    if lead_lag:
        x = x[:-output_window]
        y = y[output_window:]

    mapping = {0: "buying", 1: "daytrading", 2: "do nothing", 3: "selling"}

    if stock_compare:
        pass

    elif threshold == 0:
        buy_mask = y < 0

        sell_mask = y >= 0
        y[buy_mask] = 0
        y[sell_mask] = 1
    else:
        buy_mask = y < -threshold
        daytrade_mask = (y >= -threshold) & (y <= threshold)
        idle_mask = y == 0
        sell_mask = y > threshold
        y[buy_mask] = 0
        y[daytrade_mask] = 1
        y[sell_mask] = 2

    y = y.astype("int16")
    if neural_net:
        if df_Y.shape[0] == 1:
            y = utils.to_categorical(np.expand_dims(y, axis=-1))
        else:
            y = utils.to_categorical(y)
    return x, y

def remove_bad_data(df, date_column="trading_date"):

    df = df.loc[df[date_column] == df[date_column]] # Remove Entries without trading date
    bad_households_index = df.loc[(df["sector"] == "Households") & ((df["gender"] == "no-gender") | (df["age"] == "no-age"))].index
    df = df.drop(bad_households_index)
    return df
def get_descriptive_data(df, date_column="trading_date"):
    return
## Get Data

STOCKS = {
    "Nokia": "FI0009000681",
    #"Sonera": "FI0009007371",
    "Fortum": "FI0009007132",
    "UPM": "FI0009005987",
    "Outokumpu": "FI0009002422",
    #"Metso": "FI0009007835",
    "Sampo": "FI0009003305",
    #"Rautaruukki": "FI0009003552",
    "NesteOil": "FI0009013296",
    "NordeaBank": "FI0009902530",
    "Elisa": "FI0009007884",
    "NokianRenkaat": "FI0009005318",
    "TeliaSonera": "SE0000667925",
    "Wärtsilä": "FI0009003727",
    "Elektrobit": "FI0009007264",
    "Raisio": "FI0009002943",
    "YIT": "FI0009800643",
    "FSecure": "FI0009801310",
    "StoraEnso": "FI0009005961",
    "Metsä": "FI0009000665",
    "Kesko": "FI0009000202",
    #"Comptel": "FI0009008221",
    "Tieto": "FI0009000277",
    "Konecranes": "FI0009005870",
    #"Pohjola": "FI0009003222",
    #"Elcoteq": "FI0009006738",
    #"Perlos": "FI0009007819",
    "Outotec": "FI0009014575",
    "Uponor": "FI0009002158",
    "Innofactor": "FI0009007637",
    "Kemira": "FI0009004824",
    "Orion": "FI0009800346",
    "Huhtamäki": "FI0009000459",
    #"Stonesoft": "FI0009801302",
    #"GeoSentric": "FI0009004204",
    #"SaunalahtiGroup": "FI0009008569",
    #"AldataSolution": "FI0009007918",
    #"Ramirent": "FI0009007066",
    #"Cramo": "FI0009900476",
    #"Sponda": "FI0009006829",
    "Cencorp": "FI0009006951",
    "Teleste": "FI0009007728",
    #"PKCGroup": "FI0009006381",
    #"Eimo": "FI0009007553",
    #"SoonCommunications": "FI0009006787",
    "Finnair": "FI0009003230",
    #"Basware": "FI0009008403",
    "Tecnotree": "FI0009010227",
    "HKScan": "FI0009006308",
    #"BiotieTherapies": "FI0009011571",
    "DovreGroup": "FI0009008098",
    "eQ": "FI0009008676",
    "AfarakGroup": "FI0009800098",
    "SSH": "FI0009008270",
    #"Talvivaara": "FI0009014716",
    "Efore": "FI0009900054",
    "TrainersHouse": "FI0009008122",
    "Fiskars": "FI0009000400",
    "Marimekko": "FI0009007660",
}
STOCKS = {
    "Nokia": "FI0009000681",
    "Fortum": "FI0009007132",
    "UPM": "FI0009005987",
    "Outokumpu": "FI0009002422",
    "Sampo": "FI0009003305",
    "Elisa": "FI0009007884",
    "NokianRenkaat": "FI0009005318",
    "Konecranes": "FI0009005870",
    "StoraEnso": "FI0009005961",
    "YIT": "FI0009800643",
    "Metsä": "FI0009000665",
    "Kesko": "FI0009000202",
    "Tieto": "FI0009000277",
}
DATA = {}
DESC_DATA = {}
date_column = "trading_date" #SET THIS ALWAYS to trading_date

for stock_name in STOCKS:
    print("Loading", stock_name)
    DATA[stock_name] = {}
    df = pd.read_csv(f"{os.getcwd()}/Data/{stock_name}_categorized.csv", low_memory=False)
    df = remove_bad_data(df, date_column)
    #for fields in ['sector']:#[['sector', 'gender'], ['sector'], ['gender']]:
    fields = ['sector']
    categories = pd.Categorical(zip(*[df[field] for field in fields]))
    df['categories'] = [" ".join(s) for s in categories]
    groupby = "categories"
    df = df.loc[df[date_column] == df[date_column]]
    df = df.loc[df[groupby] == df[groupby]]
    net_scaled_volume = calulate_net_scaled_volume(df, date_column, groupby)
    net_scaled_volume2 = calulate_net_scaled_volume2(df, date_column, groupby)
    cross_flows = calulate_cross_flows(df, date_column, groupby)
    volume_contributions = calulate_volume_contributions(df, date_column, groupby)
    nsv_variance = calulate_net_scaled_volume_variances(df, date_column, groupby)
    group_activity = calculate_group_activity(df, date_column, groupby)
    data = [d.set_index(date_column) for d in [net_scaled_volume, net_scaled_volume2, nsv_variance,
                                               cross_flows, volume_contributions, group_activity]]
    DATA[stock_name] = pd.concat(data, axis=1).reset_index()

print("Data fetched")

##
accuracies = {}
STOCKS = {
    "Nokia": "FI0009000681",
    #"Sonera": "FI0009007371",
    "Fortum": "FI0009007132",
    "UPM": "FI0009005987",
    "Outokumpu": "FI0009002422",
    #"Metso": "FI0009007835",
    "Sampo": "FI0009003305",
    #"Rautaruukki": "FI0009003552",
    "NesteOil": "FI0009013296",
    "NordeaBank": "FI0009902530",
    "Elisa": "FI0009007884",
    "NokianRenkaat": "FI0009005318",
    "TeliaSonera": "SE0000667925",
    "Wärtsilä": "FI0009003727",
    "Elektrobit": "FI0009007264",
    "Raisio": "FI0009002943",
    "YIT": "FI0009800643",
    "FSecure": "FI0009801310",
    "StoraEnso": "FI0009005961",
    "Metsä": "FI0009000665",
    "Kesko": "FI0009000202",
    #"Comptel": "FI0009008221",
    "Tieto": "FI0009000277",
    "Konecranes": "FI0009005870",
    #"Pohjola": "FI0009003222",
    #"Elcoteq": "FI0009006738",
    #"Perlos": "FI0009007819",
    "Outotec": "FI0009014575",
    "Uponor": "FI0009002158",
    "Innofactor": "FI0009007637",
    "Kemira": "FI0009004824",
    "Orion": "FI0009800346",
    "Huhtamäki": "FI0009000459",
    #"Stonesoft": "FI0009801302",
    #"GeoSentric": "FI0009004204",
    #"SaunalahtiGroup": "FI0009008569",
    #"AldataSolution": "FI0009007918",
    #"Ramirent": "FI0009007066",
    #"Cramo": "FI0009900476",
    #"Sponda": "FI0009006829",
    "Cencorp": "FI0009006951",
    "Teleste": "FI0009007728",
    #"PKCGroup": "FI0009006381",
    #"Eimo": "FI0009007553",
    #"SoonCommunications": "FI0009006787",
    "Finnair": "FI0009003230",
    #"Basware": "FI0009008403",
    "Tecnotree": "FI0009010227",
    "HKScan": "FI0009006308",
    #"BiotieTherapies": "FI0009011571",
    "DovreGroup": "FI0009008098",
    "eQ": "FI0009008676",
    "AfarakGroup": "FI0009800098",
    "SSH": "FI0009008270",
    #"Talvivaara": "FI0009014716",
    "Efore": "FI0009900054",
    "TrainersHouse": "FI0009008122",
    "Fiskars": "FI0009000400",
    "Marimekko": "FI0009007660",
}
STOCKS = {
    "Nokia": "FI0009000681",
    "Fortum": "FI0009007132",
    "UPM": "FI0009005987",
    "Outokumpu": "FI0009002422",
    "Sampo": "FI0009003305",
    "Elisa": "FI0009007884",
    "NokianRenkaat": "FI0009005318",
    "Konecranes": "FI0009005870",
    "StoraEnso": "FI0009005961",
    "YIT": "FI0009800643",
    "Metsä": "FI0009000665",
    "Kesko": "FI0009000202",
    #"Comptel": "FI0009008221",
    "Tieto": "FI0009000277",
}


input_window = 1
output_window = 1
threshold = 0
neural_net = False #Transforms labels to one-hot-encoding
lead_lag = False
stock_compare = False
stock_predict = True
investor_predict = not stock_predict
date_column = "trading_date"
start_date = datetime.date(2000, 1, 5)
split_date = datetime.date(2005, 1, 5)
end_date = datetime.date(2009, 10, 8)
# end_date = datetime.date(2009, 11, 10)
price_change_intervals = [output_window]  # list(range(5, 6, 1))

varX = [#"net_scaled_volume",
        #"household_share",
        "Government institutions",
        "Households",
        "Non-Financial corporations",
        "Financial-Insurance corporations",
        "Non-Profit institutions",
        "EU institutions",
        "Rest-World",
        "Non-EU institutions",
    ]
varY = [
        # "EU institutions",
        # "Financial-Insurance corporations",
        # "Government institutions",
        #"Households-net_scaled_rolling",
        #"Households female-net_scaled_rolling",
        #"Households_net_scaled_volume1",
        #"Households male-net_scaled_rolling",
        #"d_return",
        "Closing price",
        # "Non-Financial corporations",
        # "intraday_volatility",
        # "price_change",
        # "night_change"
    ]



if output_window > input_window:
    print("Warning! Output window may overlap with output window!")

if stock_compare and investor_predict:
    print("Warning! Using stock comparison when trying to predict investor trading direction")

STOCK_DATA = {}
Xtrain, Ytrain, Xtest, Ytest, Ynaive_proba, Yprev_proba, Ynaive, Yprev = [], [], [], [], [], [], [], []
for stock_name in STOCKS:
    stock_df = stock_data(stock_name, volatility_len=input_window,
                          price_change_intervals=price_change_intervals).fillna(0)
    stock_df = stock_df.loc[stock_df["Date"] < end_date]
    stock_df = stock_df.loc[stock_df["Date"] >= start_date]
    STOCK_DATA[stock_name] = stock_df.pivot_table(values=stock_df.columns,
                                                  columns="Date", dropna=False)

if stock_compare:
    asd = list(combinations(STOCKS, 2))
    shuffle(asd)
    print("Loading combination")

else:
    asd = [STOCKS]

#LOOPS
for stocks in asd:

    stocks = list(stocks)
    shuffle(stocks)
    # Shuffle so that the same stock isn't always first in pair
    if stock_compare:
        print(stocks, end=" - ")
        df_X_pair = []
        df_Y_pair = []
    else:
        print("Loading input data")

    for stock_name in stocks:
        if not stock_compare:
            print(stock_name, end=" - ")
        trading_data = DATA[stock_name]

        pivot_stock = STOCK_DATA[stock_name]
        if stock_compare and pivot_stock.columns[0] > start_date:
            print(stock_name, "doesn't have long enough time series")
            break


        pivot_trading_data = trading_data.pivot_table(columns=date_column, dropna=False, fill_value=0)

        df = pd.concat([pivot_stock, pivot_trading_data],
                       join="outer", sort=True).loc[:, pivot_stock.columns].fillna(0)
        df_X = df.loc[df.index.str.contains("|".join(varX), regex=True)].copy()
        df_Y = df.loc[df.index.str.contains("|".join(varY), regex=True)].copy()

        if stock_compare:
            df_Y.index = [stocks.index(stock_name)]
            # df_X.index = [stocks.index(stock_name)]
            df_X_pair.append(df_X)
            df_Y_pair.append(df_Y)

        else:
            # If we are predicting returns, calculate periods returns at this point
            if any(["Closing price" in s for s in varY]):
                df_Y.loc[["Closing price"]] = df_Y.loc[["Closing price"]].pct_change(output_window, axis=1, fill_method=None).copy()


            Xtrain_sample, Ytrain_sample = create_input_data(df_X.loc[:, :split_date],
                                                             df_Y.loc[:, :split_date],
                                                             input_window, output_window, threshold,
                                                             neural_net, lead_lag, stock_compare)

            Xtest_sample, Ytest_sample = create_input_data(df_X.loc[:, split_date:],
                                                           df_Y.loc[:, split_date:],
                                                             input_window, output_window, threshold,
                                                             neural_net, lead_lag, stock_compare)

            if Xtrain_sample.shape[0] > 0:
                Xtrain.append(Xtrain_sample)
                Ytrain.append(Ytrain_sample)

            if Xtest_sample.shape[0] > 0:
                Ynaive_sample, Ynaive_proba_sample = naive_model(Ytest_sample)
                Ynaive_proba.append(Ynaive_proba_sample)
                Ynaive.append(Ynaive_sample)

                _, Yfull_sample = create_input_data(df_X, df_Y, input_window, output_window,
                                             threshold, neural_net, lead_lag, stock_compare)
                Yprev_sample, Yprev_proba_sample = prev_model(Yfull_sample, Ytest_sample, output_window)
                Yprev_proba.append(Yprev_proba_sample)
                Yprev.append(Yprev_sample)
                if len(Yprev_proba_sample) != len(Ytest_sample):
                    print(stock_name)
                    break
                Xtest.append(Xtest_sample)
                Ytest.append(Ytest_sample)



    # When comparing stocks, price and trading paths must be aligned between pairs.
    if stock_compare and len(df_X_pair) > 1:
        df_X_pair = pd.concat(df_X_pair, join="outer", sort=True).fillna(0)
        df_Y_pair = pd.concat(df_Y_pair, join="outer", sort=True).fillna(method="pad", axis=1)
        df_Y_pair = df_Y_pair.pct_change(output_window, axis=1, fill_method=None)
        df_Y_pair.loc["outperformer"] = df_Y_pair.idxmax(axis=0)
        df_Y_pair = df_Y_pair.loc[["outperformer"]]

        Xtrain_sample, Ytrain_sample = create_input_data(df_X_pair.loc[:, :split_date], df_Y_pair.loc[:, :split_date],
                                                         input_window, output_window, threshold, neural_net, lead_lag, stock_compare)

        Xtest_sample, Ytest_sample = create_input_data(df_X_pair.loc[:, split_date:], df_Y_pair.loc[:, split_date:],
                                                       input_window, output_window, threshold, neural_net, lead_lag, stock_compare)

        Ynaive_sample, Ynaive_proba_sample = naive_model(Ytest_sample)
        Ynaive_proba.append(Ynaive_proba_sample)
        Ynaive.append(Ynaive_sample)

        _, Yfull_sample = create_input_data(df_X_pair, df_Y_pair, input_window, output_window,
                                            threshold, neural_net, lead_lag, stock_compare)
        Yprev_sample, Yprev_proba_sample = prev_model(Yfull_sample, Ytest_sample, output_window)
        Yprev_proba.append(Yprev_proba_sample)
        Yprev.append(Yprev_sample)


        Xtrain.append(Xtrain_sample)
        Ytrain.append(Ytrain_sample)
        Xtest.append(Xtest_sample)
        Ytest.append(Ytest_sample)

Xtrain = np.concatenate(Xtrain)
Ytrain = np.concatenate(Ytrain)
Xtest = np.concatenate(Xtest)
Ytest = np.concatenate(Ytest)
Ynaive_proba = np.concatenate(Ynaive_proba)
Ynaive = np.concatenate(Ynaive)
Yprev_proba = np.concatenate(Yprev_proba)
Yprev = np.concatenate(Yprev)
print()
print("Input data fetched")

##
ml_models = {
    "LDA": LinearDiscriminantAnalysis(),
    "LogReg": LogisticRegression(max_iter=100000, class_weight="balanced"),
    #"SVM (linear kernel)": SVC(kernel='linear', probability=True),
    #"SVM (RBF kernel)": SVC(kernel='rbf', probability=True),
    "RF": RandomForestClassifier(n_estimators=111, class_weight="balanced"),
    "RF_optim": RandomForestClassifier(criterion='gini', max_depth=4, max_features='log2', n_estimators=72, class_weight="balanced"),
    "RF_optim2": RandomForestClassifier(criterion='gini', max_depth=4, max_features='sqrt', n_estimators=63, class_weight="balanced"),
    "XGB": XGBClassifier(n_estimators=111),
    #"TSRF": TimeSeriesForestClassifier(n_estimators=55),
    #"AB": AdaBoostClassifier(n_estimators=55),
    #"LSTMFCN": LSTMFCNClassifier(batch_size=32, n_epochs=200, verbose=True, dropout=0.8, filter_sizes=(32, 64, 32), kernel_sizes=(6, 4, 2)),
}

ml_predictions = {}
conf_matrixes = {}
feature_importances = {}
for category in range(Ytrain.shape[1]):
    category_name = df_Y.index[category]
    print(f"Starting category {category}: {category_name}")
    category_predictions = {}
    category_conf_matrixes = {}

    for model_name, model in ml_models.items():
        Ypredict, Ypredict_proba = train_model(model, Xtrain[:, 1:], Ytrain[:, category], Xtest[:, 1:])
        category_predictions[model_name] = get_metrics(Ytest[:, category], Ypredict, Ypredict_proba)
        category_conf_matrixes[model_name] = metrics.confusion_matrix(Ytest[:, category], Ypredict)
        #feature_importances[model_name] = model.feature_importances_.reshape((Xtest.shape[1], Xtest.shape[2]), order='F').sum(axis=1)
        #feature_importances[model_name] = pd.DataFrame(feature_importances[model_name], index=df_X.index)
        print(model_name)
        print(pd.DataFrame.from_dict(category_conf_matrixes[model_name]))


    naive_metrics = get_metrics(Ytest[:, category], Ynaive[:, category], Ynaive_proba[:, category])
    category_predictions["Naive"] = naive_metrics

    prev_metrics = get_metrics(Ytest[:, category], Yprev[:, category], Yprev_proba[:, category])
    category_predictions["Previous period"] = prev_metrics


    ml_predictions[category_name] = category_predictions
    conf_matrixes[category_name] = category_conf_matrixes
    print(pd.DataFrame.from_dict(ml_predictions[category_name]).to_string())
    print("Finished category", category_name)
    print()


##
for category in range(Xtrain.shape[1]):
    category_name = df_X.index[category]
    print(f"Starting category {category}: {category_name}")
    category_predictions = {}
    category_conf_matrixes = {}

    for model_name, model in ml_models.items():
        Ypredict, Ypredict_proba = train_model(model, Xtrain[:, [category]], Ytrain[:, 0], Xtest[:, [category]])
        category_predictions[model_name] = get_metrics(Ytest[:, 0], Ypredict, Ypredict_proba)
        category_conf_matrixes[model_name] = metrics.confusion_matrix(Ytest[:, 0], Ypredict)

        #print(model_name)
        #print(pd.DataFrame.from_dict(category_conf_matrixes[model_name]))

    naive_metrics = get_metrics(Ytest[:, 0], Ynaive[:, 0], Ynaive_proba[:, 0])
    category_predictions["Naive"] = naive_metrics

    prev_metrics = get_metrics(Ytest[:, 0], Yprev[:, 0], Yprev_proba[:, 0])
    category_predictions["Previous period"] = prev_metrics


    ml_predictions[category_name] = category_predictions
    conf_matrixes[category_name] = category_conf_matrixes
    print(pd.DataFrame.from_dict(ml_predictions[category_name]).to_string())
    print("Finished category", category_name)
    print()


##
for category, result in ml_predictions.items():
    result = pd.DataFrame.from_dict(result, orient='index').T
    # normalized_result = result.div(result.max(axis=1), axis=0)
    # category_name = CATEGORY_MAPPING[category]
    # category_name = df_Y.index[category]
    ax = sb.heatmap(result, annot=result, cbar=False, vmin=0, vmax=1, cmap=sb.color_palette("coolwarm", as_cmap=True))
    ax.set(ylabel="Metric", xlabel="Model", title=(category, category_name))
    fig = ax.get_figure()
    fig.savefig(f"{date_column} ML {category} {category_name}  results")
    ax.clear()

a = pd.DataFrame.from_dict(conf_matrixes, orient='index')

## Statistical testing.
import statsmodels.api as sm
dfx = pd.DataFrame(Xtrain[:, :, -1], columns=list(df_X.index))
dfx2 = pd.DataFrame(Xtest[:, :, -1], columns=list(df_X.index))
dfy = pd.DataFrame(Ytrain, columns=list(df_Y.index))
dfy2 = pd.DataFrame(Ytest, columns=list(df_Y.index))

log_reg = sm.Logit(dfy, dfx).fit()
print(log_reg.summary())
yhat = log_reg.predict(dfx2[output_window:-output_window:output_window])
ypred = yhat.round().astype(int)
ypred_proba = np.zeros((len(yhat), 2))
ypred_proba[:, -1] = yhat.copy()
ypred_proba[:, 0] = 1 - ypred_proba[:, -1]
print(get_metrics(dfy2[output_window:-output_window:output_window], ypred, ypred_proba))
print()
print()


## Statistical testing.
import statsmodels.api as sm
dfx = pd.DataFrame(Xtrain[:, :, -1], columns=list(df_X.index))
dfx2 = pd.DataFrame(Xtest[:, :, -1], columns=list(df_X.index))
dfy = pd.DataFrame(Ytrain, columns=list(df_Y.index))
dfy2 = pd.DataFrame(Ytest, columns=list(df_Y.index))
for index in df_Y.index:
    print(index)
    yt = dfy.copy()[[index]]
    yt2 = dfy2.copy()[[index]]
    xt = dfx.copy().loc[:, dfx.columns != index]
    xt[f"{index}-1"] = dfx.copy()[[index]].copy().shift(-input_window)
    xt2 = dfx2.copy().loc[:, dfx2.columns != index]
    xt2[f"{index}-1"] = dfx2.copy()[[index]].copy().shift(-input_window)

    #xt = sm.add_constant(xt)
    #xt2 = sm.add_constant(xt2)
    # min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    # yt = min_max_scaler.fit_transform(yt)
    yt = yt[output_window:-output_window:output_window]
    xt = xt[output_window:-output_window:output_window]
    log_reg = sm.Logit(yt, xt).fit()
    print(log_reg.summary())
    yhat = log_reg.predict(xt2[output_window:-output_window:output_window])
    ypred = yhat.round().astype(int)
    ypred_proba = np.zeros((len(yhat), 2))
    ypred_proba[:, -1] = yhat.copy()
    ypred_proba[:, 0] = 1 - ypred_proba[:, -1]
    print(get_metrics(yt2[output_window:-output_window:output_window], ypred, ypred_proba))
    print()
    print()

## Plotterinoes

plt.figure(figsize=(10, 10))
# Plot model's accuracy information based on epochs
plt.subplot(211)
plt.title(f'Accuracy')
plt.plot(accuracies, label='LogReg')
plt.plot(naive_accuracies, label='Naive')
# plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.ylim([0, 1])
# plt.xlim(min_pri)
plt.legend(loc='lower right')
plt.savefig(f'Category{i}.png')

##


# %% Compile model
#nn_model = "cnn"
# nn_model = "mlp"
nn_model = "lstm"
max_nodes = 64
nb_layers = 2
dropout = 0.7
epochs = 500
kernel_size = (3, 3)
batch_size = 64
learning_rate = 0.001
activation = 'relu'
loss_func = "binary_crossentropy"
training_metrics = ['accuracy']
output_nodes = 1
output_activation = "sigmoid"
callback = EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0, verbose=1, mode='max',
                         restore_best_weights=True)
opt = tf.optimizers.Adam(learning_rate=learning_rate)

# NN
nn_predictions = {}
print(f"Running {epochs} epochs with batch size of {batch_size} and learning rate of {learning_rate}.")
for category in range(1):
    nn_predictions[category] = {}
    # stratify_base1 = np.argmax(Y[:, category], axis=1).squeeze()
    # stratify_base1 = Y[:, category]
    '''
    if stock_predict == False and lead_lag == False and any(["Household" in ''.join(s) for s in df_X.index]):
        category_name = df_Y.index[category]
        X_copy = np.delete(X, df_X.index==category_name, axis=1)#[:, np.newaxis]#np.delete(X, category, axis=1)
    else:
        X_copy = X.copy()

    Y_copy = Y.copy()[:, category]
    first_split = int(X_copy.shape[0]*2/3)
    second_split = int(X_copy.shape[0]*3/4)
    #second_split = 2800
    '''
    testX, testY = Xtest.copy(), Ytest.copy()  # ""[:, category]

    trainX, trainY = Xtrain.copy()[:10000], Ytrain.copy()[:10000]  # [:, category]
    validX, validY = Xtrain.copy()[10000:], Ytrain.copy()[10000:]  # [500:, category]
    '''
    trainX, trainY = Xtrain.copy(), Ytrain.copy()  # ""[:, category]

    validX, validY = Xtest.copy()[:7500], Ytest.copy()[:7500]  # [:, category]
    testX, testY = Xtest.copy()[7500:], Ytest.copy()[7500:]  # [500:, category]
'''
    # trainX, validX, trainY, validY = train_test_split(X, Y[:, category], test_size=0.4, random_state=10, stratify=stratify_base1)
    # stratify_base2 = np.argmax(validY, axis=1).squeeze()
    # stratify_base2 = validY
    # validX, testX, validY, testY = train_test_split(validX, validY, test_size=0.5, random_state=10, stratify=stratify_base2)
    # y_true = np.argmax(testY[:, category], axis=1)

    predY_proba = nn_compile(nn_model, category, max_nodes, nb_layers, dropout, activation, output_nodes,
                             output_activation, kernel_size,
                             opt, loss_func, training_metrics, epochs, batch_size, callback,
                             trainX, trainY,
                             validX, validY,
                             testX, testY)
    predY = predY_proba.round().squeeze()
    nn_predictions[category][nn_model] = get_metrics(testY, predY, predY_proba)
    print(pd.DataFrame.from_dict(nn_predictions[category]).to_string())
    print("Finished category", category)

for category, result in nn_predictions.items():
    result = pd.DataFrame.from_dict(result, orient='index').T
    # normalized_result = result.div(result.max(axis=1), axis=0)
    # category_name = CATEGORY_MAPPING[category]
    category_name = df_Y.index[category]
    ax = sb.heatmap(result, annot=result, cbar=False, vmin=0, vmax=1, cmap=sb.color_palette("coolwarm", as_cmap=True))
    ax.set(ylabel="Metric", xlabel="Model", title=(category, category_name))
    fig = ax.get_figure()
    fig.savefig(f"NN {category} {category_name}  results")
    ax.clear()

##
