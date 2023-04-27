from data_manipulator import DataManipulator
from data_preProcessing import data_preProcessing_script
from data_cleaner import DataCleaner
from data_exploration import exploration
import mlflow.tensorflow
import numpy as np
import pandas as pd
import dvc.api
import os
import datetime
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from statsmodels.tsa.stattools import adfuller, acf, pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from IPython.display import Markdown, display, Image
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join('..')))
# importing scripts
sys.path.insert(0, '../scripts/')


class ltsm_time:

    def __init__(self, WINDOW_SIZE, BATCH_SIZE, sales_data):
        self.WINDOW_SIZE = WINDOW_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        # data_agg = sales_data.groupby("Date").agg({"Sales": "mean"})
        self.SIZE = len(sales_data["Sales"])

        self.scaled_df = sales_data

        self.DateTrain = np.reshape(
            self.scaled_df.index.values[0:BATCH_SIZE], (-1, 1))
        self.DateValid = np.reshape(
            self.scaled_df.index.values[BATCH_SIZE:], (-1, 1))

        self.train_sales, self.valid_sales, self.TrainDataset, self.ValidDataset = self.prepare_data(WINDOW_SIZE,
                                                                                                     BATCH_SIZE,
                                                                                                     self.scaled_df)

    def scaler(self, df, columns, mode="minmax"):
        if (mode == "minmax"):
            minmax_scaler = MinMaxScaler()
            return pd.DataFrame(minmax_scaler.fit_transform(df), columns=columns), minmax_scaler
        elif (mode == "standard"):
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(df), columns=columns), scaler
        elif (mode == "robust"):
            scaler = RobustScaler()
            return pd.DataFrame(scaler.fit_transform(df), columns=columns), scaler

    def add_scaled_sales(self, df):
        scaled_sales, scaler_obj = self.scaler(
            df[["Sales"]], mode="minmax", columns="scaled_sales")
        df["scaled_sales"] = scaled_sales["scaled_sales"].to_list()
        return df, scaler_obj

    def prepare_data(self, WINDOW_SIZE, BATCH_SIZE, scaled_df):
        train_sales = scaled_df["Sales"].values[0:BATCH_SIZE].astype(
            'float32')
        valid_sales = scaled_df["Sales"].values[BATCH_SIZE:].astype(
            'float32')
        TrainDataset = self.windowed_dataset(
            train_sales, WINDOW_SIZE, BATCH_SIZE)
        ValidDataset = self.windowed_dataset(
            valid_sales, WINDOW_SIZE, BATCH_SIZE)

        return train_sales, valid_sales, TrainDataset, ValidDataset

    def train(self, EPOCHS, verbose=1):

        mlflow.set_experiment('Rossman-' + 'Lstm_model')

        mlflow.tensorflow.autolog(every_n_iter=2, log_models=True)

        mlflow.end_run()
        with mlflow.start_run(run_name="Lstm_model-Base-line"):

            model = Sequential()
            model.add(LSTM(20, input_shape=[None, 1], return_sequences=True))
            model.add(LSTM(10, input_shape=[None, 1]))
            model.add(Dense(1))
            model.compile(loss="huber_loss", optimizer='adam')
            model.summary()

            history = model.fit(self.TrainDataset, epochs=EPOCHS,
                                validation_data=self.ValidDataset, verbose=verbose)

        self.plot_history(history)

        return model, history

    def plot_history(self, history):
        fig = plt.figure(figsize=(12, 9))
        plt.plot(history.history['loss'], label="loss")
        plt.plot(history.history['val_loss'], label="val_loss")
        plt.legend()
        plt.show()

        return fig

    def model_forecast_test(self, model):

        series = self.scaled_df["Sales"].values[:, np.newaxis]

        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(self.WINDOW_SIZE, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(self.WINDOW_SIZE))
        ds = ds.batch(self.SIZE).prefetch(1)
        forecast = model.predict(ds)

        Results = forecast[self.BATCH_SIZE-self.WINDOW_SIZE:-1]
        Results1 = self.scaled_df.inverse_transform(Results.reshape(-1, 1))
        XValid1 = self.scaled_df.inverse_transform(
            self.valid_sales.reshape(-1, 1))

        fig, MAE, RMSE = self.plot_forcast(
            Results, Results1, XValid1, self.DateValid, self.WINDOW_SIZE)

        return forecast, fig, MAE, RMSE

    def plot_forcast(self, Results, Results1, XValid1, DateValid,  WINDOW_SIZE):
        fig = plt.figure(figsize=(30, 8))
        plt.title("LSTM Model Forecast Compared to Validation Data")

        plt.plot(DateValid.astype('datetime64'),
                 Results1, label='Forecast series')
        plt.plot(DateValid.astype('datetime64'), np.reshape(
            XValid1, (2*WINDOW_SIZE, 1)), label='Validation series')

        plt.xlabel('Date')
        plt.ylabel('Thousands of Units')
        plt.xticks(DateValid.astype('datetime64')[:, -1], rotation=90)
        plt.legend(loc="upper right")

        MAE = tf.keras.metrics.mean_absolute_error(
            XValid1[:, -1], Results[:, -1]).numpy()
        RMSE = np.sqrt(tf.keras.metrics.mean_squared_error(
            XValid1[:, -1], Results[:, -1]).numpy())

        textstr = "MAE = " + \
            "{:.3f}".format(MAE) + "  RMSE = " + "{:.3f}".format(RMSE)

        # place a text box in upper left in axes coords
        plt.annotate(textstr, xy=(0.87, 0.05), xycoords='axes fraction')
        plt.grid(True)

        plt.show()

        return fig, MAE, RMSE

    def windowed_dataset(self, series, window_size, batch_size):
        series = tf.expand_dims(series, axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(
            lambda window: window.batch(window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    def forcast_next_one_sale(self, model, sales):
        data_feat = None
        WINDOW_SIZE = 49
        try:
            data_feat = sales[["Sales", "Date"]]

            if (data_feat.shape[0] < 49):
                print("To make prediction, we need atleast data of 49 dates")
                return

            SIZE = len(self.scaled_df["Sales"])

            series = self.scaled_df["Sales"].values[:, np.newaxis]

            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
            ds = ds.batch(SIZE).prefetch(1)

            forecast = model.predict(ds)
            Results = list(forecast.reshape(
                1, forecast.shape[0] * forecast.shape[1])[0].copy())

            return Results

        except KeyError:
            print("Sales Data is expeceted to have Sales and Date columns")
            return False

    def forcast_next_sales(self, model, sales, daysToForcast=1):
        forcasts = []
        scaled_forcasts = []
        dates = []

        new_sales_df = sales.copy()
        while len(forcasts) < daysToForcast:
            forcast, scaled_forcast = self.forcast_next_one_sale(
                model, new_sales_df)
            forcasts += forcast

            scaled_forcasts += scaled_forcast
            size = len(new_sales_df["Sales"])

            truncated_sales = new_sales_df.tail(size - len(scaled_forcast))

            new_sales = truncated_sales['Sales'].to_list() + scaled_forcast
            next_dates = []

            for i in range(len(scaled_forcast)):
                next_date = new_sales_df["Date"].to_list(
                )[-1] + datetime.timedelta(days=1)
                next_dates.append(next_date)

            new_dates = truncated_sales['Date'].to_list() + next_dates
            new_sales_df = pd.DataFrame()
            new_sales_df["Date"] = new_dates
            new_sales_df["Sales"] = new_sales

        res_df = pd.DataFrame()
        res_df["Date"] = new_dates
        res_df["forcasts"] = forcasts

        return res_df
