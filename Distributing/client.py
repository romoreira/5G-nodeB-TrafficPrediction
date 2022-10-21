from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from tsai.all import *
import argparse
import sys
import os
from tsai.all import *

sys.path.append("../../")
from fedlab.core.client.manager import ActiveClientManager
from fedlab.core.client.trainer import SGDClientTrainer
from fedlab.utils.dataset.sampler import RawPartitionSampler
from fedlab.core.network import DistNetwork
from fedlab.utils.functional import AverageMeter, evaluate
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp
from hyperopt import space_eval

import time
from fastai.callback.tracker import EarlyStoppingCallback
import gc

import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tsai
from tsai.all import *

model_name = "ResCNN"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#Creating the model
class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]

def get_data(client_id):

    file_name = "dataset.pkl"
    df = pd.read_pickle(file_name)

    '''Choose one dataset for each client'''
    if client_id == 1:
        df = df['ElBorn']
    elif client_id == 2:
        df = df['LesCorts']
    elif client_id == 3:
        df = df['PobleSec']
    else:
        print("Number of clients > dataset")

    df.set_index(df.iloc[:, 0].name)
    df.index.names = ['TimeStamp']

    data_columns = list(df.columns.values)
    data = df[data_columns].values
    data = np.clip(data, 0.0, np.percentile(data.flatten(), 99))  # we use 99% as the threshold
    df[data_columns] = data

    aggregated_time_series = np.sum(data, axis=1)
    df_ts = pd.DataFrame()
    df_ts['data'] = aggregated_time_series   # Plot in Mbps

    df = df.assign(aggregated_ts=df_ts['data'].tolist())

    df.fillna(0, inplace=True)

    df_min_max_scaled = df.copy()
    column = 'aggregated_ts'
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
            df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    #print(df_min_max_scaled)
    df = df_min_max_scaled

    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30
    target = f"{target_sensor}_lead{forecast_lead}"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]
    #print(df)

    return df, target

def get_train_test(df):
    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30
    target = f"{target_sensor}_lead{forecast_lead}"

    train_ind = int(len(df) * 0.8)
    train = df[:train_ind]
    test = df[train_ind:]
    train_length = train.shape[0]
    test_length = test.shape[0]
    print('\nTraining size: ', train_length)
    print('Test size: ', test_length)
    print('Test ratio: ', test_length / (test_length + train_length))
    print("\n")
    return train, test, train_length, test_length, features, target

def load_data(client_id):
    file_name = "dataset.pkl"
    df = pd.read_pickle(file_name)

    '''Choose one dataset for each client'''
    if client_id == 1:
        df = df['ElBorn']
    elif client_id == 2:
        df = df['LesCorts']
    elif client_id == 3:
        df = df['PobleSec']
    else:
        print("Number of clients > dataset")


    df.set_index(df.iloc[:, 0].name)
    df.index.names = ['TimeStamp']



    data_columns = list(df.columns.values)
    data = df[data_columns].values
    data = np.clip(data, 0.0, np.percentile(data.flatten(), 99))  # we use 99% as the threshold
    df[data_columns] = data

    aggregated_time_series = np.sum(data, axis=1)
    df_ts = pd.DataFrame()
    df_ts['data'] = aggregated_time_series / 1000  # Plot in Mbps

    #df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True)
    df = df.assign(aggregated_ts=df_ts['data'].tolist())

    df.fillna(0, inplace=True)


    #Normalizing the aggregated column
    df_min_max_scaled = df.copy()
    column = 'aggregated_ts'
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    df = df_min_max_scaled
    #create_graph(df_min_max_scaled, "DF_Normalized")
    #exit()



    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30
    target = f"{target_sensor}_lead{forecast_lead}"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]

    # divide data into train and test
    train_ind = int(len(df) * 0.8)
    df_train = df[:train_ind].copy()
    df_test = df[train_ind:].copy()
    #print(df_train.head())
    #print(df_test.head())
    train_length = df_train.shape[0]
    test_length = df_test.shape[0]
    print('Training size: ', train_length)
    print('Test size: ', test_length)
    print('Test ratio: ', test_length / (test_length + train_length))

    #reating the dataset and the data loaders for real
    torch.manual_seed(101)

    batch_size = 32
    sequence_length = 30

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    print("Target: "+str(target))
    print("Features: "+str(features))
    print("sequence_length: "+str(sequence_length))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

#Training phase
def train_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def check_error(orig, pred, name_col='', index_name=''):
    bias = np.mean(orig - pred)
    mse = mean_squared_error(orig, pred)
    rmse = sqrt(mean_squared_error(orig, pred))
    mae = mean_absolute_error(orig, pred)
    mape = np.mean(np.abs((orig - pred) / orig)) * 100

    error_group = [bias, mse, rmse, mae, mape]
    result = pd.DataFrame(error_group, index=['BIAS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[name_col])
    result.index.name = index_name
    print("Result: " + str(result))
    return result

def plot_real_pred_alone(df_out, model_name, test_start, graph_image_name):
    #Creating graph

    df_completo = df_out.index
    test_start = df_completo[test_start]


    df_out = df_out.rename(columns={"Model forecast": "Predicted"})
    plot_template = dict(
            layout=go.Layout({
                "font_size": 18,
                "xaxis_title_font_size": 24,
                "yaxis_title_font_size": 24})
        )

    fig = px.line(df_out, labels=dict(created_at="Date", value="Resources Usage"))
    fig.add_vline(x=test_start, line_width=4, line_dash="dash")
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )
    fig.write_image("Resultados/"+str(model_name)+ str("_")+str(graph_image_name)+"_Real_versus_Pred_graph_federado-CLIENT.png")

def plot_real_pred_detailed(data, model_name, graph_image_name, figsize=(12, 9), lags=24, rotation=0):
    # Creating the column error
    data['Error'] = data.iloc[:, 0] - data.iloc[:, 1]

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Plotting actual and predicted values
    ax1.plot(data.iloc[:, 0:2])
    ax1.legend(['Real', 'Pred'])
    ax1.set_title('Real Value vs Prediction')
    ax1.xaxis.set_tick_params(rotation=rotation)

    # Error vs Predicted value
    ax2.scatter(data.iloc[:, 1], data.iloc[:, 2])
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual vs Predicted Values')

    # Residual QQ Plot
    sm.graphics.qqplot(data.iloc[:, 2], line='r', ax=ax3)

    # Autocorrelation Plot of residual
    plot_acf(data.iloc[:, 2], lags=lags, zero=False, ax=ax4)
    plt.tight_layout()
    plt.show()
    plt.savefig("Resultados/" + str(model_name) + str("_") + str(graph_image_name) + '_autoCorrelation_federado-CLIENT.png', bbox_inches='tight', pad_inches=0.1)

def test_model(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y)

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

class AsyncTrainer(SGDClientTrainer):
    @property
    def uplink_package(self):
        return [self.model_parameters, self.round]

    def local_process(self, payload, id):
        model_parameters = payload[0]
        self.round = payload[1]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

class AsyncClientTrainer(SGDClientTrainer):

    def __init__(self,
                 model,
                 data_loader,
                 epochs,
                 optimizer,
                 criterion,
                 cuda=True,
                 logger=None):
        super().__init__(model, data_loader, epochs, optimizer, criterion,
                         cuda, logger)
        self.time = 0

    def local_process(self, payload):
        self.time = payload[1].item()
        return super().local_process(payload)

    @property
    def uplink_package(self):
        return [self.model_parameters, torch.Tensor([self.time])]

def get_ready_test(client_id):
    df, target = get_data(client_id)

    train_ind = int(len(df) * 0.8)
    train = df[:train_ind]
    test = df[train_ind:]
    # print(train.head())
    # print(test.head())
    train_length = train.shape[0]
    test_length = test.shape[0]

    input_features = [target]
    data = df[input_features].values

    length = data.shape[0]
    print(length)

    x_data = []
    y_data = []
    for i in range(length - history - horizon + 1):
        x = data[i:i + history, :]  # input historical time steps
        y = data[i + history:i + history + horizon:, 0]  # output predicted time steps, we only predict value_avg
        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_data = np.swapaxes(x_data, 1, 2)

    test_length = test_length - horizon + 1

    train_valid_length = x_data.shape[0] - test_length

    train_length = int(train_valid_length * 0.8)
    valid_length = train_valid_length - train_length

    X_train = x_data[:train_length]
    y_train = y_data[:train_length]
    X_valid = x_data[train_length:train_valid_length]
    y_valid = y_data[train_length:train_valid_length]
    X_test = x_data[train_valid_length:]
    y_test = y_data[train_valid_length:]

    print("Train Size X: " + str(X_train.shape))
    print("Train Size Y: " + str(y_train.shape))
    print("Valid Size X: " + str(X_valid.shape))
    print("Valid Size Y: " + str(y_valid.shape))
    print("Test Size X: " + str(X_test.shape))
    print("Test Size Y: " + str(y_test.shape))
    return X_test, y_test

def get_ready(client_id):
    history = 24  # input historical time steps
    horizon = 1  # output predicted time steps
    test_ratio = 0.2  # testing data ratio
    max_evals = 1  # maximal trials for hyper parameter tuning

    df, target = get_data(client_id)

    train_ind = int(len(df) * 0.8)
    train = df[:train_ind]
    test = df[train_ind:]
    #print(train.head())
    #print(test.head())
    train_length = train.shape[0]
    test_length = test.shape[0]

    input_features = [target]
    data = df[input_features].values

    length = data.shape[0]
    print(length)

    x_data = []
    y_data = []
    for i in range(length - history - horizon + 1):
        x = data[i:i + history, :]  # input historical time steps
        y = data[i + history:i + history + horizon:, 0]  # output predicted time steps, we only predict value_avg
        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_data = np.swapaxes(x_data, 1, 2)

    test_length = test_length - horizon + 1

    train_valid_length = x_data.shape[0] - test_length

    train_length = int(train_valid_length * 0.8)
    valid_length = train_valid_length - train_length

    X_train = x_data[:train_length]
    y_train = y_data[:train_length]
    X_valid = x_data[train_length:train_valid_length]
    y_valid = y_data[train_length:train_valid_length]
    X_test = x_data[train_valid_length:]
    y_test = y_data[train_valid_length:]

    print("Train Size X: " + str(X_train.shape))
    print("Train Size Y: " + str(y_train.shape))
    print("Valid Size X: " + str(X_valid.shape))
    print("Valid Size Y: " + str(y_valid.shape))
    print("Test Size X: " + str(X_test.shape))
    print("Test Size Y: " + str(y_test.shape))

    X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
    tfms = [None, [TSRegression()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

    batch_size = 32
    tfms = [None, [TSRegression()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    # set num_workers for memory bottleneck
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

    trainloader, testloader = load_data(args.client_id)

    if model_name == "ResCNN":
        arch = ResCNN
    k = {
        'layers': 25,
        'ks': 5,
        'conv_dropout': 0.5
    }

    model = create_model(arch, d=False, dls=dls)
    # Add a Sigmoid layer
    model = nn.Sequential(model, nn.Sigmoid())

    return dsets, model, dls



parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--client_id', type=int, default=1)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=100)
args = parser.parse_args()


if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

#model = ShallowRegressionLSTM(num_sensors=11, hidden_units=16)
#trainloader, testloader = load_data(args.client_id)
#print("type: "+str(trainloader))
#print("type: "+str(testloader))

dsets, model, dls = get_ready(args.client_id)
trainloader, testloader = load_data(args.client_id)
print(type(dls.train))


optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
handler = AsyncClientTrainer(model,
                             dls.train,
                             epochs=args.epochs,
                             optimizer=optimizer,
                             criterion=criterion,
                             cuda=args.cuda)


network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=args.rank)

Manager = ActiveClientManager(trainer=handler, network=network)
Manager.run()

learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
valid_dl = dls.valid
X_test, y_test = get_ready_test(args.client_id)

test_ds = valid_dl.dataset.add_test(X_test, y_test)  # use the test data
test_dl = valid_dl.new(test_ds)

start = time.time()
test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
prediction_time = time.time() - start
test_probas, test_targets, test_preds

y_true = test_targets.numpy()
y_pred = test_preds.numpy()

y_true = y_true.reshape(y_true.shape[0], horizon, -1)
y_pred = y_pred.reshape(y_pred.shape[0], horizon, -1)

print("Y_true: "+str(y_true.shape))
print("Y_pred: "+str(y_pred.shape))

"""
The training and test time spent:
"""

# %%
print('Training time (in seconds): ', training_time)
print('Test time (in seconds): ', prediction_time)


# %%
def check_error(orig, pred, name_col='', index_name=''):
    bias = np.mean(orig - pred)
    mse = mean_squared_error(orig, pred)
    rmse = sqrt(mean_squared_error(orig, pred))
    mae = mean_absolute_error(orig, pred)
    mape = np.mean(np.abs((orig - pred) / orig)) * 100

    error_group = [bias, mse, rmse, mae, mape]
    result = pd.DataFrame(error_group, index=['BIAS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[name_col])
    result.index.name = index_name
    print("Result: " + str(result))

    filename = "Resultados/" + str(model_name) + "/avaliation.txt"
    text_file = open(filename, "w")
    n = text_file.write(str(result))
    text_file.close()

    return result


# %%
step_to_evalute = 0
true_values = y_true[:, step_to_evalute]
pred_values = y_pred[:, step_to_evalute]

# %%
result = pd.DataFrame()

# %%
check_error(true_values, pred_values, name_col=model_name)


# %%
def plot_error(data, figsize=(12, 9), lags=24, rotation=0):
    # Creating the column error
    data['Error'] = data.iloc[:, 0] - data.iloc[:, 1]

    plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    # Plotting actual and predicted values
    ax1.plot(data.iloc[:, 0:2])
    ax1.legend(['Real', 'Pred'])
    ax1.set_title('Real Value vs Prediction')
    ax1.xaxis.set_tick_params(rotation=rotation)

    # Error vs Predicted value
    ax2.scatter(data.iloc[:, 1], data.iloc[:, 2])
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual vs Predicted Values')

    # Residual QQ Plot
    sm.graphics.qqplot(data.iloc[:, 2], line='r', ax=ax3)

    # Autocorrelation Plot of residual
    plot_acf(data.iloc[:, 2], lags=lags, zero=False, ax=ax4)
    plt.tight_layout()
    plt.show()
    plt.savefig("Resultados/"+str(model_name)+"/" + str(model_name) + '_autoCorrelation.png', bbox_inches='tight', pad_inches=0.1)


model_test = test[[target]].copy()
model_test.index = test.index
model_test.columns = ['Real']

model_test['Pred'] = pred_values

plot_error(model_test, rotation=45)