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
print('tsai       :', tsai.__version__)
print('fastai     :', fastai.__version__)
print('fastcore   :', fastcore.__version__)
print('torch      :', torch.__version__)



large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
           'legend.fontsize': med,
           'figure.figsize': (10, 6),
           'axes.labelsize': med,
           'axes.titlesize': med,
           'xtick.labelsize': med,
           'ytick.labelsize': med,
           'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

print(mpl.__version__)
print(sns.__version__)

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

def get_data():
    file_name = "dataset.pkl"
    df = pd.read_pickle(file_name)
    df = df['LesCorts']
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


history = 24  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 50  # maximal trials for hyper parameter tuning

df, target = get_data()

train_ind = int(len(df)*0.8)
train = df[:train_ind]
test = df[train_ind:]
print(train.head())
print(test.head())
train_length = train.shape[0]
test_length = test.shape[0]


input_features = [target]
data = df[input_features].values

length = data.shape[0]
print(length)

x_data = []
y_data = []
for i in range(length - history - horizon + 1):
    x = data[i:i+history, :]  # input historical time steps
    y = data[i+history:i+history+horizon:, 0]  # output predicted time steps, we only predict value_avg
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

print("Train Size X: "+str(X_train.shape))
print("Train Size Y: "+str(y_train.shape))
print("Valid Size X: "+str(X_valid.shape))
print("Valid Size Y: "+str(y_valid.shape))
print("Test Size X: "+str(X_test.shape))
print("Test Size Y: "+str(y_test.shape))

X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
tfms  = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)


search_space = {
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    "lr": hp.choice('lr', [0.01, 0.001, 0.0001]),
    "epochs": hp.choice('epochs', [20, 50, 100]),  # we would also use early stopping
    "patience": hp.choice('patience', [5, 10]),  # early stopping patience
    # "optimizer": hp.choice('optimizer', [Adam, SGD, RMSProp]),  # https://docs.fast.ai/optimizer
    "optimizer": hp.choice('optimizer', [Adam]),
    # model parameters
    "layers": hp.choice('layers', [[25, 25, 25, 25, 25, 25, 25, 25], [25, 25, 25, 25, 25, 25], [25, 25, 25, 25]]),
    "ks": hp.choice('ks', [7, 5, 3]),
    "conv_dropout": hp.choice('conv_dropout', [0.0, 0.1, 0.2, 0.5])
}


def create_model_hypopt(params):
    try:
        # clear memory
        gc.collect()
        print("Trying params:", params)
        batch_size = params["batch_size"]

        # Create data loader
        tfms = [None, [TSRegression()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        # set num_workers for memory bottleneck
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

        # Create model
        arch = TCN
        k = {
            'layers': params['layers'],
            'ks': params['ks'],
            'conv_dropout': params['conv_dropout']
        }

        model = create_model(arch, d=False, dls=dls)
        print(model.__class__.__name__)


        # Add a Sigmoid layer
        model = nn.Sequential(model, nn.Sigmoid())


        # Training the model
        learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
        start = time.time()
        learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                            cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=params['patience']))
        learn.plot_metrics()
        elapsed = time.time() - start
        print(elapsed)

        vals = learn.recorder.values[-1]
        print(vals)
        # vals[0], vals[1], vals[2]
        # train loss, valid loss, accuracy
        val_loss = vals[1]

        # delete tmp variables
        del dls
        del model
        del learn
        return {'loss': val_loss, 'status': STATUS_OK}  # if accuracy use '-' sign, model is optional
    except:
        return {'loss': None, 'status': STATUS_FAIL}


# %%
trials = Trials()
best = fmin(create_model_hypopt,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,  # test trials
            trials=trials)

print("Best parameters:")
print(space_eval(search_space, best))
params = space_eval(search_space, best)

X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])

batch_size = params["batch_size"]
tfms = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
# set num_workers for memory bottleneck
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)


arch = TCN
k = {
    'layers': params['layers'],
    'ks': params['ks'],
    'conv_dropout': params['conv_dropout']
}
model = create_model(arch, d=False, dls=dls)
print(model.__class__.__name__)

# Add a Sigmoid layer
model = nn.Sequential(model, nn.Sigmoid())

learn = Learner(dls, model, metrics=[mae, rmse], opt_func=params['optimizer'])
start = time.time()
learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                    cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=params['patience']))
training_time = time.time() - start
learn.plot_metrics()


dls = learn.dls
valid_dl = dls.valid

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
    return result


# %%
step_to_evalute = 0
true_values = y_true[:, step_to_evalute]
pred_values = y_pred[:, step_to_evalute]

# %%
result = pd.DataFrame()

# %%
check_error(true_values, pred_values, name_col="TCN")


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
    plt.savefig("Resultados/TCN/" + str("TCN") + '_autoCorrelation.png', bbox_inches='tight', pad_inches=0.1)


model_test = test[[target]].copy()
model_test.index = test.index
model_test.columns = ['Real']

model_test['Pred'] = pred_values

plot_error(model_test, rotation=45)