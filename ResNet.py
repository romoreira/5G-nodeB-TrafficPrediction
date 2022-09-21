import warnings
warnings.filterwarnings('ignore')

import tsai
from tsai.all import *
print('tsai       :', tsai.__version__)
print('fastai     :', fastai.__version__)
print('fastcore   :', fastcore.__version__)
print('torch      :', torch.__version__)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


import hyperopt
print('hyperopt   :',hyperopt.__version__)
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp
from hyperopt import space_eval

import time
from fastai.callback.tracker import EarlyStoppingCallback
import gc

import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

file_name = "dataset.pkl"
history = 24  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 1  # maximal trials for hyper parameter tuning

model_name = 'ResNet'
# Save the results
y_true_fn = '%s_true-%d-%d.pkl' % (model_name, history, horizon)
y_pred_fn = '%s_pred-%d-%d.pkl' % (model_name, history, horizon)

df = pd.read_pickle(file_name)
df = df['LesCorts']
print(df.keys())

df.set_index(df.iloc[:,0].name)
df.index.names = ['TimeStamp']

#divide data into train and test
train_ind = int(len(df)*0.8)
train = df[:train_ind]
test = df[train_ind:]
train_length = train.shape[0]
test_length = test.shape[0]

print('Training size: ', train_length)
print('Test size: ', test_length)
print('Test ratio: ', test_length / (test_length + train_length))

input_features = ['down', 'up', 'rnti_count', 'mcs_down', 'mcs_down_var', 'mcs_up',
       'mcs_up_var', 'rb_down', 'rb_down_var', 'rb_up', 'rb_up_var']

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

print("X_DATA.shape: "+str(x_data.shape))
print("Y_DATA.shape: "+str(y_data.shape))

x_data = np.swapaxes(x_data, 1, 2)
print("X_DATA.shape: "+str(x_data.shape))

test_length = test_length - horizon + 1
train_valid_length = x_data.shape[0] - test_length
print("train_valid_length: "+str(train_valid_length))

train_length = int(train_valid_length * 0.8)
valid_length = train_valid_length - train_length

X_train = x_data[:train_length]
y_train = y_data[:train_length]
X_valid = x_data[train_length:train_valid_length]
y_valid = y_data[train_length:train_valid_length]
X_test = x_data[train_valid_length:]
y_test = y_data[train_valid_length:]

print("X_train.shape: "+str(X_train.shape))
print("y_train.shape: "+str(y_train.shape))
print("X_valid.shape: "+str(X_valid.shape))
print("y_valid.shape: "+str(y_valid.shape))
print("X_test.shape: "+str(X_test.shape))
print("y_test.shape: "+str(y_test.shape))


##Building TSAI Datasets

X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])

tfms  = [None, [TSRegression()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

search_space = {
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    "lr": hp.choice('lr', [0.01, 0.001, 0.0001]),
    "epochs": hp.choice('epochs', [1]),  # we would also use early stopping
    "patience": hp.choice('patience', [5, 10]),  # early stopping patience
    # "optimizer": hp.choice('optimizer', [Adam, SGD, RMSProp]),  # https://docs.fast.ai/optimizer
    "optimizer": hp.choice('optimizer', [Adam]),
    # model parameters
}

#Clear memory
gc.collect()


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
        arch = ResNet
        model = create_model(ResNet, d=False, dls=dls)
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
    except Exception as e:
        print(e)
        return {'loss': None, 'status': STATUS_FAIL}

trials = Trials()
best = fmin(create_model_hypopt,
    space=search_space,
    algo=tpe.suggest,
    max_evals=max_evals,  # test trials
    trials=trials)


dls = learn.dls
valid_dl = dls.valid

test_ds = valid_dl.dataset.add_test(X_test, y_test)  # use the test data
test_dl = valid_dl.new(test_ds)
print(test_dl.n)

start = time.time()
test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
prediction_time = time.time() - start
test_probas, test_targets, test_preds

y_true = test_targets.numpy()
y_pred = test_preds.numpy()

print('Training time (in seconds): ', training_time)
print('Test time (in seconds): ', prediction_time)


def check_error(orig, pred, name_col='', index_name=''):
    bias = np.mean(orig - pred)
    mse = mean_squared_error(orig, pred)
    rmse = sqrt(mean_squared_error(orig, pred))
    mae = mean_absolute_error(orig, pred)
    mape = np.mean(np.abs((orig - pred) / orig)) * 100

    error_group = [bias, mse, rmse, mae, mape]
    result = pd.DataFrame(error_group, index=['BIAS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[name_col])
    result.index.name = index_name

    return result

step_to_evalute = 0
true_values = y_true[:, step_to_evalute]
pred_values = y_pred[:, step_to_evalute]

result = pd.DataFrame()

print(check_error(true_values, pred_values, name_col=model_name))