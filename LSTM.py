#Code based on: https://www.crosstab.io/articles/time-series-pytorch-lstm
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
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

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

def create_loss_graph(y1, y2, model_name):
    x = range(1, len(y2) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    lns1 = ax1.plot(x, y1, 'b--', linewidth=2, label="Validation Loss")
    lns2 = ax2.plot(x, y2, 'red', linewidth=2, label="Training Loss")

    x_range = range(0, len(y2) + 1)
    plt.xticks(np.arange(0, len(x) + 1, 5))
    plt.title(model_name, fontsize=18)

    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Validation Loss', color='b', fontsize=14)
    ax2.set_ylabel('Training Loss', color='red', fontsize=14)

    # added these three lines for insert labels
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0, frameon=False, fontsize=14)

    # plt.grid()
    # plt.figure()
    plt.savefig('Resultados/' + 'loss-graf_' + model_name + '.pdf')
    # plt.show()


def create_graph(df, name):
    plot_template = dict(
        layout=go.Layout({
            "font_size": 18,
            "xaxis_title_font_size": 24,
            "yaxis_title_font_size": 24})
    )

    fig = px.line(df, labels=dict(
        created_at="Date", value="Resources Usage", variable="Sensor"
    ))
    fig.update_layout(
        template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
    )

    fig.write_image("Resultados/"+str(name)+"_file.pdf")

pio.templates.default = "plotly_white"

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
df_ts['data'] = aggregated_time_series / 1000  # Plot in Mbps

#df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True)
df = df.assign(aggregated_ts=df_ts['data'].tolist())

df.fillna(0, inplace=True)

print(df)
create_graph(df, "DF_Original")



#Normalizing the aggregated column
df_min_max_scaled = df.copy()
column = 'aggregated_ts'
df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
            df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
print(df_min_max_scaled)
df = df_min_max_scaled
#print(df)
#create_graph(df_min_max_scaled, "DF_Normalized")
#exit()



target_sensor = "aggregated_ts"
features = list(df.columns.difference([target_sensor]))
forecast_lead = 2
target = f"{target_sensor}_lead{forecast_lead}"
df[target] = df[target_sensor].shift(-forecast_lead)
df = df.iloc[:-forecast_lead]
print(df)


test_start = "2019-01-20"
df_train = df.loc[:test_start].copy()
df_test = df.loc[test_start:].copy()
print("Test set fraction:", len(df_test) / len(df))


target_mean = df_train[target].mean()
target_stdev = df_train[target].std()
for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()
    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev

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


i = 27
sequence_length = 4
train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)

print("Target: "+str(target))
print("Features: "+str(features))
print("sequence_length: "+str(sequence_length))

X, y = train_dataset[i]
print(X)


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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)



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


#Quality function
learning_rate = 5e-5
num_hidden_units = 16

model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Plotting function
def plot_real_versus_predict(data, model_name, figsize=(12, 9), lags=24, rotation=0):
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
    plt.savefig("Resultados/" + str(model_name) + '_autoCorrelation.pdf', bbox_inches='tight', pad_inches=0.1)

#Training phase
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

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

def test_model(data_loader, model, loss_function):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss


print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

train_loss = []
test_loss = []
for ix_epoch in range(50):
    print(f"Epoch {ix_epoch}\n---------")
    train_loss.append(train_model(train_loader, model, loss_function, optimizer=optimizer))
    test_loss.append(test_model(test_loader, model, loss_function))
    print()


def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_eval_loader, model).numpy()
df_test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean

print(df_out)
plot_real_versus_predict(df_out, "LSTM")




#Creating graph
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
fig.write_image("Resultados/Real_versus_Predito_graph.pdf")


check_error(df_out[['aggregated_ts_lead2']].to_numpy(), df_out[['Model forecast']].to_numpy(), name_col="LSTM")
create_loss_graph(train_loss, test_loss, "LSTM")

