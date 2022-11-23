from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import flwr as fl
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.preprocessing import MinMaxScaler

from models.OmniScaleCNN import OmniScaleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "OmniScaleCNN"



name = './Resultados'
if os.path.isdir(name) == False:
    os.mkdir(name)

resultados_dir = './Resultados/' + model_name
if os.path.isdir(resultados_dir) == False:
    os.mkdir(resultados_dir)

def build_train_test_graph(df, client_id):
    train_ind = int(len(df) * 0.8)
    train = df[:train_ind]
    test = df[train_ind:]
    train_length = train.shape[0]

    plt.figure(figsize=[12, 6])
    plt.plot(df.index[:train_length], df['aggregated_ts_lead30'][:train_length], label='Training', color='navy')
    plt.plot(df.index[train_length:], df['aggregated_ts_lead30'][train_length:], label='Test', color='orange')
    plt.axvspan(df.index[train_length:][0], df.index[train_length:][-1], facecolor='r', alpha=0.1)

    plt.xlabel('Time')
    plt.ylabel('RAN Consumption')
    plt.legend(loc='upper center')
    #plt.show()
    plt.savefig(resultados_dir + '/' + str(model_name) + '_training_test_split'+str("_")+str(client_id)+'.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def get_train_test(client_id, df):

    target = "aggregated_ts_lead30"

    # divide data into train and test
    train_ind = int(len(df) * 0.8)
    df_train = df[:train_ind].copy()
    df_test = df[train_ind:].copy()
    # print(df_train.head())
    # print(df_test.head())
    train_length = df_train.shape[0]
    test_length = df_test.shape[0]
    print('Training size: ', train_length)
    print('Test size: ', test_length)
    print('Test ratio: ', test_length / (test_length + train_length))
    return df_train, df_test, train_length, test_length, target

def min_max_scaler(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['aggregated_ts_lead30'] = scaler.fit_transform(df['aggregated_ts_lead30'].values.reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 4))
    plt.plot(df['aggregated_ts_lead30'])
    plt.show()
    return df


def create_loss_graph(train_losses, test_losses, plt_title):
    # Cria os graficos de decaimento treino e validação (imprime na tela e salva na pasta "./Resultados")
    plt.title("Training Loss")
    plt.plot(train_losses, label='Train')
    #plt.plot(test_losses, label='Test')
    plt.legend(frameon=False)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(b=None)
    plt.legend()
    plt.grid()
    plt.savefig(resultados_dir + '/' + 'graf_' + str(plt_title) + '.pdf')
    plt.close()

def create_plot_real_pred(real, pred, client_id):
    plt.figure()
    plt.plot(pred, label='predicted')
    plt.plot(real, label='actual')
    plt.ylabel('output y')
    plt.legend()
    plt.savefig(resultados_dir + '/' + 'graf_' + str("RealPredict_") + str(client_id) + '.png')


class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=30):
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

def load_data(client_id):
    """
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples
    """

    file_name = "../fedlab/dataset.pkl"
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

    # df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True)
    df = df.assign(aggregated_ts=df_ts['data'].tolist())

    df.fillna(0, inplace=True)


    # # Normalizing the aggregated column
    # df_min_max_scaled = df.copy()
    # column = 'aggregated_ts'
    # df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
    #         df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    # print(df_min_max_scaled)
    # df = df_min_max_scaled
    # # print(df)
    # # create_graph(df_min_max_scaled, "DF_Normalized")
    # # exit()

    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30 #30 x 2 minutos -> 1 hour ahead
    target = f"{target_sensor}_lead{forecast_lead}"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]
    print(df)

    df = min_max_scaler(df)  # Normaliza entre 0 e 1 o dataframe
    build_train_test_graph(df, client_id)


    train_ind = int(len(df) * 0.8)
    df_train = df[:train_ind]
    df_test = df[train_ind:]

    # reating the dataset and the data loaders for real
    #torch.manual_seed(101)

    batch_size = 8
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

    print("Target: " + str(target))
    print("Features: " + str(len(features)))
    print("sequence_length: " + str(sequence_length))
    print("Batch Size: "+str(batch_size))

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}

    X, y = next(iter(trainloader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Num. Examples: ", num_examples)

    return trainloader, testloader, num_examples, df


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_list = []
    epoch_loss = 0.0
    for epoch in range(epochs):
        for images, labels in tqdm(trainloader):
            outputs = net(images.float().to(DEVICE))
            #print("OUTPUTS TRAINING: "+str(outputs.data))
            optimizer.zero_grad()
            loss = criterion(net(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        epoch_loss /= len(trainloader.dataset)
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}")
        loss_list.append(epoch_loss.item())
    print(loss_list)
    return loss_list


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    pred = []
    real = []
    i = 0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.float().to(DEVICE))
            labels = labels.to(DEVICE)
            if len(outputs) == 5:
                break
            loss += criterion(outputs, labels).item()
            total += labels.to(DEVICE).size(0)
            #print("OUTPUTS: "+str(outputs.data))
            #predicted = torch.max(outputs.data)
            predicted = outputs.data
            #print("REAL: "+str(labels))
            #print("PREDICTED: "+str(predicted))
            #print("PREDICTED: " + str(type(predicted)))
            pred.extend(predicted.cpu().detach().numpy())
            real.extend(labels.cpu().detach().numpy())
            correct += 1
    #print("LOSS: "+str(loss/len(testloader.dataset)))
    #print("CORRECT: "+str(correct/total))
    return loss/len(testloader.dataset), correct/total, real, pred



class ShallowRegressionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_sensors = 11  # this is the number of features
        self.hidden_units = 16
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1).to(DEVICE)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=DEVICE).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units, device=DEVICE).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

class FlowerClient(fl.client.NumPyClient):

    losses_train = []
    losses_test = []

    real = []
    predicted = []

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.losses_train = train(net, trainloader, epochs=args.epoch)
        print("SELF TRAIN: "+str(self.losses_train))
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, self.real, self.predicted = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--client_id', type=int, default=1)
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()


def get_model(model_name):
    if model_name == "OmniScaleCNN":
        c_in = 30
        seq_len = 11
        c_out = 1
        model = OmniScaleCNN(c_in, c_out, seq_len)
        return model
    elif model_name == "LSTM":
        model = ShallowRegressionLSTM()
        return model

# Load model and data
net = get_model(model_name)
net.cuda()
trainloader, testloader, num_examples, df = load_data(args.client_id)
client = FlowerClient()


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)
create_loss_graph(client.losses_train, [], str("Loss Train" + str(args.client_id)))
create_plot_real_pred(client.real, client.predicted, client_id=args.client_id)

