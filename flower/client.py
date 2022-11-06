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

warnings.filterwarnings("ignore", category=UserWarning)
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model_name = "LSTM"

name = './Resultados'
if os.path.isdir(name) == False:
    os.mkdir(name)

resultados_dir = './Resultados/' + model_name
if os.path.isdir(resultados_dir) == False:
    os.mkdir(resultados_dir)

def create_loss_graph(train_losses, plt_title):
    # Cria os graficos de decaimento treino e validação (imprime na tela e salva na pasta "./Resultados")
    plt.title(plt_title)
    plt.plot(train_losses, label='Train')
    plt.legend(frameon=False)
    plt.grid()
    plt.savefig(resultados_dir + '/' + 'graf_' + str(plt_title) + '.png')
    plt.close()



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

def load_data():
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

    # df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1, inplace=True)
    df = df.assign(aggregated_ts=df_ts['data'].tolist())

    df.fillna(0, inplace=True)

    print(df)


    # Normalizing the aggregated column
    df_min_max_scaled = df.copy()
    column = 'aggregated_ts'
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
            df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    print(df_min_max_scaled)
    df = df_min_max_scaled
    # print(df)
    # create_graph(df_min_max_scaled, "DF_Normalized")
    # exit()

    target_sensor = "aggregated_ts"
    features = list(df.columns.difference([target_sensor]))
    forecast_lead = 30 #30 x 2 minutos -> 1 hour ahead
    target = f"{target_sensor}_lead{forecast_lead}"
    df[target] = df[target_sensor].shift(-forecast_lead)
    df = df.iloc[:-forecast_lead]
    print(df)

    train_ind = int(len(df) * 0.8)
    df_train = df[:train_ind]
    df_test = df[train_ind:]

    # reating the dataset and the data loaders for real
    #torch.manual_seed(101)

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

    return trainloader, testloader, num_examples


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    correct, total, epoch_loss = 0, 0, 0.0
    loss_list = []
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
        epoch_loss /= len(trainloader.dataset)
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}")
        loss_list.append(epoch_loss.item())
    return loss_list


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Loss (AVG): {loss}")
    return loss, accuracy


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
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

class FlowerClient(fl.client.NumPyClient):

    losses_train = []
    losses_test = []

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.losses_train = train(net, trainloader, epochs=10)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
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
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Load model and data
net = ShallowRegressionLSTM().to(DEVICE)
trainloader, testloader, num_examples = load_data()
client = FlowerClient()


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)
print("Fim do Cliente: "+str(args.client_id))
create_loss_graph(client.losses_train, str("Loss Graph Client_"+str(args.client_id)))