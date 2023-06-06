import torch
import argparse
import pandas as pd
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
import pickle
from utils import processData, processLSTM, deleteMissing
from sklearn.model_selection import train_test_split
from models.SequenceDataset import SequenceDataset
from torch.utils.data import DataLoader
from torch import nn
from models.LSTM import ShallowRegressionLSTM
torch.manual_seed(99)

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


def main():
    parser = argparse.ArgumentParser()
    # todo change defaults
    parser.add_argument(
        'dataDir', type=str, default='../data-sample/', help='path to train data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    if False:
        with open('my_data.pkl', 'wb') as outp:
            all_data_train, features, time_column, event_column, daily_run_data = processData(dataDir)
            # daily_run_data, event_data, equipment = processLSTM(dataDir, daily_run_data)
            # train_data, event_data = deleteMissing(daily_run_data, event_data, equipment)
            pickle.dump(all_data_train, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(features, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(time_column, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(event_column, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(daily_run_data, outp, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(train_data, outp, pickle.HIGHEST_PROTOCOL)
            # pickle.dump(event_data, outp, pickle.HIGHEST_PROTOCOL)
    else :
        with open('my_data.pkl', 'rb') as inp:
            all_data_train = pickle.load(inp)
            features = pickle.load(inp)
            time_column = pickle.load(inp)
            event_column = pickle.load(inp)
            daily_run_data = pickle.load(inp)
            # train_data = pickle.load(inp)
            # event_data = pickle.load(inp)
    
    daily_run_data, event_data, equipment = processLSTM(dataDir, daily_run_data)
    train_data, event_data = deleteMissing(daily_run_data, event_data, equipment)
    
    # split current data into train and test set since we dont have the ultimate test y values
    # X_train = all_data_train[features]
    X_train = train_data
    # Y_train = all_data_train[[time_column, event_column]]
    # let's start with one output
    # Y_train = all_data_train[event_column]
    Y_train = event_data

    sequence_length = 5
    train_dataset = SequenceDataset(
        target=Y_train,
        features=X_train,
        sequence_length=sequence_length
    )
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=False)
    # X, y = next(iter(train_loader))

    print("Starting training")

    learning_rate = 5e-5
    num_hidden_units = 16

    model = ShallowRegressionLSTM(num_features=len(
        features), hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ix_epoch in range(50):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
    print("Saving model")

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
    print("Done")

    with open('features.pickle', 'wb') as f:
        pickle.dump(features, f)
    print("Done")


if __name__ == '__main__':
    main()
