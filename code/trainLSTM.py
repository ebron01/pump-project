import torch
import argparse
import pandas as pd
from pysurvival.models.multi_task import LinearMultiTaskModel
# from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
import pickle
from utils import processData
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

    all_data_train, features, time_column, event_column = processData(dataDir)

    X_train = all_data_train[features]
    # Y_train = all_data_train[[time_column, event_column]]
    # let's start with one output
    Y_train = all_data_train[event_column]

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

    # # Adapt class LinearMultiTaskModel to make it compatible with scikit-learn
    # LinearMultiTaskModelSkl = sklearn_adapter(LinearMultiTaskModel, time_col=time_column, event_col=event_column,
    #                                           predict_method="predict_survival", scoring_method=concordance_index)

    # # note - bins 400 is barely enough for daily resolution for a bin if max runtime is ~10k hours. should consider using larger bins
    # l_mtlr = LinearMultiTaskModelSkl(bins=400, auto_scaler=True)
    # l_mtlr.fit(X_train, Y_train, lr=1e-5, init_method='orthogonal')

    # print("Saving model")

    # with open('model.pickle', 'wb') as f:
    #     pickle.dump(l_mtlr, f)
    # print("Done")

    # with open('features.pickle', 'wb') as f:
    #     pickle.dump(features, f)
    # print("Done")


if __name__ == '__main__':
    main()
