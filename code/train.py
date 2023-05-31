import argparse
import pandas as pd
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
import pickle
from utils import processData
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    # todo change defaults
    parser.add_argument(
        'dataDir', type=str, default='../data-sample/', help='path to train data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    if True:
        with open('my_data.pkl', 'wb') as outp:
            all_data_train, features, time_column, event_column = processData(dataDir)
            pickle.dump(all_data_train, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(features, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(time_column, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(event_column, outp, pickle.HIGHEST_PROTOCOL)
    else :
        with open('my_data.pkl', 'rb') as inp:
            all_data_train = pickle.load(inp)
            features = pickle.load(inp)
            time_column = pickle.load(inp)
            event_column = pickle.load(inp)

    # split current data into train and test set since we dont have the ultimate test y values
    X_train = all_data_train[features]
    Y_train = all_data_train[[time_column, event_column]]

    # X_train, X_test, y_train, y_test = train_test_split(all_data_train[features], all_data_train[[time_column, event_column]], test_size=0.3, random_state=101)

    # save test data for further use
    # with open('test_data.pkl','wb') as test:
    #     pickle.dump(X_test, test, pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(y_test, test, pickle.HIGHEST_PROTOCOL)

    print("Starting training")
    # Adapt class LinearMultiTaskModel to make it compatible with scikit-learn
    # LinearMultiTaskModelSkl = sklearn_adapter(LinearMultiTaskModel, time_col=time_column, event_col=event_column,
    #                                           predict_method="predict_survival", scoring_method=concordance_index)
    LinearMultiTaskModelSkl = sklearn_adapter(NeuralMultiTaskModel, time_col=time_column, event_col=event_column,
                                              predict_method="predict_survival", scoring_method=concordance_index)

    # note - bins 400 is barely enough for daily resolution for a bin if max runtime is ~10k hours. should consider using larger bins
    l_mtlr = LinearMultiTaskModelSkl(structure = [ {'activation': 'relu', 'num_units': 128}, 
                          {'activation': 'tanh', 'num_units': 128},{'activation': 'tanh', 'num_units': 128} ],bins=400, auto_scaler=True,num_epochs=10000)
    l_mtlr.fit(X_train, Y_train, lr=1e-5, init_method='orthogonal')
    print("Training complete")

    print("Saving model")

    with open('model.pickle', 'wb') as f:
        pickle.dump(l_mtlr, f)
    print("Done")

    with open('features.pickle', 'wb') as f:
        pickle.dump(features, f)
    print("Done")


if __name__ == '__main__':
    main()
