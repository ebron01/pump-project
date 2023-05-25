import argparse
import pandas as pd
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
import pickle
from utils import processData

def main():
    parser = argparse.ArgumentParser()
    # todo change defaults
    parser.add_argument('dataDir', type=str, default='../data-sample/', help='path to train data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    all_data_train, features, time_column, event_column = processData(dataDir)

    X_train = all_data_train[features]
    Y_train = all_data_train[[time_column, event_column]]

    print("Starting training")
    # Adapt class LinearMultiTaskModel to make it compatible with scikit-learn
    LinearMultiTaskModelSkl = sklearn_adapter(LinearMultiTaskModel, time_col=time_column, event_col=event_column,
                                            predict_method="predict_survival", scoring_method=concordance_index)
    
    # note - bins 400 is barely enough for daily resolution for a bin if max runtime is ~10k hours. should consider using larger bins
    l_mtlr = LinearMultiTaskModelSkl(bins=400, auto_scaler=True)
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