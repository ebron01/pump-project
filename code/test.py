import argparse
import pandas as pd
from datetime import datetime
from utils import processData
from pysurvival.utils.sklearn_adapter import sklearn_adapter
from pysurvival.utils.metrics import concordance_index
from pysurvival.models.multi_task import LinearMultiTaskModel
import pickle


def main():
    parser = argparse.ArgumentParser()
    # todo change defaults
    parser.add_argument('dataDir', type=str, default='../../test/', help='path to test data dir')
    parser.add_argument('outPath', type=str, default='../../scorer1/solution.csv', help='path to output file')
    args = parser.parse_args()

    dataDirTest = args.dataDir
    outPath = args.outPath

    # process test data
    all_data_test, features_test, time_column_test, event_column_test = processData(dataDirTest)

    # read trained model
    l_mtlr = None
    LinearMultiTaskModelSkl = sklearn_adapter(LinearMultiTaskModel, time_col=time_column_test, event_col=event_column_test,
                                            predict_method="predict_survival", scoring_method=concordance_index)

    with open('model.pickle', 'rb') as f:
        l_mtlr = pickle.load(f)

    # load trained features list
    features = []
    with open('features.pickle', 'rb') as f:
        features = pickle.load(f)

    # handle new values for categorical cata
    for c in features:
        if not c in features_test:
            all_data_test[c] = 0

    times =   [3, 7, 14, 30, 45, 60, 120]
    print('Predicting...')

    X_test = all_data_test[features]
    all_predictions=[]
    for index, row in X_test.iterrows():
        id = all_data_test.iloc[index]['failure_number']
        t0 = all_data_test.at[index, 'run_days']
        predictions = []
        predictions.append(id)
        for t in times:
            p = 1 - l_mtlr.predict(row, **{'t': t0 + t})
            ## sometimes pysurvival generates probabilities <0
            if p<0:
                p=0

            ## make sure values are monotonic (should never happen, just to be safe)
            if len(predictions)>1 and p<predictions[-1]:
                p=predictions[-1]

            predictions.append(p)
        all_predictions.append(predictions)
        
    predictions_df = pd.DataFrame(all_predictions, columns=["failure_number","p3","p7","p14","p30","p45","p60","p120"])
    predictions_df.to_csv(outPath, index=False, float_format='%.3f')
    
    print('Done')

if __name__ == '__main__':
    main()