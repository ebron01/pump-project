from datetime import datetime
import pandas as pd
import argparse
import os

"""
Pump Failure Prediction Data Science Challenge offline scorer
Usage: python scorer.py --pred <path-to-solution> --truth <path-to-truth-file> --out_dir <path-to-output-dir>
The truth file is a .csv file with at least these columns:
  start_date,status,censored,failure_number,last_date,test_date
where
- the start_date,status,censored,failure_number fields are the same as given in the equipment_metadata.csv
- last_date is failed_date if the device failed, or the last known date present in the operations data file
- test_date is the date when the predictions for this device must be made
Feel free to modify this to suit your needs, and use the contest forum to ask questions.
"""

outDir = ''
dateFormat = '%Y-%m-%d'

def writeScore(s):
    path = os.path.join(outDir, 'result.txt')
    out = open(path, 'w')
    out.write(str(s))
    out.close()

id2ps = {}

def checkPreds(preds):
    for _, row in preds.iterrows():
        arr = row.to_numpy()
        id = arr[0]
        ps = arr[1:]
        if len(ps) != 7:
            return -3, f'Wrong number of predictions for {id}'
        for i in range(7):
            p = ps[i]
            if p < 0 or p > 1:
                return -4, f'Predictions must be between 0 and 1'
            if i > 0 and p < ps[i-1]:
                return -5, f'Predictions must be monotonic increasing'
        id2ps[id] = ps
    return 1, None

times =   [3, 7, 14, 30, 45, 60, 120]
weights = [1, 1, 1,  2,  2,  2,  1]

def score1(probs, t0, tEvent, censored):
    weightSum = 0
    r2sum = 0
    for i in range(len(times)):
        dt = times[i]
        t = t0 + dt
        pred = probs[i]
        if censored and t >= tEvent:
            break
        truth = 0
        if t >= tEvent:
            truth = 1
        w = weights[i]
        weightSum += w
        r2sum += w * (truth - pred) ** 2

    ret = 0
    if weightSum > 0:
        ret = r2sum / weightSum
    return ret

def main():
    parser = argparse.ArgumentParser()
    # ignore the defaults here, or rewrite them to match your setup
    parser.add_argument('--truth', type=str, default='../data/scorer-test/equipment_metadata_private.csv', help='path to gt file')
    parser.add_argument('--pred', type=str, default='../data/scorer-test/solution.csv', help='path to prediction file')
    parser.add_argument('--out_dir', type=str, default='../data/scorer-test/', help='output folder')
    args = parser.parse_args()

    global outDir
    outDir = args.out_dir
    os.makedirs(outDir, exist_ok=True)

    try:
        print('Reading truth data from', args.truth)
        truth = pd.read_csv(args.truth)
    except Exception as e:
        print('Error reading truth: ', str(e))
        writeScore(-1)
        exit(-10)
    try:
        print('Reading predictions from', args.pred)
        pred = pd.read_csv(args.pred)
    except Exception as e:
        print('Error reading predictions: ', str(e))
        writeScore(-1)
        exit(-11)

    n = len(truth)
    if len(pred) != n:
        print(f'Solution file must contain {n} predictions')
        writeScore(-1)
        exit(-1)

    try:
        err, msg = checkPreds(pred)
    except Exception as e:
        print('Error in predictions: ', str(e))
        writeScore(-1)
        exit(-2)

    if err != 1:
        print(msg)
        writeScore(-1)
        exit(err)

    sum = 0
    for index, row in truth.iterrows():
        # start_date,status,censored,failure_number,last_date,test_date
        # 1998-12-24,Current,No,123456,1999-12-09,1999-02-03
        failed = row['status'] == 'Removed'
        censored = row['censored'] == 'Yes'
        if not failed:
            censored = True
        startDate = datetime.strptime(row['start_date'], dateFormat)
        testDate = datetime.strptime(row['test_date'], dateFormat)
        lastDate = datetime.strptime(row['last_date'], dateFormat)
        id = row['failure_number']
        t0 = (testDate - startDate).days
        tEvent = (lastDate - startDate).days

        if not id in id2ps:
            print(f'No predictions for {id}')
            writeScore(-1)
            exit(-6)
        pArr = id2ps[id]
        score = score1(pArr, t0, tEvent, censored)
        sum += score

    sum = (sum / n) ** 0.5
    sum = 100 * (1 - sum)
    print(f'Score: {sum}')
    writeScore(sum)

if __name__ == '__main__':
    main()