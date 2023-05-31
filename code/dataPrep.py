from os.path import isfile, join
from os import listdir
import csv
import pandas as pd
import pickle


pd.options.display.max_rows = 9999
df = pd.read_csv('./data/train/equipment_metadata.csv')

train_operation_path = "/work/data/train/operations-data/"
train_runlife_path = "/work/data/train/run-life/"
test_operation_path = "/work/data/test/operations-data/"
test_runlife_path = "/work/data/test/run-life/"

train_operation_files = [train_operation_path + f for f in listdir(
    train_operation_path) if isfile(join(train_operation_path, f))]
train_runlife_files = [train_runlife_path + f for f in listdir(
    train_runlife_path) if isfile(join(train_runlife_path, f))]
test_operation_files = [test_operation_path + f for f in listdir(
    test_operation_path) if isfile(join(test_operation_path, f))]
test_runlife_files = [test_runlife_path + f for f in listdir(
    test_runlife_path) if isfile(join(test_runlife_path, f))]

li = []

for filename in train_operation_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_pickle("/work/wdata/dummy.pkl")
# df_train_operation = pd.concat(map(pd.read_csv, train_operation_files))
print("done")
