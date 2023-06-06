import pandas as pd
from datetime import datetime
import numpy as np
import os
from scipy.signal import find_peaks

def processData(dataDir):
    equipment = pd.read_csv(os.path.join(dataDir, "equipment_metadata.csv"))
    all_data = equipment
    all_data['failed'] = 0
    pcntChangePumpTemps = []
    pcntChangePumpSpeeds = []
    pcntChangePumpCurrents = []
    pcntChangeMotorTemps = []
    pcntChangeMotorTrqs = []
    cooldownsCount = []
    runDaysAll = []
    dailyRunDataAll = pd.DataFrame()

    # TODO - fix date parsing, this is fairly slow - avoid using dates if possible
    op_format = "%m/%d/%Y"
    run_format = "%Y-%m-%d"
    dateparse_op = lambda dates: datetime.strptime(dates[0:10], op_format)
    dateparse_run = lambda dates: datetime.strptime(dates[0:10], run_format)

    has_status = 'status' in equipment.columns

    for index, row in equipment.iterrows():
        fnum = row['failure_number']
        print("Processing device ", fnum, datetime.now())
        if has_status and row['status'] != 'Current' and row['censored']!='Yes':
                equipment.at[index,'failed']=1
                # TODO - add censored rows with failed=0 for each pump that failed. censor the data for the relevant time window (3-120 days)
        try:         
            dir = os.path.join(dataDir, 'operations-data')
            device_data = pd.read_csv(os.path.join(dir, str(fnum)+".csv"))
            device_data['date'] = pd.to_datetime(device_data['date'].str.slice(0,10), format=op_format)
            device_data['dt']=device_data['date'].dt.date
            device_data['pump_speed'] = device_data['pump_speed'].fillna(method='ffill').fillna(0)
            device_data['pump_current'] = device_data['pump_current'].fillna(method='ffill').fillna(0)
            device_data['motor_temp'] = device_data['motor_temp'].fillna(method='ffill').fillna(0)
            device_data['motor_trq'] = device_data['motor_trq'].fillna(method='ffill').fillna(0)
            device_data['pump_temp'] = device_data['pump_temp'].fillna(method='ffill').fillna(0)

            daily_data = device_data.groupby(['dt']).mean().reset_index()

            # load runhours data
            runhours = pd.DataFrame(columns=['failure_number', 'date', 'run_hours'])
            runDays = 0
            try:
                dir = os.path.join(dataDir, 'run-life')
                runhours = pd.read_csv(os.path.join(dir, str(fnum)+'_runhours.csv'))
                runhours['date'] = pd.to_datetime(runhours['date'].str.slice(0,10), format=run_format).dt.date
                runDays = runhours['run_hours'].sum() / 24
            except Exception as e:
                runDays = (daily_data['dt'].max() - daily_data['dt'].min()).days
                pass


            daily_run_data = pd.merge(daily_data, runhours, left_on='dt', right_on='date', how='left')
            active_daily_data = daily_run_data[daily_run_data['run_hours']>10]

            maxs = active_daily_data.max()
            mins = active_daily_data.min()

            # calculate rough percentage change of numerical variables over time. 
            # note - some variables contain too many empty or zero values (ex motor_trq). replace nan/inf later
            pcntChangePumpTemp = 100 * (maxs['pump_temp'].max() - mins['pump_temp']) / mins['pump_temp']
            pcntChangePumpSpeed = 100 * (maxs['pump_speed'] - mins['pump_speed']) / mins['pump_speed']
            pcntChangePumpCurrent = 100 * (maxs['pump_current'] - mins['pump_current']) / mins['pump_current']
            pcntChangeMotorTemp = 100 * (maxs['motor_temp'] - mins['motor_temp']) / mins['motor_temp']
            pcntChangeMotorTrq = 100 * (maxs['motor_trq'] - mins['motor_trq']) / mins['motor_trq']
            
            # approximation of cooldown count - find peaks of more than '10' temperature difference. should choose this value better, doesn't fit for all rows in train data
            # this is fairly slow
            peaks, _ = find_peaks(daily_run_data['pump_temp'], prominence=10)
            
            # TODO - repeat above for current changes, deadlock, etc

            pcntChangePumpTemps.append(pcntChangePumpTemp)
            pcntChangePumpSpeeds.append(pcntChangePumpSpeed)
            pcntChangePumpCurrents.append(pcntChangePumpCurrent)
            pcntChangeMotorTemps.append(pcntChangeMotorTemp)
            pcntChangeMotorTrqs.append(pcntChangeMotorTrq)
            cooldownsCount.append(len(peaks))
            runDaysAll.append(runDays)
            dailyRunDataAll = dailyRunDataAll.append(daily_run_data)        
        except Exception as e:
            print(e)
            pass


    all_data['pcnt_change_pump_temps'] = pcntChangePumpTemps
    all_data['pcnt_change_pump_speeds'] = pcntChangePumpSpeeds
    all_data['pcnt_change_pump_currents'] = pcntChangePumpCurrents
    all_data['pcnt_change_motor_temps'] = pcntChangeMotorTemps
    all_data['pcnt_change_motor_trqs'] = pcntChangeMotorTrqs
    all_data['cooldowns_count'] = cooldownsCount
    all_data['run_days'] = runDaysAll
    
    all_data["pcnt_change_pump_temps"].replace([np.inf, -np.inf], 0, inplace=True)
    all_data["pcnt_change_pump_speeds"].replace([np.inf, -np.inf], 0, inplace=True)
    all_data["pcnt_change_pump_currents"].replace([np.inf, -np.inf], 0, inplace=True)
    all_data["pcnt_change_motor_temps"].replace([np.inf, -np.inf], 0, inplace=True)
    all_data["pcnt_change_motor_trqs"].replace([np.inf, -np.inf], 0, inplace=True)
    
    all_data["pcnt_change_pump_temps"] = all_data["pcnt_change_pump_temps"].fillna(0)
    all_data["pcnt_change_pump_speeds"] = all_data["pcnt_change_pump_speeds"].fillna(0)
    all_data["pcnt_change_pump_currents"] = all_data["pcnt_change_pump_currents"].fillna(0)
    all_data["pcnt_change_motor_temps"] = all_data["pcnt_change_motor_temps"].fillna(0)
    all_data["pcnt_change_motor_trqs"] = all_data["pcnt_change_motor_trqs"].fillna(0)
    all_data["motor_power"] = all_data["motor_power"].fillna(equipment['motor_power'].mean())
    all_data["pump_size"] = all_data["pump_size"].fillna(equipment['pump_size'].mean())
    
    features=["pcnt_change_pump_temps", "pcnt_change_pump_speeds", "pcnt_change_pump_currents", "pcnt_change_motor_temps", "pcnt_change_motor_trqs", "cooldowns_count", "pump_size", "motor_power"]

    # one hot encoding for categorical values
    categories = ['vendor', 'pump_type', 'motor_type']
    all_data_encoded = pd.get_dummies(all_data, columns=categories)
    for c in all_data_encoded.columns:
        for categ in categories:
            if (categ + '_') in c:
                features.append(c)
                
    time_column = 'run_days'
    event_column = 'failed'
    
    return all_data_encoded, features, time_column, event_column, dailyRunDataAll

def processLSTM(dataDir, daily_data):
    # daily_data["event"] = 0
    print("preprocess started")
    train_data = {}
    event_data = {}
    run_format = "%Y-%m-%d"
    equipment = pd.read_csv(os.path.join(dataDir, "equipment_metadata.csv"))
    has_status = 'status' in equipment.columns
    equipment['failed'] = '0'
    for index, row in equipment.iterrows():
        if has_status and row['status'] != 'Current' and row['censored']!='Yes':
            # row['failed_date'] = pd.to_datetime(pd.Series(row['failed_date']).str.slice(0,10), format=run_format)
            equipment.at[index,'failed']= row['failed_date']
    equipment = equipment[['failure_number', 'failed']]
    equipment = equipment.set_index('failure_number')
    
    
    #delete not required columns for lstm from daily data,failure_number_x column will be used below, and will be dropped. 
    daily_data = daily_data.drop(['failure_number_y', 'date'], axis=1)
    for column in daily_data.columns:
        daily_data[column] = daily_data[column].fillna(0) 
    #manipulate data for lstm : 
    # lstmdata --> key : key_data 
    # key: unique failure numbers 
    # key_data : corresponding daily list data of failure number pump
    max = 0
    for key in daily_data["failure_number_x"].unique():
        daily_data_key = daily_data[daily_data['failure_number_x'] == key]
        event = np.zeros(daily_data_key.shape[0])
        shape = daily_data_key['dt'].shape[0]
        if shape > max : max = shape
        #find daily data of key numbered pump and drop failure number x column and add to dict by key
        if equipment.loc[key]['failed'] != '0':
            fail_date = pd.to_datetime(equipment.loc[key]['failed'])
            #
            for index, row in daily_data[daily_data['failure_number_x'] == key].iterrows():
                if row['dt'] == fail_date:
                    # daily_data[daily_data['failure_number_x'] == key].loc[index,'event'] = 1 
                    event[index] = 1
        df = daily_data[daily_data['failure_number_x'] == key].reindex(range(2468), fill_value=0)
        train_data[key] = df.drop(['failure_number_x', 'dt'], axis=1).to_numpy()
        # train_data[key] = daily_data[daily_data['failure_number_x'] == key].drop(['failure_number_x', 'dt'], axis=1).to_numpy()
        event_data[key] = event
        
    return train_data, event_data, equipment

def deleteMissing(daily_data, event_data, equipment, threshold=51):
    '''For some of the pumps there are not enough daily data to be included in train. 
    Deletes daily data for a pump if it is under a threshold.Default threshold is 50'''
    print('deleting daily data under threshold.')
    missing_value, missing_index, missing_length = [], [], [] 
    train_features, event_features = [], []
    # event = event.to_numpy()
    for index, value in (enumerate(daily_data)):
        if len(daily_data[value]) < threshold:
            missing_value.append(value)
            missing_index.append(index)
            missing_length.append(len(daily_data[value]))
    for value in missing_value:
        del daily_data[value]
        del event_data[value]
    # event = np.delete(event, missing_index)
    # for index in missing_index:
    #     event.remove(index)
    for index, value in (enumerate(daily_data)):
        train_features.append(daily_data[value])
        event_features.append(event_data[value])
    return train_features, event_features

