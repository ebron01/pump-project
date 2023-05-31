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

    # TODO - fix date parsing, this is fairly slow - avoid using dates if possible
    op_format = "%m/%d/%Y"
    run_format = "%Y-%m-%d"
    dateparse_op = lambda dates: datetime.strptime(dates[0:10], op_format)
    dateparse_run = lambda dates: datetime.strptime(dates[0:10], run_format)

    has_status = 'status' in equipment.columns

    for index, row in equipment.iterrows():
        fnum = row['failure_number']
        # print("Processing device ", fnum, datetime.now())
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
    
    return all_data_encoded, features, time_column, event_column