import numpy as np
import pandas as pd 
import os
     
#Read calibration data
data_path = os.path.join(os.getcwd(), 'EVO_Download', 'Training_data_2019.csv')
training_data = pd.read_csv(data_path, delimiter = ',', index_col=0)
training_data.index = pd.to_datetime(training_data.index)
training_data['Month'] = training_data.index.month
training_data['Days in Month'] = training_data.index.daysinmonth
training_data['Day of Week'] = training_data.index.dayofweek
training_data['Hour'] = ((training_data.index.hour) + training_data.index.minute/60)
training_data['Is_Weekday']=np.where(training_data['Day of Week']>=5, 0, 1)
training_data['SIN_Month']=np.sin(training_data['Month']*(2.* np.pi/12))
training_data['COS_Month']=np.cos(training_data['Month']*(2.* np.pi/12))
training_data['SIN_DayOfWeek']=np.sin(training_data['Day of Week']*(2.*np.pi/7))
training_data['COS_DayOfWeek']=np.cos(training_data['Day of Week']*(2.*np.pi/7))
training_data['SIN_Hour']=np.sin(training_data['Hour']*(2.*np.pi/24))
training_data['COS_Hour']=np.cos(training_data['Hour']*(2.*np.pi/24))
training_data.dropna ( axis=0 , inplace =True )
print('Number of columns in the training dataset: {}, number of data values: {}'.format(training_data.shape[1], training_data.shape[0]))

#Read testing data
data_path = os.path.join(os.getcwd(), 'EVO_Download', 'Post_no_load_2019.csv')
test_data = pd.read_csv(data_path, delimiter = ',', index_col=0)
test_data.index = pd.to_datetime(test_data.index)
test_data['Month'] = test_data.index.month
test_data['Days in Month'] = test_data.index.daysinmonth
test_data['Day of Week'] = test_data.index.dayofweek
test_data['Hour'] = ((test_data.index.hour) +test_data.index.minute/60)
test_data['Is_Weekday']=np.where(test_data['Day of Week']>=5, 0, 1)
test_data['SIN_Month']=np.sin(test_data['Month']*(2.* np.pi/12))
test_data['COS_Month']=np.cos(test_data['Month']*(2.* np.pi/12))
test_data['SIN_DayOfWeek']=np.sin(test_data['Day of Week']*(2.*np.pi/7))
test_data['COS_DayOfWeek']=np.cos(test_data['Day of Week']*(2.*np.pi/7))
test_data['SIN_Hour']=np.sin(test_data['Hour']*(2.*np.pi/24))
test_data['COS_Hour']=np.cos(test_data['Hour']*(2.*np.pi/24))
print('Number of columns in the testing dataset: {}, number of data values: {}'.format(test_data.shape[1], test_data.shape[0]))
