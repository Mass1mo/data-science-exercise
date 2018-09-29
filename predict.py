
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.externals import joblib
import time
import csv

print('Getting data...')
tr_url = 'https://raw.githubusercontent.com/Usio-Energy/data-science-exercise/master/usage_train.csv'
#tr_data = pd.read_csv('usage_train.csv')

tr_data = pd.read_csv(tr_url)
tr_data['datetime'] = pd.to_datetime(tr_data['datetime'])
tr_data = tr_data.set_index('datetime').dropna()

test_url = 'https://raw.githubusercontent.com/Usio-Energy/data-science-exercise/master/usage_test.csv'

#test_data = pd.read_csv(test_url)
test_data = pd.read_csv('usage_test.csv')
test_data['datetime'] = pd.to_datetime(test_data['datetime'])
test_data = test_data.set_index('datetime')

print('Attempting to load model from current working directory...')
try:
     model_fit = joblib.load('EN_model.pkl')
except:
     print('Unable to load model. Terminating script...')
     quit()

print('Rebuilding column index...')

def preproc(df):
    #df['datetime'] = pd.to_datetime(df['datetime'])
    #df = df.set_index('datetime')

    # this interval because we have to cover 7 days ahead
    for i in range(1,336*2):
        df['lag_{}'.format(i)] = df.usage.shift(i)
    
    #df['minute'] = df.index.minute
    df['hour'] = (df.index.hour.astype(str) + ':' +
       df.index.minute.astype(str))
    df['weekday'] = df.index.weekday # 5 and 6 represent weekends
    df['is_weekend'] = df.weekday.isin([5,6])*1
    df['month'] = df.index.month
    df = pd.get_dummies(df, 
                         columns = ['id','hour','weekday','month'],
                    prefix=['id','hour','day','month'])#, drop_first = True)
    df = df.dropna()

    #df = df.tail(350)
    df = df.drop('usage',axis = 1)
    
    return df.tail(1)
          
colnames = list(preproc(tr_data.copy())) 

print('Performing predictions...')

#households = test_data['id'].unique()

start = time.time()

test_yhat = test_data
test_yhat['usage'] = 0

for i in np.arange(0,len(test_data)): #np.arange(0,2): #
     next_obs = test_data[i:i+1].copy()
     #print(next_obs)

     h = next_obs.id[0]
     month = next_obs.index.month[0]
     wd = next_obs.index.weekday[0]
     
     this_set = tr_data[tr_data['id'] == h]
     
     for j in range(1,336*2):
          next_obs['lag_{}'.format(j)] = this_set.usage.iloc[-j]
     
     next_obs = next_obs.reindex(columns = colnames, 
                                fill_value=0)
     next_obs['id_'+format(h)] = 1
     next_obs['month_'+format(month)] = 1
     next_obs['hour_'+next_obs.index.hour[0].astype(str) + ':' +
       next_obs.index.minute[0].astype(str)] = 1
     next_obs['day_'+format(wd)] = 1 # 5 and 6 represent weekends
     next_obs['is_weekend'] = next_obs.index.weekday.isin([5,6])*1
     
     pred = model_fit.predict(next_obs)
     #print(pred)
     
     test_yhat.loc[(test_yhat.index == next_obs.index[0]) &
                     (test_yhat.id == h),'usage'] = pred
                               
     #print(test_yhat.head())
     
     tr_data = tr_data.append(
               test_yhat.loc[(test_yhat.index == next_obs.index[0]) &
                     (test_yhat.id == h)], ignore_index = True)

end = time.time()
print(end-start)

try:
     print('Saving results in current working directory...')
     csv.writer('results.csv', test_yhat)
except:
     print('Couldn\'t access working directory for saving results...')

print('Finished. Results stored in \'test_yhat\' ')

