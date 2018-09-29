
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV
from sklearn.externals import joblib

print('Getting the data...')

tr_url = 'https://raw.githubusercontent.com/Usio-Energy/data-science-exercise/master/usage_train.csv'

tr_data = pd.read_csv(tr_url)
# tr_data = pd.read_csv('usage_train.csv')
tr_data['datetime'] = pd.to_datetime(tr_data['datetime'])
tr_data = tr_data.set_index('datetime')

print('Performing pre-processing...')
# this interval because we have to cover 7 days ahead
for i in range(1,336*2):
    tr_data['lag_{}'.format(i)] = tr_data.usage.shift(i)
    
#tr_data['minute'] = tr_data.index.minute
tr_data['hour'] = (tr_data.index.hour.astype(str) + ':' +
       tr_data.index.minute.astype(str))
tr_data['weekday'] = tr_data.index.weekday # 5 and 6 represent weekends
tr_data['is_weekend'] = tr_data.weekday.isin([5,6])*1
tr_data['month'] = tr_data.index.month
tr_data = pd.get_dummies(tr_data, 
                         columns = ['id','hour','weekday','month'],
                    prefix=['id','hour','day','month'])#, drop_first = True)
tr_data = tr_data.dropna()

tr_labels = tr_data['usage']
tr_feat = tr_data.drop('usage',axis = 1)

# let's use elastic net

model = ElasticNetCV(cv= 10, 
                     normalize = True,
                     fit_intercept = False,
                     random_state = 1, 
                     #max_iter = 10000,
                     n_jobs = -1, 
                     selection = 'random')

print('Estimating Elastic Net with 10-fold CV...')
model_fit = model.fit(tr_feat, tr_labels)

print('Alpha for the estimated model is:', model_fit.alpha_) 

tr_yhat = model_fit.predict(tr_feat)
MSE = mean_squared_error(tr_labels, tr_yhat)

print('Mean Squared Error on the training set is:', np.sqrt(MSE))

total_f = len(model_fit.coef_)
retained_f = sum(model_fit.coef_ != 0)

print(retained_f, 'features retained out of', total_f,
      'initial features')

#plt.plot(tr_labels[10000:11000])
#plt.show()

#plt.plot(tr_yhat[10000:11000])
#plt.show()

#(tr_labels - tr_yhat).describe()
# save model
try:
     print('Saving model to current working directory...')
     joblib.dump(model_fit, 'EN_model.pkl') 
     print('Finished')
       
except: 
     print('Couldn\'t save model to current working directory')


     

