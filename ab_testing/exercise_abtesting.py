import numpy as np
import pandas as pd
from scipy import stats
# get data
data = pd.read_csv('advertisement_clicks.csv')
a = data[data['advertisement_id'] == 'A']
b = data[data['advertisement_id'] == 'B']
a = a['action']
b = b['action']

# get ttest
t , p = stats.ttest_ind(a , b)
print('t : {t} , p : {p}'.format(t=t , p=p))