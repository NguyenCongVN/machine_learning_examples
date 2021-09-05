import pandas as pd
from scipy.stats import chi2
import numpy as np
data = pd.read_csv('advertisement_clicks.csv')
a = data[data['advertisement_id'] == 'A']
b = data[data['advertisement_id'] == 'B']
a = a['action']
click_A = len([i for i in a if i == 1])
notClick_A = len(a) - click_A
b = b['action']
click_B = len([i for i in b if i == 1])
notClick_B = len(b) - click_B
print(click_A , click_B , notClick_A , notClick_B)
def get_p_value(T):
  # same as scipy.stats.chi2_contingency(T, correction=False)
  det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
  c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
  p = 1 - chi2.cdf(x=c2, df=1)
  return p
print([[click_A , notClick_A], [click_B , notClick_B]])
print(get_p_value(np.array([[click_A , notClick_A],[click_B , notClick_B]])))