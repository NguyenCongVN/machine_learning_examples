# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from flask import Flask, jsonify, request
from scipy.stats import beta

# create an app
app = Flask(__name__)


# define bandits
# there's no "pull arm" here
# since that's technically now the user/client
class Bandit:
  def __init__(self, name):
    self.name = name
    self.clks = 0
    self.views = 0

  def sample(self):
    a = 1 + self.clks
    b = 1 + self.views
    return np.random.beta(a,b)
  def AddViews(self):
    self.views += 1
  def AddClicks(self):
    self.clks += 1
  # TODO - what else does the Bandit need to do?


# initialize bandits
banditA = Bandit('A')
banditB = Bandit('B')



@app.route('/get_ad')
def get_ad():
  # TODO
  sampleA = banditA.sample()
  sampleB = banditB.sample()
  if sampleA > sampleB:
    banditA.AddViews()
    return jsonify({'advertisement_id': 'A'})
  else:
    banditB.AddViews()
    return jsonify({'advertisement_id': 'B'})


@app.route('/click_ad', methods=['POST'])
def click_ad():
  result = 'OK'
  if request.form['advertisement_id'] == 'A':
    # TODO
    banditA.AddClicks()
    pass
  elif request.form['advertisement_id'] == 'B':
    # TODO
    banditB.AddClicks()
    pass
  else:
    result = 'Invalid Input.'

  # nothing to return really
  return jsonify({'result': result})


if __name__ == '__main__':
  app.run(host='127.0.0.1', port='8888')