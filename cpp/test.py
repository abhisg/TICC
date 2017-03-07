from solver import *
import pandas as pd
import numpy as np
from collections import Counter

data = pd.read_csv('data.csv',header=None).values
act = []
for i in xrange(len(data)-5):
    act.append(np.hstack(data[i:i+5,:]))
act = np.array(act)
beta = 0
K = 3
rho = 1
lamb = np.full((25,25),0.01)
n = 5
w = 5
init_mu = NumpyList()
init_theta = NumpyList()
d2 = act.copy()
np.random.shuffle(d2)
for i in xrange(3):
    v = np.random.rand(25,25)
    #init_mu.push_back(np.mean(d2[200*i:min(200*(i+1),len(d2)),:],axis=0).reshape(1,25))
    init_mu.push_back(np.random.rand(1,25))
    init_theta.push_back(np.linalg.inv(np.cov(d2[200*i:min(200*(i+1),len(d2)),:],rowvar=False)))
obj = Solver(K,beta,rho,act,lamb,n,w,init_mu,init_theta)
obj.Solve(100)
ass = obj.obtainAssignment();
print Counter([a for a in ass])


