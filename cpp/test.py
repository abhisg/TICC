from solver import *
import pandas as pd
import numpy as np
from collections import Counter

data = pd.read_csv('data.csv',header=None).values
act = []
for i in xrange(len(data)-4):
    act.append(np.hstack(data[i:i+5,:]))
act = np.array(act)
beta = 50
K = 3
rho = 1
lamb = np.full((25,25),0.001)
n = 5
w = 5
init_mu = NumpyList()
init_theta = NumpyList()
d2 = act.copy()
#np.random.seed(1)
#np.random.shuffle(d2)
for i in xrange(3):
    init_mu.push_back(np.mean(d2[500*i:min(500*(i+1),len(d2)),:],axis=0).reshape(1,25))
    #init_mu.push_back(np.random.rand(1,25))
    init_theta.push_back(np.linalg.inv(np.cov(d2[500*i:min(500*(i+1),len(d2)),:],rowvar=False)))
obj = Solver(K,beta,rho,act,lamb,n,w,init_mu,init_theta)
obj.Solve(10)
t1 = obj.obtainTheta(0)
#print t1
ass = obj.obtainAssignment();
print Counter([a for a in ass])


