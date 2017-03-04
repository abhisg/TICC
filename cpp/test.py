from solver import *
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv',header=None).values
act = []
for i in xrange(len(data)-5):
    act.append(np.hstack(data[i:i+5,:]))
act = np.array(act)
beta = 25
K = 3
rho = 1
lamb = np.full((25,25),0.05)
n = 5
w = 5
init_mu = NumpyList()
init_theta = NumpyList()
v1 = act.T
for i in xrange(3):
    v = np.random.rand(25,25)
    init_mu.push_back(np.mean(act[192*i:192*i+192,:],axis=0).reshape(1,25))
    #init_mu.push_back(np.random.rand(1,25))
    init_theta.push_back(np.cov(v1[:,192*i:192*i+192]))
obj = Solver(K,beta,rho,act,lamb,n,w,init_mu,init_theta)
obj.Solve(50)
t1,t2,t3 = obj.obtainTheta(0),obj.obtainTheta(1),obj.obtainTheta(2)
for t in t1:
    print t
