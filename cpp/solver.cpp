#include "solver.h"

void Solver::computeLLE() {
    double constant = -n/2.0*log(2*PI);
    for(int i = 0 ; i < K; ++i){
        double logdet = -1/(2.0*log(Theta[i].determinant()));
        for(int j = 0 ; j < data.rows(); ++j){
            auto vec = (data.row(i)-mu[i]);
            LLE(j,i) = -0.5*(vec.transpose()*Theta[i]*vec)+logdet+constant;
        }
    }
}

void Solver::Estep() {
    //compute LLE
    computeLLE();
    vector<double> prevcost;
    vector<double> currentcost;
    vector<vector<int>> currentpath;
    for(int i = 0 ; i < K; ++i){
        prevcost.push_back(0);
        currentcost.push_back(0);
        vector<int> assign;
        currentpath.push_back(assign);
    }
    int minindex = 0;
    for(int i = 0 ; i < data.rows(); ++i){
        for(int j = 0 ; j < K; ++j){
            if(prevcost[minindex] + beta > prevcost[j]){
                currentcost[j] = prevcost[j] - LLE(i,j);
            } else {
                currentcost[j] = prevcost[minindex] + beta - LLE(i,j);
                currentpath[j] = currentpath[minindex];
            }
            currentpath[j].push_back(j);
        }
        prevcost = currentcost;
        for(int j = 0 ; j < K; ++j){
            if(prevcost[j] < prevcost[minindex]){
                minindex = j;
            }
        }
    }
    vector<int> optimal = currentpath[minindex];
    //assign points to clusters
    for(int i = 0 ; i < K ;++i){
        assignments[i].clear();
    }
    for(int i = 0 ; i < optimal.size(); ++i){
        assignments[optimal[i]].push_back(data.row(i));
    }
    for(int i = 0 ; i < K ; ++i) {
        if(assignments[i].size() > 0){
            //compute the mean
            mu[i] = Eigen::MatrixXd::Zero(1,n*w);
            for(auto mat : assignments[i]){
                mu[i] += mat;
            }
            mu[i] /= assignments[i].size();
            //compute the empirical covariance across w timestamps
            for(int j = 0 ; j < w; ++j){
                auto cov = Eigen::MatrixXd::Zero(n,n);
                for(int k = 0 ; k < assignments[i].size();++k){
                    auto dat = assignments[i][k];
                    cov += (dat.block(0,j*n,1,n)-mu[i].block(0,j*n,1,n)).transpose() * (dat.block(0,0,1,n)-mu[i].block(0,0,1,n));
                }
                cov /= assignments[i].size();
                for(int k = j ; k < w; ++k){
                    S[i].block(k*n,(k-j)*n,n,n) = cov;
                    S[i].block((k-j)*n,k*n,n) = cov.tranpose();
                }
            }
        }
    }
}

void Solver::Mstep(){
}






