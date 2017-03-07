#include "solver.hpp"
#include <iostream>

void Solver::computeLLE() {
    double constant = -n/2.0*log(2*PI);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    for(int i = 0 ; i < K; ++i){
        double logdet = -1/(2.0*log(Theta[i].determinant()));
        for(int j = 0 ; j < data.rows(); ++j){
            auto vec = (data.row(j)-mu[i]);
            LLE(j,i) = -0.5*((vec*Theta[i]*vec.transpose())(0,0))+constant+logdet;
        }
    }
}

void Solver::Estep() {
    //compute LLE
    computeLLE();
    std::vector<double> prevcost;
    std::vector<double> currentcost;
    std::vector<std::vector<int>> currentpath;
    for(int i = 0 ; i < K; ++i){
        prevcost.push_back(0);
        currentcost.push_back(0);
        std::vector<int> assign;
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
    current_optimal = currentpath[minindex];
    
    //provide points to empty clusters
    std::map<int,int> cluster_count;
    std::unordered_set<int> assigned_points;
    for(int i = 0 ; i < K; ++i){
        cluster_count[i] = 0;
    }
    for(auto cluster : current_optimal) {
        cluster_count[cluster]++;
    }
    for(int i = 0 ; i < K ; ++i){
        if (cluster_count[i] == 0) {
            //pick one point and its adjancent timestep randomly and assign to the new cluster
            int counter = 0;
            while(counter < current_optimal.size()){
                int rand_idx = rand() % (data.rows()-1);
                if(assigned_points.find(rand_idx) == assigned_points.end()) {
                    current_optimal[rand_idx] = i;
                    current_optimal[rand_idx+1] = i;
                    assigned_points.insert(rand_idx);
                    assigned_points.insert(rand_idx+1);
                    break;
                }
                counter++;
            }
        }
    }
    //assign points to clusters
    for(int i = 0 ; i < K ;++i){
        assignments[i].clear();
    }
    for(int i = 0 ; i < current_optimal.size(); ++i){
        assignments[current_optimal[i]].push_back(data.row(i));
    }
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    for(int i = 0 ; i < K ; ++i) {
        std::cout << i << " " << assignments[i].size() << " ";
        if(assignments[i].size() > 0){
            //compute the mean
            mu[i].setZero(1,n*w);
            for(auto mat : assignments[i]){
                mu[i] += mat;
            }
            mu[i] /= assignments[i].size();
            //compute the empirical covariance across of timestep 0 with each other timestep in [0,w]
            S[i].setZero(n*w,n*w);
            for(auto mat : assignments[i]) {
                S[i] += (mat-mu[i]).transpose() * (mat-mu[i]);
            }
            S[i] /= assignments[i].size();
        }
    }
    std::cout << "\n";
}

void Solver::Mstep(){
    //for now keep the number of iterations fixed
    int counter = 0;
    while(counter < 1000){
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 32)
        #endif
        for(int i = 0 ; i < K ;++i) {
            if(assignments[i].size() > 0){
                //theta update
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp((Z[i]-U[i])/rho-S[i]);
                Eigen::VectorXd eig = decomp.eigenvalues();
                for (int j = 0; j < eig.rows(); ++j ){
                    eig(j,0) += sqrt(eig(j,0) * eig(j,0) + 4*rho);
                }
                Eigen::MatrixXd D = eig.asDiagonal();
                Eigen::MatrixXd Q = decomp.eigenvectors();
                Theta[i] = rho/2.0 * Q * D * Q.transpose();
                auto SL = Theta[i] + U[i];
                //Z update
                for(int j = 0 ; j < w; ++j){
                    Eigen::MatrixXd update(n,n);
                    Eigen::MatrixXd updateS(n,n);
                    Eigen::MatrixXd updateQ(n,n);
                    updateQ.setZero(n,n);
                    updateS.setZero(n,n);
                    for(int k = j ; k < w; ++k){
                        updateQ += (lambda.block(k*n,(k-j)*n,n,n) + lambda.block((k-j)*n,k*n,n,n).transpose());
                        updateS += rho*(SL.block(k*n,(k-j)*n,n,n) + SL.block((k-j)*n,k*n,n,n).transpose());
                    }
                    double denom = 2*rho*(w-j);
                    for(int i1 = 0; i1 < n; ++i1){
                        for(int i2 = 0; i2 < n; ++i2){
                            if(updateS(i1,i2) > updateQ(i1,i2)){
                                update(i1,i2) = (updateS(i1,i2) - updateQ(i1,i2))/denom;
                            } else if(updateS(i1,i2) <  -updateQ(i1,i2)){
                                update(i1,i2) = (updateS(i1,i2) + updateQ(i1,i2))/denom;
                            } else {
                                update(i1,i2) = 0;
                            }
                        }
                    }
                    for(int k = j ; k < w; ++k){
                        Z[i].block(k*n,(k-j)*n,n,n) = update;
                        Z[i].block((k-j)*n,k*n,n,n) = update.transpose();
                    }
                }
                //U update
                U[i] += Theta[i] - Z[i];
            }
        }
        counter++;
    }
}






