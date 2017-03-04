#ifndef SOLVER_H_
#define SOLVER_H_

#include <vector>
#include <cmath>
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Eigen"
#define PI 3.14159265358979323846

class Solver
{
    protected:
        std::vector<Eigen::MatrixXd> Theta;
        std::vector<Eigen::MatrixXd> mu;
        std::vector<Eigen::MatrixXd> Z;
        std::vector<Eigen::MatrixXd> U;
        std::vector<Eigen::MatrixXd> S;
        std::vector<std::vector<Eigen::MatrixXd> > assignments;
        Eigen::MatrixXd LLE;

        int K; //number of clusters
        double beta; //switching penalty
        double rho; //regularisation constant
        Eigen::MatrixXd data; //data points in nw x nw format
        Eigen::MatrixXd lambda; //sparseness matrix
        int n; //number of sensors
        int w; //window size
        void computeLLE();
        void Estep();
        void Mstep();

    public:
        Solver(int K, double beta, double rho, Eigen::MatrixXd data, Eigen::MatrixXd lambda, int n, int w,std::vector<Eigen::MatrixXd> init_mu,std::vector<Eigen::MatrixXd> init_theta) {
            this->K = K;
            this->beta = beta;
            this->rho = rho;
            this->data = data;
            this->lambda = lambda;
            this->n = n;
            this->w = w;
            for(int i = 0 ; i < K ;++i){
                Theta.push_back(init_theta[i]);
                Eigen::MatrixXd mat(n*w,n*w);
                mat.setZero(n*w,n*w);
                Z.push_back(mat);
                U.push_back(mat);
                S.push_back(mat);
                mu.push_back(init_mu[i]);
                std::vector<Eigen::MatrixXd> vec;
                vec.clear();
                assignments.push_back(vec);
            }
            LLE = Eigen::MatrixXd(data.rows(),K);
            LLE.setZero(data.rows(),K);
        }

        void Solve(int steps) {
            //M step is called before E step since the initialisation would have already been done
            for(int i = 0 ; i < steps; ++i){
                Estep();
                Mstep();
            }
        }

        Eigen::MatrixXd obtainTheta(int idx){
            return Theta[idx];
        }
};
            


#endif
