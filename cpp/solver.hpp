#ifndef SOLVER_H_
#define SOLVER_H_

#include<vector>
#include<cmath>
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
        std::vector<std::vector<Eigen::MatrixXd>> assignments;
        Eigen::MatrixXd LLE;

        int K; //number of clusters
        double beta; //switching penalty
        Eigen::MatrixXd data; //data points in nw x nw format
        Eigen::MatrixXd lambda; //sparseness matrix
        int n; //number of sensors
        int w; //window size

    public:
        Solver(int K, double beta, double rho, Eigen::MatrixXd data, Eigen::MatrixXd lambda, int n, int w,std::vector<Eigen::MatrixXd> init_mu,std::vector<Eigen::MatrixXd> init_theta) {
            K = this->K;
            beta = this->beta;
            rho = this->rho;
            data = this->data;
            lambda = this->lambda;
            n = this->n;
            w = this->w;
            for(int i = 0 ; i < K ;++i){
                Theta[i] = init_theta[i];
                Z[i] = Eigen::MatrixXd::Zero(n*w,n*w);
                U[i] = Eigen::MatrixXd::Zero(n*w,n*w);
                mu[i] = init_mu[i];
                S = Eigen::MatrixXd::Zero(n*w,n*w);
                std::vector<Eigen::MatrixXd> vec;
                assignments.push_back(vec);
            }
            LLE = Eigen::MatrixXd::Zero(data.rows(),K);
        }
        void computeLLE();
        void Estep();
        void Mstep();
        Eigen::MatrixXd Solve(int);
};
            


#endif
