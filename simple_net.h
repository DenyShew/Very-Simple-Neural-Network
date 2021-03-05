//
// Created by denys on 04/03/2021.
//

#ifndef SIMPLE_NET_SIMPLE_NET_H
#define SIMPLE_NET_SIMPLE_NET_H

#include <eigen3/Eigen/Dense>
#include <random>
#include <ctime>
#include <vector>
#include <cmath>

class simple_net
{
public:
    simple_net(const std::vector<size_t>& topology) : rate(0.00381)
    {
        std::random_device rd;
        std::mt19937 mersenne(rd());
        std::uniform_real_distribution<> dist(-1.0, 1.0);
        this->errors.resize(topology.size());
        this->weights.resize(topology.size());
        this->outputs.resize(topology.size());
        this->bias.resize(topology.size());
        this->weights[0] = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(topology[0], topology[0]);
        this->outputs[0] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[0]);
        this->errors[0] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[0]);
        this->bias[0] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[0]);
        for(int i=1;i<weights.size();i++)
        {
            this->weights[i] =  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(topology[i], topology[i - 1]);
            this->outputs[i] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[i]);
            this->errors[i] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[i]);
            this->bias[i] = Eigen::Matrix<double, Eigen::Dynamic, 1>(topology[i]);
        }
        for(int i=0;i<topology.size();i++)
        {
            for(int j=0;j<this->weights[i].cols();j++)
            {
                for(int k=0;k<this->weights[i].rows();k++)
                {
                    this->weights[i](k,j) = dist(mersenne);
                }
            }
            for(int j=0;j<this->bias[i].rows();j++)
            {
                this->bias[i](j) = dist(mersenne);
            }
        }
    }
    void forward_prop(const Eigen::Matrix<double, Eigen::Dynamic, 1>& input_value)
    {
        this->input = input_value;
        this->outputs[0] = weights[0] * input_value;
        this->outputs[0]+=this->bias[0];
        this->outputs[0] = this->outputs[0].unaryExpr([](double x) {return tanh(x);});
        for(int i=1;i<outputs.size();i++)
        {
            this->outputs[i] = weights[i] * outputs[i - 1];
            this->outputs[i]+=this->bias[i];
            this->outputs[i] = this->outputs[i].unaryExpr([](double x) {return tanh(x);});
        }
    }
    void backward_prop(const Eigen::Matrix<double, Eigen::Dynamic, 1>& target)
    {
        this->errors.back() = this->outputs.back() - target;
        for(int i=0;i<this->errors.back().rows();i++)
        {
            this->errors.back() = this->errors.back().array() * this->outputs.back().unaryExpr([](double x){return 1 - x*x;}).array();
        }
        for(int i=this->errors.size() - 2;i>=0;i--)
        {
            this->errors[i] = this->weights[i + 1].transpose() * this->errors[i + 1];
            this->errors[i] = this->errors[i].array() * this->outputs[i].unaryExpr([](double x){return 1 - x*x;}).array();
        }
        for(int i=this->weights.size() - 1;i>0;i--)
        {
            this->weights[i] -= rate * this->errors[i] * this->outputs[i - 1].transpose();
        }
        this->weights[0] -= rate * this->errors[0] * this->input.transpose();
        for(int i=0;i<this->bias.size();i++)
        {
            this->bias[i] -= rate * this->errors[i];
        }
    }
    Eigen::Matrix<double, Eigen::Dynamic, 1> get_output()const
    {
        return this->outputs.back();
    }
private:
    simple_net() = delete;
private:
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> weights;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> outputs;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> errors;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 1>> bias;
    Eigen::Matrix<double, Eigen::Dynamic, 1> input;
    double rate;
};

#endif //SIMPLE_NET_SIMPLE_NET_H
