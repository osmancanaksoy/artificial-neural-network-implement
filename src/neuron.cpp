#include "neuron.h"
#include <QDebug>

Neuron::Neuron()
{
    input_dimension = 0;
    sample_size = 0;
    lr = error = 0;
    weights = nullptr;
    inputs = nullptr;
    outputs = nullptr;
}

Neuron::Neuron(double *w, double lr, double *inp, double *out, double error, int inp_dim, int samp_size)
{
    inputs = inp;
    outputs = out;
    this->lr = lr;
    this->error = error;
    input_dimension = inp_dim;
    sample_size = samp_size;
    set_weights(w);
}

Neuron::~Neuron()
{
    delete[] weights;
}

void Neuron::configure(int dim, int sample_size)
{
    input_dimension = dim;
    this->sample_size = sample_size;
}

void Neuron::set_weights(double *w)
{
    weights = new double[input_dimension + 1];
    for(int i = 0; i < input_dimension + 1; i++) {
        weights[i] = w[i];
    }
}

void Neuron::random_weights()
{
    weights = new double[input_dimension];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    for (int i = 0;i < input_dimension; i++)
    {
        weights[i] = dis(gen);
    }
}

double *Neuron::get_weights() const
{
    return weights;
}

void Neuron::set_inputs(double *val)
{
    inputs = val;
}

void Neuron::set_outputs(double *val)
{
    outputs = val;
}

double *Neuron::get_outputs() const
{
    return outputs;
}

void Neuron::convert_outputs()
{
    for(int i = 0; i < sample_size; i++) {
        if(outputs[i] == 2)
            outputs[i] = -1;
    }

}

void Neuron::set_lr(double val)
{
    lr = val;
}

double Neuron::get_lr() const
{
    return lr;
}

void Neuron::set_error(double val)
{
    error = val;
}

double Neuron::get_error() const
{
    return error;
}

int Neuron::get_how_many_cycle() const
{
    return how_many_cycle;
}

double Neuron::sgn(double val)
{
    return (val >= 0) ? 1 : -1;
}

void Neuron::perceptron_rule()
{
    int counter = 0;
    how_many_cycle = 0;
    while(counter != sample_size)
    {
        counter = 0;
        for (int i=0;i< sample_size;i++)
        {
            int o = calc_output(i);
            if(o == outputs[i]) { counter++; continue; }
            update_weights(i, o);
        }
        how_many_cycle++;
    }
}

void Neuron::delta_rule()
{
    double e = 1000;
    how_many_cycle = 0;
    while(0.5*e>error)
    {
        //std::cout << "error: " << e << std::endl;
        e = 0;
        for (int i=0;i< sample_size;i++)
        {
            double o = calc_activation(i);
            e += (outputs[i]-o)*(outputs[i]-o);
            update_weights_delta(i, o);
        }
        how_many_cycle++;
    }

}

int Neuron::calc_output(int index)
{
    double net = 0;
    for(int i=0;i<input_dimension;i++)
    {
        net += weights[i]*inputs[index*input_dimension+i];
    }
    net += weights[2];
    if(net>0) return 1;
    return 2;
}

double Neuron::calc_activation(int index)
{
    double net = 0;
    for(int i=0;i < input_dimension;i++)
    {
        net += weights[i]*(double)inputs[index*input_dimension+i];
    }
    net += weights[2];
    return (2.0 / (exp(-net) + 1.0)) - 1.0;
}

void Neuron::update_weights(int index, int output)
{
    double constant = lr*(outputs[index]-output);
    for (int i = 0; i < input_dimension + 1; i++)
    {
        weights[i] = weights[i]+constant*inputs[input_dimension*index+i];
    }
}

void Neuron::update_weights_delta(int index, int output)
{
    double derivative = 0.5*(1 - output*output);
    //    double derivative = output*(1-output);
    double constant = lr*(outputs[index]-output)*derivative;
    for (int i=0; i< input_dimension + 1; i++)
    {
        weights[i] = weights[i]+constant*inputs[input_dimension*index+i];
    }
}
