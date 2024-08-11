#ifndef NEURON_H
#define NEURON_H

#include "point.h"
#include <random>
#include <math.h>
#include <iostream>

class Neuron
{
public:
    Neuron();
    Neuron(double* w, double lr, double* inp, double* out,double error, int inp_dim, int samp_size);
    ~Neuron();

    void configure(int dim, int sample_size);

    void set_weights(double *w);
    void random_weights();
    double* get_weights() const;

    void set_inputs(double *val);

    void set_outputs(double* val);
    double* get_outputs() const;
    void convert_outputs();

    void set_lr(double val);
    double get_lr() const;

    void set_error(double val);
    double get_error() const;

    int get_how_many_cycle() const;

    double sgn(double val);
    void perceptron_rule();

    void delta_rule();
private:
    int input_dimension, sample_size, how_many_cycle;
    //Point* inputs;
    double *weights, *inputs, *outputs;
    double lr, error;

    int calc_output(int index);

    double calc_activation(int index);

    void update_weights(int index, int output);
    void update_weights_delta(int index, int output);
};

#endif // NEURON_H
