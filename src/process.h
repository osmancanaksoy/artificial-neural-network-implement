#ifndef PROCESS_H
#define PROCESS_H
#include "math.h"
#include <vector>

float sgn(float x);
float sigmoid(float net);
float sigmoid_derivative(float net);

float* init_array_random(int len);
void z_score_parameters(float* x, int size, int dim, float* mean, float* std);

//Single Layer
float single_neuron_delta_rule(float* inputs, float* weights, float* targets, float* bias,float lr, int sample_size, int input_dim);
float multi_class_delta_rule(float* inputs, float* weights, std::vector<float> targets, float* bias, float lr, int sample_size, int input_dim, int class_count);

//Two Layer
float multi_layer_delta_rule(float* inputs, float* v, float*w, float* targets, float* hidden_bias, float* output_bias,
                              int input_dim, int hidden_size, int output_size, int num_samples, float lr);
float multi_layer_delta_with_momentum(float* inputs, float* v, float*w, float* targets, float* hidden_bias, float* output_bias,
                                      double* prev_hidden_delta, double* prev_out_delta, double* prev_hidden_bias_delta, double* prev_out_bias_delta,int input_dim, int hidden_size, int output_size, int num_samples, float lr, float momentum);

int test_forward(float* x, float* weight, float* bias, int num_class, int input_dim);
int test_forward_2(float* x, float* v, float* w, float* b1, float* b2, int num_class, int input_dim, int hidden_dim);
#endif // PROCESS_H
