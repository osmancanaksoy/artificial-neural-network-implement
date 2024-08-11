#include "process.h"
#include <vector>
#include <random>


float sgn(float x) {
    return (x >= 0) ? 1 : -1;
}

float sigmoid(float net) {
    return (2 / (1.0 + exp(-net))) - 1;
}

float sigmoid_derivative(float net) {
    return 0.5 * (1 - pow(sigmoid(net), 2));
}

float *init_array_random(int len)
{
    srand(time(0));
    float* arr = new float[len];
    for (int i = 0; i < len; i++)
        arr[i] = ((float)rand() / RAND_MAX) - 0.5f;
    return arr;
}

void z_score_parameters(float *x, int size, int dim, float *mean, float *std)
{
    float* total = new float[dim];

    for (int i = 0; i < dim; i++) {
        mean[i] = std[i] = total[i] = 0.0;
    }
    for (int i = 0; i < size; i++)
        for (int j = 0; j < dim; j++)
            total[j] += x[i * dim + j];
    for (int i = 0; i < dim; i++)
        mean[i] = total[i] / float(size);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < dim; j++)
            std[j] += ((x[i * dim + j] - mean[j]) * (x[i * dim + j] - mean[j]));

    for (int j = 0; j < dim; j++)
        std[j] = sqrt(std[j] / float(size));

    delete[] total;
}

float single_neuron_delta_rule(float *inputs, float *weights, float *targets, float* bias, float lr, int sample_size, int input_dim)
{
    float error = 0.0;
    for(int i = 0; i < sample_size; i++) {
        float net = 0;
        for(int j = 0; j < input_dim; j++) {
            net += weights[j]*inputs[i * input_dim + j];
        }
        net += bias[0];

        float output = sigmoid(net);
        float output_der = sigmoid_derivative(net);

        float desired;
        if(targets[i] == 1) {
            desired = -1.0;
        }
        else {
            desired = 1.0;
        }

        for(int j = 0; j < input_dim; j++) {
            weights[i] += lr * (desired - output) * output_der *inputs[i * input_dim + j];
        }
        bias[0] += lr * (desired - output) * output_der * 1;

        error += 0.5 * (desired - output) * (desired - output);
    }
    error /= sample_size;
    return error;
}

float multi_class_delta_rule(float *inputs, float *weights, std::vector<float> targets, float* bias, float lr, int sample_size, int input_dim, int class_count)
{
    float total_error = 0.0;

    for(int i = 0; i < sample_size; i++) {
        std::vector<float> net(class_count, 0.0);
        for(int j = 0; j < input_dim; j++) {
            for (int classIndex = 0; classIndex < class_count; classIndex++)
            {
                net[classIndex] += weights[classIndex * input_dim + j] * inputs[i * input_dim + j];
            }
        }

        for (int classIndex = 0; classIndex < class_count; classIndex++)
        {
            net[classIndex] += bias[classIndex];
        }

        for (int classIndex = 0; classIndex < class_count; classIndex++)
        {
            float output = sigmoid(net[classIndex]);
            float output_der = sigmoid_derivative(net[classIndex]);

            float target = targets[i * class_count + classIndex];

            total_error += 0.5 * (target - output) * (target - output);

            for (int k = 0; k < input_dim; k++)
            {
                weights[classIndex * input_dim + k] += lr * (target - output) * output_der * inputs[i * input_dim + k];
            }

            bias[classIndex] += lr * (target - output) * output_der;
        }
    }

    total_error /= sample_size;
    return total_error;
}

float multi_layer_delta_rule(float* inputs, float* v, float*w, float* targets, float* hidden_bias, float* output_bias,
                              int input_dim, int hidden_size, int output_size, int num_samples, float lr)
{
    float total_error = 0.0;
    for(int i = 0; i < num_samples; i++) {
        std::vector<float> hidden_layer_output(hidden_size);

        for(int j = 0; j < hidden_size; j++) {
            hidden_layer_output[j] = 0.0;
            for(int k = 0; k < input_dim; k++) {
                hidden_layer_output[j] += inputs[i * input_dim + k] * v[k * hidden_size + j];
            }
            hidden_layer_output[j] += hidden_bias[j];
            hidden_layer_output[j] = sigmoid(hidden_layer_output[j]);
        }

        float output = 0.0;
        for(int j = 0; j < hidden_size; j++) {
            output += w[j * output_size] * hidden_layer_output[j];
        }
        output += output_bias[0];
        output = sigmoid(output);

        float d;
        if(targets[i] == 1) {
            d = -1.0;
        }
        else {
            d = 1.0;
        }
        float fnet = sigmoid_derivative(output);
        total_error += 0.5 * (d - output) * (d - output);

        for(int j = 0; j < hidden_size; j++) {
            w[j * output_size] += lr * (d - output) * fnet * hidden_layer_output[j];
        }
        output_bias[0] += lr * (d - output) * fnet;

        for(int j = 0; j < hidden_size; j++) {
            for(int k = 0; k < input_dim; k++) {
                v[k * hidden_size + j] += lr * (d - output) * fnet * w[j * output_size] * (1.0 - hidden_layer_output[j]* hidden_layer_output[j]) * inputs[i * input_dim + k];
            }
            hidden_bias[j] += lr * (d - output) * fnet * w[j * output_size] * (1.0 - hidden_layer_output[j]* hidden_layer_output[j]);
        }
    }
    total_error /= num_samples;
    return total_error;
}

float multi_layer_delta_with_momentum(float* inputs, float* v, float*w, float* targets, float* hidden_bias, float* output_bias,
                                      double* prev_hidden_delta, double* prev_out_delta, double* prev_hidden_bias_delta, double* prev_out_bias_delta,
                                      int input_dim, int hidden_size, int output_size, int num_samples, float lr, float momentum)
{
    double total_error = 0.0;

    for(int i = 0; i < num_samples; i++) {
        std::vector<double> hidden_layer_output(hidden_size);

        for(int j = 0; j < hidden_size; j++) {
            hidden_layer_output[j] = 0.0;
            for(int k = 0; k < input_dim; k++) {
                hidden_layer_output[j] += inputs[i * input_dim + k] * v[k * hidden_size + j];
            }
            hidden_layer_output[j] += hidden_bias[j];
            hidden_layer_output[j] = sigmoid(hidden_layer_output[j]);
        }

        double output = 0.0;
        for(int j = 0; j < hidden_size; j++) {
            output += w[j * output_size] * hidden_layer_output[j];
        }
        output += output_bias[0];
        output = sigmoid(output);

        float d;
        if(targets[i] == 1.0) {
            d = -1.0;
        }
        else {
            d = 1.0;
        }
        double fnet = sigmoid_derivative(output);
        total_error += 0.5 * (d - output) * (d - output);

        for(int j = 0; j < hidden_size; j++) {
            double delta_w = lr * (d - output) * fnet * hidden_layer_output[j] + momentum * prev_out_delta[j * hidden_size];
            w[j * output_size] += delta_w;
            prev_out_delta[j * output_size] = delta_w;
        }
        output_bias[0] += lr * (d - output) * fnet + momentum * prev_out_bias_delta[0];
        prev_out_bias_delta[0] = lr * (d - output) * fnet;

        for(int j = 0; j < hidden_size; j++) {
            for(int k = 0; k < input_dim; k++) {
                double delta_v = lr * (d - output) * fnet * w[j * output_size] * (1.0 - hidden_layer_output[j] * hidden_layer_output[j]) * inputs[i * input_dim + k]
                                 + momentum * prev_hidden_delta[k * hidden_size + j];
                v[k * hidden_size + j] += delta_v;
                prev_hidden_delta[k * hidden_size + j] = delta_v;
            }
            double delta_bias = lr * (d - output) * fnet * w[j * output_size] * (1.0 - hidden_layer_output[j]* hidden_layer_output[j]) + momentum * prev_hidden_bias_delta[j];
            hidden_bias[j] += delta_bias;
            prev_hidden_bias_delta[j] = delta_bias;
        }
    }
    total_error /= num_samples;
    return total_error;
}

int test_forward(float *x, float *weight, float *bias, int num_class, int input_dim)
{
    int index_max;
    if (num_class > 2) {
        float* output = new float[num_class];
        // Calculation of the output layer input
        for (int i = 0; i < num_class; i++) {
            output[i] = 0.0f;
            for (int j = 0; j < input_dim; j++)
                output[i] += weight[i * input_dim + j] * x[j];
            output[i] += weight[2];
        }
        for (int i = 0; i < num_class; i++)
            output[i] = tanh(output[i]);

        // Softmax activation function
        //float sumExp = 0.0f;
        //for (int i = 0; i < num_Class; i++)
        //	sumExp += exp(output[i]);

        //for (int i = 0; i < num_Class; i++)
        //	output[i] = exp(output[i]) / sumExp;
        //Find Maximum in neuron
        float temp = output[0];
        index_max = 0;
        for (int i = 1; i < num_class; i++)
            if (temp < output[i]) {
                temp = output[i];
                index_max = i;
            }

        delete[] output;
    }
    else {
        float output = 0.0f;
        for (int j = 0; j < input_dim; j++)
            output += weight[j] * x[j];
        output += weight[2];
        output = tanh(output);
        if (output > 0.0f)
            index_max = 0;
        else index_max = 1;
    }
    return index_max;
}





int test_forward_2(float* x, float* v, float* w, float* b1, float* b2, int num_class, int input_dim, int hidden_dim)
{
    int index_Max = 0;

    // Giriþ katmaný Feedforward
    float* hidden_layer_output = new float[hidden_dim];
    for (int j = 0; j < hidden_dim; j++) {
        hidden_layer_output[j] = 0.0f;
        for (int k = 0; k < input_dim; k++) {
            hidden_layer_output[j] += v[k * hidden_dim + j] * x[k];
        }
        hidden_layer_output[j] += b1[j];
        hidden_layer_output[j] = tanh(hidden_layer_output[j]);
    }

    // Çýkýþ katmaný Feedforward
    float output = 0.0f;
    for (int j = 0; j < hidden_dim; j++) {
        output += w[j] * hidden_layer_output[j];
    }
    output += b2[0];
    output = tanh(output);

    if (output > 0.0f) {
        index_Max = 0;
    }
    else {
        index_Max = 1;
    }

    delete[] hidden_layer_output;

    return index_Max;
}




