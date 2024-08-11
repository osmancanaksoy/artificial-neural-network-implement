#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include "point.h"
#include "process.h"
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void set_interface();
    float* add_data(float* sample, int size, float* x, int dim);
    float* add_labels(float* labels, int size, int label);
    void normalize();
    int YPoint(int x, float* weight, float* bias, int multiplier = 1);
    void draw_sample(float temp_x, float temp_y, int label);
    void draw_line(QPen pen, float x1, float y1, float x2, float y2);
    void add_samples(Point);
    void draw_samples();
    void draw_lines();
    void draw_lines_2();

private slots:
    // Get X and Y Values
    void clicked_graph(QMouseEvent* event);
    void move_mouse(QMouseEvent* event);

    // Set Class CheckBox
    void on_add_samples_clicked();

    //Single - Layer
    void on_actionRandom_triggered();
    void on_actionTrain_Delta_triggered();

    // Multi - Layer
    void on_actionRandom_2_triggered();
    void on_actionTrain_Delta_2_triggered();
    void on_actionTrain_Delta_with_Moment_triggered();

    // Test
    void on_actionTest_triggered();
    void on_actionTest_2_triggered();

    void on_actionRestart_triggered();

private:
    Ui::MainWindow *ui;
    int class_count = 0;
    int num_samples = 0;
    int input_dim = 2;
    int hidden_size;
    int output_size;

    //Single Layer
    float* weights = nullptr;
    float* bias = nullptr;
    //Two Layer
    float* hidden_layer_weights = nullptr;
    float* hidden_layer_bias = nullptr;
    float* output_layer_weights = nullptr;
    float* output_layer_bias = nullptr;
    // double* prev_hidden_delta = nullptr;
    // double* prev_out_delta = nullptr;
    // double* prev_hidden_bias_delta = nullptr;
    // double* prev_out_bias_delta = nullptr;


    float learning_rate;

    float* inputs = nullptr;
    float* targets = nullptr;
    float* mean = new float[2];
    float* std = new float[2];
    float mean_x, mean_y, std_x, std_y;
};
#endif // MAINWINDOW_H
