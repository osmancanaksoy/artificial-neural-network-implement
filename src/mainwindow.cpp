#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QDebug>
#include <QGraphicsEllipseItem>

QVector<QColor> myColors = {QColor(128,0,0),QColor(170,110,0),QColor(0,0,128),
                            QColor(128,128,0),QColor(230, 25, 75),QColor(128, 128, 128),
                            QColor(0, 0, 0),QColor(0, 128, 128),QColor(255,0,255),
                            QColor(255,0,127),QColor(128,128,128),QColor(204,255,229)};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->customPlot, SIGNAL(mousePress(QMouseEvent*)), SLOT(clicked_graph(QMouseEvent*)));
    connect(ui->customPlot, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(move_mouse(QMouseEvent*)));
    set_interface();
}

MainWindow::~MainWindow()
{
    delete[] weights;    
    delete[] inputs;
    delete[] targets;
    delete[] bias;
    delete[] hidden_layer_weights;
    delete[] hidden_layer_bias;
    delete[] output_layer_weights;
    delete[] output_layer_bias;
    delete[] mean;
    delete[] std;

    delete ui;
}

void MainWindow::set_interface()
{
    //Set Main Plot
    QPen pen;
    pen.setWidth(3);
    ui->customPlot->xAxis->grid()->setPen(Qt::NoPen);
    ui->customPlot->xAxis->grid()->setZeroLinePen(pen);

    ui->customPlot->yAxis->grid()->setPen(Qt::NoPen);
    ui->customPlot->yAxis->grid()->setZeroLinePen(pen);


    ui->customPlot->xAxis->setRange(-25, 25);
    ui->customPlot->yAxis->setRange(-25, 25);
    //ui->customPlot->setInteractions(QCP::iRangeZoom | QCP::iRangeDrag);
    ui->customPlot->addGraph();

    // Set Error Plot
    ui->error_plot->addGraph();
    ui->error_plot->graph(0)->setLineStyle(QCPGraph::lsLine);
    ui->error_plot->graph(0)->setPen(QPen(myColors[0]));

    //Set Default Values
    ui->class_lineEdit->setText("2");
    ui->epoch_value->setText("10000");
    ui->learning_value->setText("0.1");
    ui->max_error_value->setText("0.01");
    ui->hidden_value->setText("2");
    ui->output_value->setText("1");
}

float *MainWindow::add_data(float *sample, int size, float *x, int dim)
{
    float* temp;
    temp = new float[size * dim];
    for (int i = 0; i < (size - 1) * dim; i++)
        temp[i] = sample[i];
    for (int i = 0; i < dim; i++)
        temp[(size - 1) * dim + i] = x[i];
    //Deallocate memory
    delete[] sample;
    return temp;
}

float *MainWindow::add_labels(float *labels, int size, int label)
{
    float* temp;
    temp = new float[size];
    for (int i = 0; i < size - 1; i++)
        temp[i] = labels[i];
    temp[size - 1] = float(label);
    //Deallocate memory
    delete[] labels;
    return temp;
}

void MainWindow::draw_sample(float temp_x, float temp_y, int label)
{
    QPen pen;
    pen.setWidth(2);
    pen.setColor(myColors[label]);

    QCPItemLine *line = new QCPItemLine(ui->customPlot);
    line->setPen(pen);
    line->start->setCoords(temp_x - 0.75, temp_y);
    line->end->setCoords(temp_x + 0.75, temp_y);

    QCPItemLine *line_2 = new QCPItemLine(ui->customPlot);
    line_2->setPen(pen);
    line_2->start->setCoords(temp_x, temp_y - 0.75);
    line_2->end->setCoords(temp_x, temp_y + 0.75);

    ui->customPlot->replot();
}

void MainWindow::draw_line(QPen pen, float x1, float y1, float x2, float y2)
{
    QCPItemStraightLine* line = new QCPItemStraightLine(ui->customPlot);
    line->setPen(pen);
    line->point1->setCoords(x1, y1);
    line->point2->setCoords(x2, y2);

    ui->customPlot->replot();
}

// void MainWindow::add_samples(Point p)
// {

//     if(p.label == -1){
//         return;
//     }
//     else if(p.label == 0) {
//         return;
//     }
//     else {
//         QPen pen;
//         pen.setWidth(1);
//         pen.setColor(myColors[p.label]);

//         QCPItemLine *line = new QCPItemLine(ui->customPlot);
//         line->setPen(pen);
//         line->start->setCoords(p.x - 0.2, p.y);
//         line->end->setCoords(p.x + 0.2, p.y);

//         QCPItemLine *line_2 = new QCPItemLine(ui->customPlot);
//         line_2->setPen(pen);
//         line_2->start->setCoords(p.x, p.y - 0.2);
//         line_2->end->setCoords(p.x, p.y + 0.2);

//         ui->customPlot->replot();



//         Point* temp;
//         temp = samples;
//         num_samples++;
//         samples = new Point[num_samples];

//         for(int i = 0; i < num_samples - 1; i++) {
//             samples[i].label = temp[i].label;
//             samples[i].x = temp[i].x;
//             samples[i].y = temp[i].y;
//         }

//         samples[num_samples - 1].x = p.x;
//         samples[num_samples - 1].y = p.y;
//         samples[num_samples - 1].label = p.label;

//         targets = new Point [num_samples];
//         for(int i = 0; i < num_samples; i++) {
//             targets[i] = samples[i];
//         }
//     }
// }

void MainWindow::draw_samples()
{
    for(int i = 0; i < num_samples; i++) {
        QPen pen;
        pen.setWidth(2);
        pen.setColor(myColors[targets[i]]);

        float temp_x = inputs[i * input_dim];
        float temp_y = inputs[i * input_dim + 1];

        QCPItemLine *line = new QCPItemLine(ui->customPlot);
        line->setPen(pen);
        line->start->setCoords(temp_x - 0.75, temp_y);
        line->end->setCoords(temp_x + 0.75, temp_y);

        QCPItemLine *line_2 = new QCPItemLine(ui->customPlot);
        line_2->setPen(pen);
        line_2->start->setCoords(temp_x, temp_y - 0.75);
        line_2->end->setCoords(temp_x, temp_y + 0.75);

        ui->customPlot->replot();
    }
}

void MainWindow::draw_lines()
{
    if(class_count > 2) {
        for(int i = 0; i < class_count; i++) {
            QPen pen;
            pen.setWidth(1);
            pen.setColor(myColors[i+1]);
            int min_x, max_x, min_y, max_y;

            min_x = (ui->customPlot->width()) / -2;
            min_y = YPoint(min_x, &weights[i * 2], &bias[i]);
            max_x = (ui->customPlot->width()) / 2;
            max_y = YPoint(max_x, &weights[i * 2], &bias[i]);

            draw_line(pen, min_x, min_y, max_x, max_y);
        }
    }
    else {
        QPen pen;
        pen.setWidth(1);
        pen.setColor(myColors[0]);
        int min_x, max_x, min_y, max_y;

        min_x = (ui->customPlot->width()) / -2;
        min_y = YPoint(min_x, weights, bias);
        max_x = (ui->customPlot->width()) / 2;
        max_y = YPoint(max_x, weights, bias);

        draw_line(pen, min_x, min_y, max_x, max_y);
    }
}

void MainWindow::draw_lines_2()
{
    int hidden_layer = 2;
    for(int i = 0; i < hidden_layer; i++) {
        QPen pen;
        pen.setWidth(1);
        pen.setColor(myColors[i+1]);
        int min_x, max_x, min_y, max_y;

        min_x = (ui->customPlot->width()) / -2;
        min_y = YPoint(min_x, &hidden_layer_weights[i * 2], &hidden_layer_bias[i]);
        max_x = (ui->customPlot->width()) / 2;
        max_y = YPoint(max_x, &hidden_layer_weights[i * 2], &hidden_layer_bias[i]);

        draw_line(pen, min_x, min_y, max_x, max_y);
    }
}

void MainWindow::clicked_graph(QMouseEvent *event)
{
    QPoint point = event->pos();
    // Point sample(ui->customPlot->xAxis->pixelToCoord(point.x()),
    //               ui->customPlot->yAxis->pixelToCoord(point.y()),ui->label_comboBox->currentIndex());

    // add_samples(sample);

    if( class_count == 0)
        qDebug() << "The Network Architeture should be firtly set up";
    else {
        float* x= new float[input_dim];
        float temp_x = ui->customPlot->xAxis->pixelToCoord(point.x());
        float temp_y = ui->customPlot->yAxis->pixelToCoord(point.y());
        x[0] = temp_x;
        x[1] = temp_y;
        int label;
        int numLabel = ui->label_comboBox->currentIndex();
        if (numLabel > class_count)
            qDebug() << "The class label cannot be greater than the maximum number of classes.";
        else {
            label = numLabel;
            if (num_samples == 0) {
                num_samples = 1;
                inputs = new float[num_samples * input_dim]; targets = new float[num_samples];
                for (int i = 0; i < input_dim; i++)
                    inputs[i] = x[i];
                targets[0] = float(label);
            }
            else {
                num_samples++;
                inputs = add_data(inputs, num_samples, x, input_dim);
                targets = add_labels(targets, num_samples, label);
            }
            draw_sample(temp_x, temp_y, label);
            delete [] x;
        }
    }
}

void MainWindow::move_mouse(QMouseEvent* event) {
    QPoint mouse_position = event->pos();

    float x = ui->customPlot->xAxis->pixelToCoord(mouse_position.x());
    float y = ui->customPlot->yAxis->pixelToCoord(mouse_position.y());

    ui->pos_label->setText("X:" + QString::number(x) + " Y:" + QString::number(y));
}


void MainWindow::on_add_samples_clicked()
{
    class_count = (ui->class_lineEdit->text()).toInt();

    for(int i = 1; i <= class_count; i++) {
        ui->label_comboBox->addItem(QString::number(i));
    }
}

void MainWindow::normalize()
{
    float mean_x = 0;
    float mean_y = 0;

    for (int i = 0; i < num_samples; i++) {
        mean_x += inputs[i * input_dim];
        mean_y += inputs[i * input_dim + 1];
    }

    mean_x /= num_samples;
    mean_y /= num_samples;

    //calculating variance
    float variance_x = 0;
    float variance_y = 0;

    for (int i = 0; i < num_samples; i++) {
        variance_x += pow(inputs[i * input_dim] - mean_x, 2);
        variance_y += pow(inputs[i * input_dim + 1] - mean_y, 2);
    }

    variance_x = sqrt(variance_x / num_samples);
    variance_y = sqrt(variance_y / num_samples);



    for (int i = 0; i < num_samples; i++) {
        inputs[i * input_dim] = (inputs[i * input_dim] - mean_x) / variance_x;
        inputs[i * input_dim + 1] = (inputs[i * input_dim + 1] - mean_y) / variance_y;
    }
    this->mean_x = mean_x;
    this->mean_y = mean_y;
    this->std_x = variance_x;
    this->std_y = variance_y;
}

int MainWindow::YPoint(int x, float *weight, float* bias,int multiplier)
{
    return (int)((float)(-1 * (float)multiplier * bias[0] - weight[0] * x) / (float)(weight[1]));
}

void MainWindow::on_actionRandom_triggered()
{
    if(class_count > 2) {
        weights = init_array_random(class_count * input_dim);
        bias = init_array_random(class_count);
    }
    else {
        int num_out_neuron = 1;
        weights = init_array_random(input_dim + num_out_neuron);
        bias = init_array_random(num_out_neuron);
    }
    ui->w1_value->setText(QString::number(weights[0]));
    ui->w2_value->setText(QString::number(weights[1]));
    ui->w3_value->setText(QString::number(weights[2]));

    ui->customPlot->clearItems();
    ui->customPlot->replot();
    draw_lines(); draw_samples();
}

void MainWindow::on_actionTrain_Delta_triggered()
{
    if(ui->normalize_check_box->isChecked() == true) {
        z_score_parameters(inputs, num_samples, input_dim, mean, std);
        for (int step = 0; step < num_samples; step++)
        {
            for (int n = 0; n < input_dim; n++)
            {
                inputs[step * input_dim + n] = (inputs[step * input_dim + n] - mean[n]) / std[n];
            }
        }
    }

    long int Epoch = ui->epoch_value->text().toInt();
    float lr = ui->learning_value->text().toFloat();
    float max_error = ui->max_error_value->text().toFloat();

    ui->error_plot->xAxis->setRange(0,Epoch);
    ui->error_plot->yAxis->setRange(0,0.5);
    int cycles = 0;
    float error;

    if(class_count > 2) {

        std::vector<float> targetsss(num_samples * class_count, 0.0);

        for (int step = 0; step < num_samples; step++)
        {
            int classD = targets[step];
            // Çoklu sýnýflar için one-hot encoding kullanarak hedefleri güncelle
            for (int classIndex = 0; classIndex < class_count; classIndex++)
            {
                if ((classIndex + 1) == classD)
                    targetsss[step * class_count + classIndex] = 1.0;
                else
                    targetsss[step * class_count + classIndex] = -1.0;
            }
        }
        while(cycles < Epoch) {
            error = 0.0f;
            error += multi_class_delta_rule(inputs, weights, targetsss, bias,lr, num_samples, input_dim, class_count);

            cycles++;

            ui->error_plot->xAxis->setRange(1,cycles);
            ui->error_plot->graph()->addData(cycles, error);
            ui->error_plot->replot();
            if ((error <= max_error) || (cycles >= Epoch)) {
                ui->error_value->setText(QString::number(error));
                break;
            }
        }
        ui->cycles_value->setText(QString::number(cycles));
        ui->customPlot->clearItems();
        ui->customPlot->replot();
        draw_lines();
        draw_samples();
    }
    else {
        while(cycles < Epoch) {
            error = 0.0f;
            error += single_neuron_delta_rule(inputs, weights, targets, bias, lr, num_samples, input_dim);

            cycles++;

            ui->error_plot->xAxis->setRange(1,cycles);
            ui->error_plot->graph()->addData(cycles, error);
            ui->error_plot->replot();
            if ((error <= max_error) || (cycles >= Epoch)) {
                ui->error_value->setText(QString::number(error));
                break;
            }
        }
        ui->cycles_value->setText(QString::number(cycles));
        ui->customPlot->clearItems();
        ui->customPlot->replot();
        draw_lines();
        draw_samples();
    }

    if(ui->normalize_check_box->isChecked() == true) {
        for (int i = 0; i < num_samples; i++)
        {
            for (int k = 0; k < input_dim; k++)
            {
                inputs[i * input_dim + k] = inputs[i * input_dim + k] * 10;
            }
        }
    }

}

void MainWindow::on_actionRandom_2_triggered()
{
    hidden_size = ui->hidden_value->text().toInt();
    output_size = ui->output_value->text().toInt();

    hidden_layer_weights = init_array_random(input_dim * hidden_size);
    hidden_layer_bias = init_array_random(hidden_size);

    output_layer_weights = init_array_random(hidden_size * output_size);
    output_layer_bias = init_array_random(output_size);

    ui->customPlot->clearItems();
    ui->customPlot->clearPlottables();
    ui->customPlot->replot();
    draw_lines_2(); draw_samples();
}


void MainWindow::on_actionTrain_Delta_2_triggered()
{
    int Epoch = ui->epoch_value->text().toInt();
    float lr = ui->learning_value->text().toFloat();
    float max_error = ui->max_error_value->text().toFloat();

    ui->error_plot->xAxis->setRange(0,Epoch);
    ui->error_plot->yAxis->setRange(0,1);
    int cycles = 0;
    float error;

    if(ui->normalize_check_box->isChecked() == true) {
        z_score_parameters(inputs, num_samples, input_dim, mean, std);
        for (int step = 0; step < num_samples; step++)
        {
            for (int n = 0; n < input_dim; n++)
            {
                inputs[step * input_dim + n] = (inputs[step * input_dim + n] - mean[n]) / std[n];
            }
        }
    }

    while(cycles < Epoch) {
        error = 0.0f;
        error += multi_layer_delta_rule(inputs, hidden_layer_weights, output_layer_weights,targets, hidden_layer_bias, output_layer_bias, input_dim, hidden_size, output_size, num_samples, lr);

        cycles++;

        ui->error_plot->xAxis->setRange(1,cycles);
        ui->error_plot->graph()->addData(cycles, error);
        ui->error_plot->replot();
        this->update();
        if ((error <= max_error) || (cycles >= Epoch)) {
            ui->error_value->setText(QString::number(error));
            break;
        }
    }
    ui->cycles_value->setText(QString::number(cycles));
    ui->customPlot->clearItems();
    ui->customPlot->replot();
    draw_samples();
    draw_lines_2();
    for (int i = 0; i < num_samples; i++)
    {
        for (int k = 0; k < input_dim; k++)
        {
            inputs[i * input_dim + k] = inputs[i * input_dim + k] * 10;
        }
    }
}


void MainWindow::on_actionTrain_Delta_with_Moment_triggered()
{
    int Epoch = ui->epoch_value->text().toInt();
    float lr = ui->learning_value->text().toFloat();
    float max_error = ui->max_error_value->text().toFloat();

    ui->error_plot->xAxis->setRange(0,Epoch);
    ui->error_plot->yAxis->setRange(0,1);
    int cycles = 0;
    double error;

    if(ui->normalize_check_box->isChecked() == true) {
        z_score_parameters(inputs, num_samples, input_dim, mean, std);
        for (int step = 0; step < num_samples; step++)
        {
            for (int n = 0; n < input_dim; n++)
            {
                inputs[step * input_dim + n] = (inputs[step * input_dim + n] - mean[n]) / std[n];
            }
        }
    }

    double* prev_hidden_delta = new double[input_dim * hidden_size]();
    double* prev_out_delta = new double[hidden_size * output_size]();
    double* prev_hidden_bias_delta = new double [hidden_size]();
    double* prev_out_bias_delta = new double[output_size]();

    while(cycles < Epoch) {
        error = 0.0;

        for (int step = 0; step < num_samples; step++) {
            // FeedForward
            double* hiddenLayerOutput = new double[hidden_size];
            for (int j = 0; j < hidden_size; j++) {
                hiddenLayerOutput[j] = 0.0;
                for (int k = 0; k < input_dim; k++) {
                    hiddenLayerOutput[j] += hidden_layer_weights[k * hidden_size + j] * inputs[step * input_dim + k];
                }
                hiddenLayerOutput[j] += hidden_layer_bias[j];
                hiddenLayerOutput[j] = 2.0 / (1.0 + exp(-hiddenLayerOutput[j])) - 1.0;
            }

            double output = 0.0;
            for (int j = 0; j < hidden_size; j++) {
                output += output_layer_weights[j * output_size] * hiddenLayerOutput[j];
            }
            output += output_layer_bias[0];
            output = sigmoid(output);


            // Backpropagation
            float d;
            if(targets[step] == 1) {
                d = 1;
            }
            else {
                d = -1;
            }
            double fnet = sigmoid_derivative(output);
            error += 0.5 * pow((d - output), 2);

            //Output Aðýrlýk Deðerleri Güncelle (moment ile)
            for (int j = 0; j < hidden_size; j++) {
                double delta = lr * (d - output) * fnet * hiddenLayerOutput[j] + 0.9 * prev_out_delta[j * input_dim];
                output_layer_weights[j * output_size] += delta;
                prev_out_delta[j * output_size] = delta;
            }
            output_layer_bias[0] += lr * (d - output) * fnet + 0.9 * prev_out_bias_delta[0];
            prev_out_bias_delta[0] = lr * (d - output) * fnet;

            //Giriþ Katman Aðýrlýk Deðerlerini Güncelle (moment ile)
            for (int j = 0; j < hidden_size; j++) {
                for (int k = 0; k < input_dim; k++) {
                    double delta = lr * (d - output) * fnet * output_layer_weights[j * output_size] * (1.0 - hiddenLayerOutput[j] * hiddenLayerOutput[j]) * inputs[step * input_dim + k] + 0.9 * prev_hidden_delta[k * hidden_size + j];
                    hidden_layer_weights[k * hidden_size + j] += delta;
                    prev_hidden_delta[k * hidden_size + j] = delta;
                }
                double deltaBias = lr * (d - output) * fnet * output_layer_weights[j * output_size] * (1.0 - hiddenLayerOutput[j] * hiddenLayerOutput[j]) + 0.9 * prev_hidden_bias_delta[j];
                hidden_layer_bias[j] += deltaBias;
                prev_hidden_bias_delta[j] = deltaBias;
            }

            delete[] hiddenLayerOutput;
        }
        cycles++;
        error /= num_samples;
        //qDebug() << error;

        ui->error_plot->xAxis->setRange(1,cycles);
        ui->error_plot->graph()->addData(cycles, error);
        ui->error_plot->replot();
        this->update();
        if ((error <= max_error) || (cycles >= Epoch)) {
            ui->error_value->setText(QString::number(error));
            break;
        }
    }

    ui->cycles_value->setText(QString::number(cycles));
    ui->customPlot->clearItems();
    ui->customPlot->replot();
    draw_samples();
    draw_lines_2();
    for (int i = 0; i < num_samples; i++)
    {
        for (int k = 0; k < input_dim; k++)
        {
            inputs[i * input_dim + k] = inputs[i * input_dim + k] * 20;
        }
    }
    delete[] prev_hidden_delta;
    delete[] prev_out_delta;
    delete[] prev_hidden_bias_delta;
    delete[] prev_out_bias_delta;

}

void MainWindow::on_actionTest_triggered()
{
    z_score_parameters(inputs, num_samples, input_dim, mean, std);
    ui->customPlot->clearItems();
    ui->customPlot->replot();
    QPixmap pix(ui->customPlot->width(), ui->customPlot->width());
    QImage image = pix.toImage();

    float* x = new float[2];
    int num;
    QColor c;
    for(int row = ui->customPlot->height() - 1; row > 0; row-=2) {
        for(int col = ui->customPlot->width() - 1; col > 0; col-=2) {

            x[0] = (float)(col - (ui->customPlot->width() / 2));
            //x[0] = (x[0] - mean_x) / std_x;
            x[0] = (x[0] - mean[0]) / std[0];
            x[1] = (float)((ui->customPlot->height() / 2) - row);
            //x[1] = (x[1] - mean_y) / std_y;
            x[1] = (x[1] - mean[1]) / std[1];

            num = test_forward(x, weights, bias, class_count, input_dim);
            switch (num) {
            case 0: c = QColor(0, 0, 0); break;
            case 1: c = QColor(255, 0, 0); break;
            case 2: c = QColor(0, 0, 255); break;
            case 3: c = QColor(0, 255, 0); break;
            case 4: c = QColor(255, 255, 0); break;
            case 5: c = QColor(255, 165, 0); break;
            default: c = QColor(0, 255, 255);
            }

            image.setPixelColor(col,row, c);
        }
    }

    pix = QPixmap::fromImage(image);
    ui->customPlot->setBackground(pix, false, Qt::KeepAspectRatio);
    draw_samples();
    delete [] x;
}

void MainWindow::on_actionTest_2_triggered()
{
    z_score_parameters(inputs, num_samples, input_dim, mean, std);
    ui->customPlot->clearItems();
    ui->customPlot->replot();
    QPixmap pix(ui->customPlot->width(), ui->customPlot->width());
    QImage image = pix.toImage();

    float* x = new float[2];
    int num;
    QColor c;
    for(int row = 0; row < ui->customPlot->height(); row+=2) {
        for(int col = 0; col < ui->customPlot->width(); col+=2) {

            x[0] = (float)(col - (ui->customPlot->width() / 2));
            //x[0] = (x[0] - mean_x) / std_x;
            x[0] = (x[0] - mean[0]) / std[0];
            x[1] = (float)((ui->customPlot->height() / 2) - row);
            //x[1] = (x[1] - mean_y) / std_y;
            x[1] = (x[1] - mean[1]) / std[1];

            num = test_forward_2(x, hidden_layer_weights, output_layer_weights, hidden_layer_bias, output_layer_bias, class_count, input_dim, hidden_size);
            switch (num) {
            case 0: c = QColor(0, 0, 0); break;
            case 1: c = QColor(255, 0, 0); break;
            case 2: c = QColor(0, 0, 255); break;
            case 3: c = QColor(0, 255, 0); break;
            case 4: c = QColor(255, 255, 0); break;
            case 5: c = QColor(255, 165, 0); break;
            default: c = QColor(0, 255, 255);
            }

            image.setPixelColor(col,row, c);
        }
    }

    pix = QPixmap::fromImage(image);
    ui->customPlot->setBackground(pix, false, Qt::KeepAspectRatio);
    draw_samples();
    delete [] x;
}


void MainWindow::on_actionRestart_triggered()
{
    qApp->quit();
    QProcess::startDetached(qApp->arguments()[0], qApp->arguments());
}

