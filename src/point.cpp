#include "point.h"

Point::Point(){}

Point::Point(double x, double y, int label)
{
    this->x = x;
    this->y = y;
    this->label = label;
}

Point::~Point(){}
