#ifndef POINT_H
#define POINT_H


class Point
{
public:
    Point();
    Point(double x,double y, int label);
    ~Point();
    double x;
    double y;
    int label;
};

#endif // POINT_H
