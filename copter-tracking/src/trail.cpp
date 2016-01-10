#include "trail.h"

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


vector<cv::Point> Trail::getPoints()
{
    return vector<cv::Point>(Trail::points);
}

int Trail::getMaximumLength()
{
    return Trail::maximumLength;
}


Trail::Trail(int maximumLength)
{
    points = vector<Point>();
    this->maximumLength = maximumLength;
}


void Trail::addPoint(Point point)
{
    points.push_back(point);

    if (points.size() > Trail::maximumLength) {
        points.erase(points.begin(), points.begin() + 1);
    }
}
