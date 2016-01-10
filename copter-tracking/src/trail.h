#pragma once

#include <vector>
#include <opencv2/opencv.hpp>


class Trail
{
private:
    std::vector<cv::Point> points;
    int maximumLength;

public:
    std::vector<cv::Point> getPoints();
    int getMaximumLength();

    Trail(int maximumLength);

    void addPoint(cv::Point point);
};
