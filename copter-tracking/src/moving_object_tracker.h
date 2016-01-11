#pragma once

#include <vector>
#include <opencv2/opencv.hpp>


class MovingObjectTracker
{
private:
    std::vector<cv::Point> object;

    bool isInOffsetLimits(std::vector<cv::Point> object, cv::Point lastPosition);

public:
    float maxOffsetX = 40;
    float maxOffsetY = 40;

    std::vector<cv::Point> getObject();
    bool getIsObjectDetected();

    MovingObjectTracker();

    void trackObject(std::vector<std::vector<cv::Point>> objects, cv::Point lastPosition);
};
