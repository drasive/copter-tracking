#pragma once

#include <vector>
#include <opencv2/opencv.hpp>


class MovingObjectDetector
{
public:
    bool filterByArea;
    int minArea;
    int maxArea;

    MovingObjectDetector();

    std::vector<std::vector<cv::Point>> detect(cv::Mat &frameDifference);
};
