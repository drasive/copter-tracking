#pragma once

#include <vector>
#include <opencv2/opencv.hpp>


class MovingObjectDetector
{
public:
    static std::vector<std::vector<cv::Point>> detect(cv::Mat &frameDifference);
};
