#pragma once

#include "trail.h"

#include <string>
#include <opencv2/opencv.hpp>


class OpenCvHelper
{
private:
    static cv::Point calculateLinePosition(int lineIndex);

public:
    static void drawRectangle(cv::Mat &image, cv::Scalar color, cv::Rect rectangle);
    static void drawCrosshair(cv::Mat &image, cv::Scalar color, cv::Point point);
    static void drawText(cv::Mat &image, cv::Scalar color, cv::Point location, std::string text, int fontScale = 1, bool centerHorizontal = true);
    static void drawTrail(cv::Mat &image, cv::Scalar color, Trail trail);
    static void drawStreamInfo(cv::Mat &outputFrame, cv::VideoCapture stream);

    static cv::Point calculateRectangleCenter(cv::Rect rectangle);
    static float calculatePointDistance(cv::Point point1, cv::Point point2);
};
