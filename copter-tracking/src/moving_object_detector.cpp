#include "moving_object_detector.h"

#include <vector>
#include <opencv2/opencv.hpp>

#include "open_cv_helper.h"

using namespace std;
using namespace cv;


MovingObjectDetector::MovingObjectDetector()
{
    MovingObjectDetector::filterByArea = false;
    MovingObjectDetector::minArea = 0;
    MovingObjectDetector::maxArea = INT32_MAX;
}


vector<vector<Point>> MovingObjectDetector::detect(cv::Mat &frameDifference)
{
    // Detect contours
    Mat frameDifferenceCopy;
    frameDifference.copyTo(frameDifferenceCopy);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(frameDifferenceCopy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // Filter contours by area
    vector<vector<Point>> contoursFiltered;
    for (auto &contour : contours) {
        float contourArea = cv::contourArea(contour);
        if (!MovingObjectDetector::filterByArea ||
            (contourArea >= MovingObjectDetector::minArea && contourArea <= MovingObjectDetector::maxArea)) {
            contoursFiltered.push_back(contour);
        }
    }

    return contoursFiltered;
}
