#include "moving_object_detector.h"

#include <vector>
#include <opencv2/opencv.hpp>

#include "open_cv_helper.h"

using namespace std;
using namespace cv;


vector<vector<Point>> MovingObjectDetector::detect(cv::Mat &frameDifference)
{
    // Using contour detection
    // TODO: Add and test PROCESSING_RESOLUTION_FACTOR. Outsource into class (downscaling image, calling processing method, upscaling result)
    Mat frameDifferenceCopy;
    frameDifference.copyTo(frameDifferenceCopy);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(frameDifferenceCopy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // TODO: Filter contours by minimum and maximum area
    // TODO: Filter contours by minimum and maximum height/width relation

    return contours;
}
