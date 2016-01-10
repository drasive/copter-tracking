#include "trail.h"
#include "open_cv_helper.h"

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


void OpenCvHelper::drawRectangle(Mat &image, Scalar color, Rect rectangle)
{
    const int LINE_THICKNESS = 1;
    const int LINE_TYPE = 8;

    Point topLeft = Point(rectangle.x, rectangle.y);
    Point bottomRight = Point(rectangle.x + rectangle.width, rectangle.y + rectangle.height);

    cv::rectangle(image, topLeft, bottomRight, color, LINE_THICKNESS, LINE_TYPE);
}

void OpenCvHelper::drawCrosshair(Mat &target, Scalar color, Point location)
{
    const int CIRCLE_SIZE = 8;
    const int LINE_LENGTH = 12;
    const int LINE_THICKNESS = 1;
    const int LINE_TYPE = 8;

    // Draw circle
    circle(target, location, CIRCLE_SIZE, color, LINE_THICKNESS, LINE_TYPE);

    // Draw top line
    if (location.y - LINE_LENGTH > 0) {
        line(target, location, Point(location.x, location.y - LINE_LENGTH), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(target, location, Point(location.x, 0), color, LINE_THICKNESS, LINE_TYPE);
    }

    // Draw bottom line
    if (location.y + LINE_LENGTH < target.rows) {
        line(target, location, Point(location.x, location.y + LINE_LENGTH), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(target, location, Point(location.x, target.rows), color, LINE_THICKNESS, LINE_TYPE);
    }

    // Draw left line
    if (location.x - LINE_LENGTH > 0) {
        line(target, location, Point(location.x - LINE_LENGTH, location.y), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(target, location, Point(0, location.y), color, LINE_THICKNESS, LINE_TYPE);
    }

    // Draw right line
    if (location.x + LINE_LENGTH < target.cols) {
        line(target, location, Point(location.x + LINE_LENGTH, location.y), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(target, location, Point(target.cols, location.y), color, LINE_THICKNESS, LINE_TYPE);
    }
}

void OpenCvHelper::drawText(Mat &image, Scalar color, Point location, string text, int fontScale)
{
    const float TEXT_OFFSET_X_FACTOR = 9.0;
    const int FONT_FACE = 1;
    const int FONT_SCALE_FACTOR = 1;
    const int FONT_THICKNESS = 1;
    const int LINE_TYPE = 8;

    fontScale = fontScale * FONT_SCALE_FACTOR;
    putText(image, text, Point(location.x - (text.length() * fontScale / 2 * TEXT_OFFSET_X_FACTOR), location.y),
        FONT_FACE, fontScale, color, FONT_THICKNESS, LINE_TYPE);
}

void OpenCvHelper::drawTrail(cv::Mat &image, cv::Scalar color, Trail trail)
{
    const int MINIMUM_THICKNESS = 1;
    const int MAXIMUM_THICKNESS = 2;
    const int LINE_TYPE = 8;

    vector<Point> points = trail.getPoints();
    if (points.size() < 2) {
        return;
    }

    for (int pointIndex = 0; pointIndex < points.size() - 2; pointIndex++) {
        if (points[pointIndex].x == -1 || points[pointIndex + 1].x == -1) {
            break;
        }

        Point startPoint = Point(points[pointIndex].x, points[pointIndex].y);
        Point endPoint = Point(points[pointIndex + 1].x, points[pointIndex + 1].y);
        // TODO: Fix when framerate changes (pointIndex seems to be the problem)
        // TODO: Fix not getting thicker?
        int thickness = round(MINIMUM_THICKNESS + (MAXIMUM_THICKNESS - MINIMUM_THICKNESS) / (trail.getMaximumLength()) * pointIndex); // Linear progression

        line(image, startPoint, endPoint, color, thickness, LINE_TYPE);
    }
}


Point OpenCvHelper::calculateRectangleCenter(cv::Rect rectangle)
{
    return Point(
        rectangle.x + rectangle.width / 2,
        rectangle.y + rectangle.height / 2);
}

float OpenCvHelper::calculatePointDistance(cv::Point point1, cv::Point point2)
{
    int deltaX = point2.x - point1.x;
    int deltaY = point2.y - point1.y;

    return sqrt(deltaX*deltaX + deltaY*deltaY);
}
