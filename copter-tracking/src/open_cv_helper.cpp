#include "trail.h"
#include "open_cv_helper.h"

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


Point OpenCvHelper::calculateLinePosition(int lineIndex) {
    const int MARGIN_TOP = 30;
    const int MARGIN_LEFT = 5;
    const int LINE_HEIGHT = 30;

    return Point(MARGIN_LEFT, MARGIN_TOP + (LINE_HEIGHT * (lineIndex - 1)));
}


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

void OpenCvHelper::drawText(Mat &image, Scalar color, Point location, string text, int fontScale, bool centerHorizontal)
{
    const float TEXT_CENTER_HORIZONTAL_FACTOR = 9.0;
    const int FONT_FACE = 1;
    const int FONT_SCALE_FACTOR = 1;
    const int FONT_THICKNESS = 1;
    const int LINE_TYPE = 8;

    int positionX;
    if (centerHorizontal) {
        positionX = location.x - (text.length() * fontScale / 2 * TEXT_CENTER_HORIZONTAL_FACTOR);
    }
    else {
        positionX = location.x;
    }

    fontScale = fontScale * FONT_SCALE_FACTOR;
    putText(image, text, Point(positionX, location.y),
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

void OpenCvHelper::drawStreamInfo(Mat &outputFrame, VideoCapture stream) {
    const string TIME_FORMAT = "";
    const Scalar FONT_COLOR = Scalar(255, 255, 255);
    const int FONT_SCALE = 2;

    int currentFrame = (int)stream.get(CV_CAP_PROP_POS_FRAMES);
    int totalFrames = (int)stream.get(CV_CAP_PROP_FRAME_COUNT);
    // TODO: Format time
    int currentTime = currentFrame / stream.get(CV_CAP_PROP_FPS);
    int totalTime = totalFrames / stream.get(CV_CAP_PROP_FPS);

    OpenCvHelper::drawText(outputFrame, FONT_COLOR, calculateLinePosition(1),
        to_string(currentTime) + "/" + to_string(totalTime) + " (" + to_string(currentFrame) + "/" + to_string(totalFrames) + ")",
        FONT_SCALE, false);
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
