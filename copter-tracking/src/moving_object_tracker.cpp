#include "moving_object_tracker.h"
#include "open_cv_helper.h"

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


const Point NOT_DETECTED_POSITION = Point(-1, -1);

vector<Point> MovingObjectTracker::getObject()
{
    return MovingObjectTracker::object;
}

bool MovingObjectTracker::getIsObjectDetected()
{
    return MovingObjectTracker::object[0] != NOT_DETECTED_POSITION;
}


MovingObjectTracker::MovingObjectTracker()
{
    MovingObjectTracker::object = vector<Point>{ NOT_DETECTED_POSITION };
}

MovingObjectTracker::~MovingObjectTracker()
{
}


bool MovingObjectTracker::isInOffsetLimits(vector<Point> object, Point lastPosition) {
    Rect objectBoundingBox = boundingRect(object);
    Point objectCenter = OpenCvHelper::calculateRectangleCenter(objectBoundingBox);
    
    int offsetX = abs(objectCenter.x - lastPosition.x);
    int offsetY = abs(objectCenter.y - lastPosition.y);

    return offsetX <= MovingObjectTracker::maxOffsetX && offsetY <= MovingObjectTracker::maxOffsetY;
}


void MovingObjectTracker::trackObject(vector<vector<Point>> contours, Point lastPosition)
{
    const float MAXMIMUM_OFFSET_LIMIT_DURATION = 1.0;

    /*if (targetObject.size() < 1) {
        throw invalid_argument("targetObject has to be defined by at least one point");
    }*/

    if (contours.empty()) {
        MovingObjectTracker::object = vector<Point>{ NOT_DETECTED_POSITION };
        return;
    }

    // TODO: Test "always nearest object" strategy
    //vector<Point> biggestObject = contours.at(contours.size() - 1);
    //if (isInOffsetLimits(biggestObject, lastPosition)) {
    //    // Assume biggest object is primary object
    //    MovingObjectTracker::object = biggestObject;
    //}
    //else {
        // Assume object nearest to last position of primary object is still primary object
        cout << "Warning: Assuming primary object is nearest to last position (not largest)" << endl;

        //Point primaryObjectLastCenter = OpenCvHelper::calculateRectangleCenter(primaryObjectLastBoundingRectangle);
        float minimumDistance = UINT32_MAX;
        vector<Point> nearestObject;
        for (auto &contour : contours) {
            Point contourCenter = OpenCvHelper::calculateRectangleCenter(boundingRect(contour));

            if (OpenCvHelper::calculatePointDistance(contourCenter, lastPosition) < minimumDistance) {
                minimumDistance = OpenCvHelper::calculatePointDistance(contourCenter, lastPosition);
                nearestObject = contour;
            }
        }

        // Make sure nearest object doesn't violate maximum offset limits
        Rect nearestObjectBoundingRectangle = boundingRect(nearestObject);
        //int framesElapsedSinceLastDetection = max(stream.get(CAP_PROP_POS_FRAMES) - primaryObjectLastDetectedFrame, (double)1);
        if (isInOffsetLimits(nearestObject, lastPosition) /* || framesElapsedSinceLastDetection > framerate * MAXMIMUM_OFFSET_LIMIT_DURATION*/) {
            MovingObjectTracker::object = nearestObject;
            return;
        }

        MovingObjectTracker::object = vector<Point>{ NOT_DETECTED_POSITION };
    //}
}
