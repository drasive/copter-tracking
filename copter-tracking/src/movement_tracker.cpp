#include "trail.h"
#include "open_cv_helper.h"
#include "moving_object_detector.h"
#include "moving_object_tracker.h"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


// Global consts
//const string FILENAME = "";
const string FILENAME = "data/still_camera_1.mp4";
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const float TRAIL_DURATION = 1.5;

const string RAW_FRAME_WINDOW_NAME = "Raw Input";
const string DIFFERENCE_FRAME_WINDOW_NAME = "Raw Difference";
const string THRESHOLDED_FRAME_WINDOW_NAME = "Thresholded Difference";
const string BLURRED_FRAME_WINDOW_NAME = "Blurred Difference (Final)";
const string SETTINGS_WINDOW_NAME = "Settings";

// Settings
int framerate = 30;
int thresholdSensitivity = 30;
int blurSize = 20;

// Global variables
bool debugMode = false;
VideoCapture stream;

vector<Point> primaryObjectLastPosition;
int primaryObjectLastDetectedFrame = -1;
Trail primaryObjectTrail = Trail(framerate * TRAIL_DURATION);


void createNamedWindow(const string &name) {
    namedWindow(name, WINDOW_NORMAL);
}

void createSettingsWindow() {
    createNamedWindow(SETTINGS_WINDOW_NAME);

    createTrackbar("Threshold Sensitivity", SETTINGS_WINDOW_NAME, &thresholdSensitivity, 255);
    createTrackbar("Blur Sensitivity", SETTINGS_WINDOW_NAME, &blurSize, 255);

    createTrackbar("Framerate", SETTINGS_WINDOW_NAME, &framerate, 120);
    setTrackbarMin("Framerate", SETTINGS_WINDOW_NAME, 1);
}

void drawObjectMarkers(Mat &outputFrame, vector<vector<Point>> objects, MovingObjectTracker primaryObjectTracker) {
    const Scalar PRIMARY_OBJECT_DETECTED_MARKER_COLOR = Scalar(0, 255, 0);
    const Scalar PRIMARY_OBJECT_NOT_DETECTED_MARKER_COLOR = Scalar(0, 0, 255);
    const Scalar PRIMARY_OBJECT_DETECTED_TRAIL_COLOR = Scalar(0, 255, 0);
    const Scalar PRIMARY_OBJECT_NOT_DETECTED_TRAIL_COLOR = Scalar(0, 255, 0);
    const Scalar PRIMARY_OBJECT_DETECTED_OFFSET_LIMITS_COLOR = Scalar(0, 0, 0);
    const Scalar PRIMARY_OBJECT_NOT_DETECTED_OFFSET_LIMITS_COLOR = Scalar(0, 0, 0);
    const Scalar SECONDARY_OBJECTS_TRACKING_COLOR = Scalar(255, 255, 255);
    const bool DRAW_PRIMARY_OBJECT_TRAIL = true;
    const bool DRAW_PRIMARY_OBJECT_OFFSET_LIMITS = true;
    const bool DRAW_SECONDARY_OBJECT_MARKERS = true;
    const int TEXT_OFFSET_Y = 40;

    // Draw secondary object markers
    if (DRAW_SECONDARY_OBJECT_MARKERS) {
        for (auto &object : objects) {
            OpenCvHelper::drawRectangle(outputFrame, SECONDARY_OBJECTS_TRACKING_COLOR, boundingRect(object));
        }
    }

    // Draw primary object marker
    if (primaryObjectTracker.getIsObjectDetected()) {
        // Draw current position
        Rect boundingBox = boundingRect(primaryObjectTracker.getObject());
        Point objectCenter = OpenCvHelper::calculateRectangleCenter(boundingBox);

        OpenCvHelper::drawRectangle(outputFrame, PRIMARY_OBJECT_DETECTED_MARKER_COLOR, boundingBox);
        OpenCvHelper::drawCrosshair(outputFrame, PRIMARY_OBJECT_DETECTED_MARKER_COLOR, objectCenter);
    }
    else if (primaryObjectLastPosition.size() > 0) {
        // Draw last known position
        Rect boundingBox = boundingRect(primaryObjectLastPosition);
        Point objectCenter = OpenCvHelper::calculateRectangleCenter(boundingBox);

        OpenCvHelper::drawRectangle(outputFrame, PRIMARY_OBJECT_NOT_DETECTED_MARKER_COLOR, boundingBox);
        OpenCvHelper::drawCrosshair(outputFrame, PRIMARY_OBJECT_NOT_DETECTED_MARKER_COLOR, objectCenter);
        OpenCvHelper::drawText(outputFrame, PRIMARY_OBJECT_NOT_DETECTED_MARKER_COLOR,
            Point(objectCenter.x, objectCenter.y + TEXT_OFFSET_Y),
            "Object lost at (" + to_string(objectCenter.x) + "," + to_string(objectCenter.y) + ")");
    }

    // Draw primary object trail
    if (DRAW_PRIMARY_OBJECT_TRAIL) {
        Scalar trailColor;
        if (primaryObjectTracker.getIsObjectDetected()) {
            trailColor = PRIMARY_OBJECT_DETECTED_TRAIL_COLOR;
        }
        else {
            trailColor = PRIMARY_OBJECT_NOT_DETECTED_TRAIL_COLOR;
        }

        OpenCvHelper::drawTrail(outputFrame, trailColor, primaryObjectTrail);
    }

    // Draw primary object offset limits
    if (DRAW_PRIMARY_OBJECT_OFFSET_LIMITS) {
        Point objectCenter;
        Scalar offsetLimitColor;
        if (primaryObjectTracker.getIsObjectDetected()) {
            objectCenter = OpenCvHelper::calculateRectangleCenter(boundingRect(primaryObjectTracker.getObject()));
            offsetLimitColor = PRIMARY_OBJECT_DETECTED_OFFSET_LIMITS_COLOR;
        }
        else {
            objectCenter = OpenCvHelper::calculateRectangleCenter(boundingRect(primaryObjectLastPosition));
            offsetLimitColor = PRIMARY_OBJECT_NOT_DETECTED_OFFSET_LIMITS_COLOR;
        }

        OpenCvHelper::drawRectangle(outputFrame, offsetLimitColor, Rect(
            objectCenter.x - primaryObjectTracker.maxOffsetX,
            objectCenter.y - primaryObjectTracker.maxOffsetY,
            primaryObjectTracker.maxOffsetX * 2, primaryObjectTracker.maxOffsetY * 2));
    }
}


int main(int argc, char* argv[]) {
    const int MINIMUM_CONTOUR_AREA = 400;
    const int MAXIMUM_CONTOUR_AREA = 50000;
    const int MAXIMUM_OFFSET_X = 20;
    const int MAXIMUM_OFFSET_Y = 20;

    bool trackingEnabled = true;
    bool pause = false;

    Mat currentFrame, nextFrame;
    Mat currentFrameGrayscale, nextFrameGrayscale;
    Mat frameDifference;
    Mat frameDifferenceThresholded;
    Mat frameDifferenceBlurred;

    MovingObjectDetector objectDetector = MovingObjectDetector();
    objectDetector.filterByArea = true;
    objectDetector.minArea = MINIMUM_CONTOUR_AREA;
    objectDetector.maxArea = MAXIMUM_CONTOUR_AREA;

    MovingObjectTracker objectTracker = MovingObjectTracker();

    createSettingsWindow();

    while (true) {
        // Open video stream
        bool liveStream = FILENAME == "";

        if (liveStream) {
            stream.open(0);
        }
        else {
            stream.open(FILENAME);
        }

        if (!stream.isOpened()) {
            string streamSource;
            if (liveStream) {
                streamSource = "webcam";
            }
            else {
                streamSource = FILENAME;
            }

            cout << "Error acquiring video stream (" + streamSource + ")\n";
            return -1;
        }

        // Reset object tracking
        primaryObjectLastPosition = vector<Point>();
        primaryObjectLastDetectedFrame = -1;
        primaryObjectTrail = Trail(framerate * TRAIL_DURATION);



        // Analyze frames
        while (liveStream || stream.get(CV_CAP_PROP_POS_FRAMES) < stream.get(CV_CAP_PROP_FRAME_COUNT) - 1) {
            chrono::high_resolution_clock::time_point frameProcessingStartTime = chrono::high_resolution_clock::now();

            // Get current frame as grayscale
            stream.read(currentFrame);
            stream.read(nextFrame);

            // Convert to grayscale
            cvtColor(currentFrame, currentFrameGrayscale, COLOR_BGR2GRAY);
            cvtColor(nextFrame, nextFrameGrayscale, COLOR_BGR2GRAY);

            // Get difference between current and next frame
            absdiff(currentFrameGrayscale, nextFrameGrayscale, frameDifference);

            // Threshold the difference to filter out noise
            threshold(frameDifference, frameDifferenceThresholded, thresholdSensitivity, 255, THRESH_BINARY);

            // Blur the difference to filter out more noise
            if (blurSize > 0) {
                blur(frameDifferenceThresholded, frameDifferenceBlurred, cv::Size(blurSize, blurSize));
            }

            // Threshold the blur output to get binary image
            threshold(frameDifferenceBlurred, frameDifferenceBlurred, thresholdSensitivity, 255, THRESH_BINARY);


            // Detect objects
            vector<vector<Point>> objects = objectDetector.detect(frameDifferenceBlurred);

            // Track objects
            if (trackingEnabled && objects.size() > 0) {
                vector<Point> targetObject;
                if (objectTracker.getIsObjectDetected()) {
                    // Reuse the last object
                    targetObject = objectTracker.getObject();
                }
                else {
                    // TODO: Used too much
                    // Use the biggest object
                    targetObject = objects.at(objects.size() - 1);
                }
                Point targetObjectCenter = OpenCvHelper::calculateRectangleCenter(boundingRect(targetObject));

                int framesElapsedSinceLastDetection = max(stream.get(CAP_PROP_POS_FRAMES) - primaryObjectLastDetectedFrame, (double)1);
                objectTracker.maxOffsetX = MAXIMUM_OFFSET_X * framesElapsedSinceLastDetection;
                objectTracker.maxOffsetY = MAXIMUM_OFFSET_Y * framesElapsedSinceLastDetection;

                objectTracker.trackObject(objects, targetObjectCenter);
                if (objectTracker.getIsObjectDetected()) {
                    primaryObjectLastPosition = objectTracker.getObject();
                    primaryObjectLastDetectedFrame = stream.get(CAP_PROP_POS_FRAMES);

                    Point objectCenter = OpenCvHelper::calculateRectangleCenter(boundingRect(objectTracker.getObject()));
                    primaryObjectTrail.addPoint(objectCenter);
                }
            }

            // Draw
            OpenCvHelper::drawStreamInfo(currentFrame, stream);
            drawObjectMarkers(currentFrame, objects, objectTracker);
            
            // Calculate wait time
            auto frameProcessingDuration = chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now() - frameProcessingStartTime).count();

            // TODO: Reuse second frame as first frame next time
            int waitTime = 2 * 1000 / framerate - frameProcessingDuration;
            if (waitTime <= 0) {
                OpenCvHelper::drawText(currentFrame, Scalar(0, 0, 255), Point(FRAME_WIDTH / 2, 40),
                    "PROCESSING DELAY (" + to_string(-waitTime) + "ms)", 2);
                waitTime = 1;
            }

            // Display output
            createNamedWindow(RAW_FRAME_WINDOW_NAME);
            //resizeWindow(name, 1280, 960);
            imshow(RAW_FRAME_WINDOW_NAME, currentFrame);

            if (debugMode) {
                createNamedWindow(DIFFERENCE_FRAME_WINDOW_NAME);
                createNamedWindow(THRESHOLDED_FRAME_WINDOW_NAME);
                createNamedWindow(BLURRED_FRAME_WINDOW_NAME);

                imshow(DIFFERENCE_FRAME_WINDOW_NAME, frameDifference);
                imshow(THRESHOLDED_FRAME_WINDOW_NAME, frameDifferenceThresholded);
                imshow(BLURRED_FRAME_WINDOW_NAME, frameDifferenceBlurred);
            }
            else {
                destroyWindow(DIFFERENCE_FRAME_WINDOW_NAME);
                destroyWindow(THRESHOLDED_FRAME_WINDOW_NAME);
                destroyWindow(BLURRED_FRAME_WINDOW_NAME);
            }

            // Check for user input
            const int EXIT_KEY = 27; // "Esc" key
            const int TOGGLE_TRACKING_KEY = 't';
            const int TOGGLE_DEBUG_MODE_KEY = 'd';
            const int TOGGLE_PAUSE_KEY = 'p';

            switch (waitKey(waitTime)) {
            case EXIT_KEY:
                return 0;
            case TOGGLE_TRACKING_KEY:
                trackingEnabled = !trackingEnabled;
                if (trackingEnabled) {
                    cout << "Tracking enabled" << endl;
                }
                else {
                    cout << "Tracking disabled" << endl;
                }
                break;
            case TOGGLE_DEBUG_MODE_KEY:
                debugMode = !debugMode;
                if (debugMode) {
                    cout << "Debug mode enabled" << endl;
                }
                else {
                    cout << "Debug mode disabled" << endl;
                }
                break;
            case TOGGLE_PAUSE_KEY:
                pause = true;
                cout << "Playback paused" << endl;

                while (pause) {
                    switch (waitKey()) {
                    case TOGGLE_PAUSE_KEY:
                        pause = false;
                        cout << "Playback resumed" << endl;
                        break;
                    }
                }
            }
        }

        stream.release();
    }

    return 0;
}
