#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

// Global consts
//const string FILENAME = "";
const string FILENAME = "input/still_camera_1.mp4";
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const float TRAIL_DURATION = 0.8;

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

int primaryObjectLastPosition[2] = { -1, -1 };
Rect primaryObjectLastBoundingRectangle = Rect(-1, -1, -1, -1);
int primaryObjectLastDetectedFrame = -1;
vector<Point> trail = vector<Point>();


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

void drawCrosshair(Mat &frame, Scalar color, int x, int y) {
    const int CIRCLE_SIZE = 8;
    const int LINE_LENGTH = 12;
    const int LINE_THICKNESS = 1;
    const int LINE_TYPE = 8;

    // Draw circle
    circle(frame, Point(x, y), CIRCLE_SIZE, color, LINE_THICKNESS, LINE_TYPE);

    // Draw top line
    if (y - LINE_LENGTH > 0) {
        line(frame, Point(x, y), Point(x, y - LINE_LENGTH), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(frame, Point(x, y), Point(x, 0), color, LINE_THICKNESS, LINE_TYPE);
    }
    // Draw bottom line
    if (y + LINE_LENGTH < FRAME_HEIGHT) {
        line(frame, Point(x, y), Point(x, y + LINE_LENGTH), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(frame, Point(x, y), Point(x, FRAME_HEIGHT), color, LINE_THICKNESS, LINE_TYPE);
    }

    // Draw left line
    if (x - LINE_LENGTH > 0) {
        line(frame, Point(x, y), Point(x - LINE_LENGTH, y), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(frame, Point(x, y), Point(0, y), color, LINE_THICKNESS, LINE_TYPE);
    }

    // Draw right line
    if (x + LINE_LENGTH < FRAME_WIDTH) {
        line(frame, Point(x, y), Point(x + LINE_LENGTH, y), color, LINE_THICKNESS, LINE_TYPE);
    }
    else {
        line(frame, Point(x, y), Point(FRAME_WIDTH, y), color, LINE_THICKNESS, LINE_TYPE);
    }
}

void drawRectangle(Mat &frame, Scalar color, Rect rectangle) {
    const int LINE_THICKNESS = 1;
    const int LINE_TYPE = 8;

    Point topLeft = Point(rectangle.x, rectangle.y);
    Point bottomRight = Point(rectangle.x + rectangle.width, rectangle.y + rectangle.height);

    cv::rectangle(frame, topLeft, bottomRight, color, LINE_THICKNESS, LINE_TYPE);
}

void drawText(Mat &frame, Scalar color, int x, int y, string text, int fontScale = 1) {
    const float TEXT_OFFSET_X_FACTOR = 9.0;
    const int FONT_FACE = 1;
    const int FONT_SCALE_FACTOR = 1;
    const int FONT_THICKNESS = 1;
    const int LINE_TYPE = 8;

    fontScale = fontScale * FONT_SCALE_FACTOR;
    putText(frame, text, Point(x - (text.length() * fontScale / 2 * TEXT_OFFSET_X_FACTOR), y),
        FONT_FACE, fontScale, color, FONT_THICKNESS, LINE_TYPE);
}

void drawTrail(Mat &frame, Scalar color) {
    const int MINIMUM_THICKNESS = 1;
    const int MAXIMUM_THICKNESS = 2;
    const int LINE_TYPE = 8;

    if (trail.size() < 2) {
        return;
    }

    for (int pointIndex = 0; pointIndex < trail.size() - 2; pointIndex++) {
        if (trail[pointIndex].x == -1 || trail[pointIndex + 1].x == -1) {
            break;
        }

        Point startingPoint = Point(trail[pointIndex].x, trail[pointIndex].y);
        Point endPoint = Point(trail[pointIndex + 1].x, trail[pointIndex + 1].y);
        float maximumTrailLength = framerate * TRAIL_DURATION;
        // TODO: Fix when framerate changes (pointIndex seems to be the problem)
        int thickness = round(MINIMUM_THICKNESS + (MAXIMUM_THICKNESS - MINIMUM_THICKNESS) / (maximumTrailLength) * pointIndex); // Linear progression

        line(frame, startingPoint, endPoint, color, thickness, LINE_TYPE);
    }
}

void addTrailPoint(Point point) {
    trail.push_back(point);

    int maximumTrailLength = framerate * TRAIL_DURATION;
    if (trail.size() > maximumTrailLength) {
        trail.erase(trail.begin(), trail.begin() + 1);
    }
}

Point calculateRectangleCenter(Rect rectangle) {
    return Point(
        rectangle.x + rectangle.width / 2,
        rectangle.y + rectangle.height / 2);
}

float calculatePointDistance(Point point1, Point point2) {
    int deltaX = point2.x - point1.x;
    int deltaY = point2.y - point1.y;

    return sqrt(deltaX*deltaX + deltaY*deltaY);
}

const float MAXIMUM_OFFSET_FACTOR_X = 0.03;
const float MAXIMUM_OFFSET_FACTOR_Y = 0.03;
vector<Point> identifyPrimaryObject(vector<vector<Point>> contours) {
    if (contours.empty()) {
        return vector<Point>();
    }

    if (primaryObjectLastPosition[0] == -1) {
        // Just start off with the biggest object
        return contours.at(contours.size() - 1);
    }
    else {
        // TODO: Filter contours by minimum and maximum area
        // TODO: Filter contours by minimum and maximum height/width relation

        vector<Point> biggestObject = contours.at(contours.size() - 1);
        Rect biggestObjectBoundingRectangle = boundingRect(biggestObject);
        Point biggestObjectCenter = calculateRectangleCenter(biggestObjectBoundingRectangle);

        int framesElapsedSinceLastDetection = max(stream.get(CAP_PROP_POS_FRAMES) - primaryObjectLastDetectedFrame, (double)1);
        int maximumOffsetX = (biggestObjectBoundingRectangle.width / 2) + FRAME_WIDTH * MAXIMUM_OFFSET_FACTOR_X * framesElapsedSinceLastDetection;
        int maximumOffsetY = (biggestObjectBoundingRectangle.height / 2) + FRAME_HEIGHT * MAXIMUM_OFFSET_FACTOR_Y * framesElapsedSinceLastDetection;
        int offsetX = abs(biggestObjectCenter.x - primaryObjectLastBoundingRectangle.x);
        int offsetY = abs(biggestObjectCenter.y - primaryObjectLastBoundingRectangle.y);
        if (offsetX <= maximumOffsetX && offsetY <= maximumOffsetY) {
            // Assume biggest object is primary object
            return biggestObject;
        }
        else {
            // Assume object nearest to last position of primary object is still primary object
            cout << "Warning: Assuming primary object is nearest to last position (not largest)" << endl;

            Point primaryObjectLastCenter = calculateRectangleCenter(primaryObjectLastBoundingRectangle);
            float minimumDistance = UINT32_MAX;
            vector<Point> nearestObject;
            for (auto &contour : contours) {
                Point contourCenter = calculateRectangleCenter(boundingRect(contour));

                if (calculatePointDistance(contourCenter, primaryObjectLastCenter) < minimumDistance) {
                    minimumDistance = calculatePointDistance(contourCenter, primaryObjectLastCenter);
                    nearestObject = contour;
                }
            }

            return nearestObject;
        }
    }
}

void drawPrimaryObjectMaximumOffsetLimits(Mat &outputFrame, int x , int y) {
    int framesElapsedSinceLastDetection = max(stream.get(CAP_PROP_POS_FRAMES) - primaryObjectLastDetectedFrame, (double)1);
    int maximumOffsetX = (primaryObjectLastBoundingRectangle.width / 2) + FRAME_WIDTH * MAXIMUM_OFFSET_FACTOR_X * framesElapsedSinceLastDetection;
    int maximumOffsetY = (primaryObjectLastBoundingRectangle.height / 2) + FRAME_HEIGHT * MAXIMUM_OFFSET_FACTOR_Y * framesElapsedSinceLastDetection;

    drawRectangle(outputFrame, Scalar(0, 0, 0), Rect(x - maximumOffsetX, y - maximumOffsetY, maximumOffsetX * 2, maximumOffsetY * 2));
}

void trackObjects(Mat frameDifference, Mat &outputFrame) {
    const bool USE_CONTOUR_DETECTION = true;
    const Scalar PRIMARY_OBJECT_TRACKING_COLOR = Scalar(0, 255, 0);
    const Scalar PRIMARY_OBJECT_LOST_COLOR = Scalar(0, 0, 255);
    const Scalar SECONDARY_OBJECTS_TRACKING_COLOR = Scalar(255, 255, 255);
    const bool DRAW_PRIMARY_OBJECT_TRAIL = true;
    const bool DRAW_PRIMARY_OBJECT_OFFSET_LIMITS = true;
    const bool DRAW_SECONDARY_OBJECT_MARKERS = true;
    const int TEXT_OFFSET_Y = 40;

    if (USE_CONTOUR_DETECTION) {
        // Using contour detection
        // TODO: Add and test PROCESSING_RESOLUTION_FACTOR

        // Detect contours
        Mat frameDifferenceCopy;
        frameDifference.copyTo(frameDifferenceCopy);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(frameDifferenceCopy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        // Draw secondary object markers
        if (DRAW_SECONDARY_OBJECT_MARKERS) {
            for (auto &contour : contours) {
                drawRectangle(outputFrame, SECONDARY_OBJECTS_TRACKING_COLOR, boundingRect(contour));
            }
        }

        // Identify primary object
        vector<Point> primaryObject = identifyPrimaryObject(contours);

        // Draw primary object marker
        if (!primaryObject.empty()) {
            primaryObjectLastBoundingRectangle = boundingRect(primaryObject);
            primaryObjectLastDetectedFrame = stream.get(CAP_PROP_POS_FRAMES);

            Point primaryObjectCenter = calculateRectangleCenter(primaryObjectLastBoundingRectangle);
            int x = primaryObjectLastPosition[0] = primaryObjectCenter.x;
            int y = primaryObjectLastPosition[1] = primaryObjectCenter.y;

            addTrailPoint(Point(x, y));

            drawRectangle(outputFrame, PRIMARY_OBJECT_TRACKING_COLOR, primaryObjectLastBoundingRectangle);
            drawCrosshair(outputFrame, PRIMARY_OBJECT_TRACKING_COLOR, x, y);

            if (DRAW_PRIMARY_OBJECT_OFFSET_LIMITS) {
                drawPrimaryObjectMaximumOffsetLimits(outputFrame, x , y);
            }
        }
        else if (primaryObjectLastPosition[0] != -1) {
            int x = primaryObjectLastPosition[0];
            int y = primaryObjectLastPosition[1];

            drawText(outputFrame, PRIMARY_OBJECT_LOST_COLOR, x, y + TEXT_OFFSET_Y, "Object lost at (" + to_string(x) + "," + to_string(y) + ")");

            drawRectangle(outputFrame, PRIMARY_OBJECT_LOST_COLOR, primaryObjectLastBoundingRectangle);
            drawCrosshair(outputFrame, PRIMARY_OBJECT_LOST_COLOR, x, y);
        }

        if (DRAW_PRIMARY_OBJECT_TRAIL) {
            drawTrail(outputFrame, PRIMARY_OBJECT_TRACKING_COLOR);
        }
    }
    else {
        // Using blob detection
        const float PROCESSING_RESOLUTION_FACTOR = 0.2;

        // Setup blob detection parameters
        cv::SimpleBlobDetector::Params params;
        params.filterByConvexity = false;
        params.filterByColor = false;

        params.filterByArea = true;
        params.minArea = 10.0f;
        params.maxArea = 200.0f;

        params.filterByCircularity = true;
        params.minCircularity = 0.20;
        params.maxCircularity = 0.70;

        params.filterByInertia = true;
        params.minInertiaRatio = 0.20;
        params.maxInertiaRatio = 0.75;

        params.minDistBetweenBlobs = 20.0f;

        // Detect blobs
        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
        vector<cv::KeyPoint> keypoints;
        Mat frameDifferenceResized;
        cv::resize(frameDifference, frameDifferenceResized, Size(), PROCESSING_RESOLUTION_FACTOR, PROCESSING_RESOLUTION_FACTOR, INTER_AREA);
        detector->detect(frameDifferenceResized, keypoints);

        // Draw secondary object markers
        if (DRAW_SECONDARY_OBJECT_MARKERS) {
            for (int keypointIndex = 0; keypointIndex < keypoints.size(); keypointIndex++) {
                float x = keypoints[keypointIndex].pt.x * (1 / PROCESSING_RESOLUTION_FACTOR);
                float y = keypoints[keypointIndex].pt.y * (1 / PROCESSING_RESOLUTION_FACTOR);

                drawCrosshair(outputFrame, SECONDARY_OBJECTS_TRACKING_COLOR, x, y);
            }
        }

        // TODO: Identify main object

        // TODO: Draw primary object marker

    }
}


int main() {
    bool trackingEnabled = true;
    bool pause = false;

    Mat currentFrame, nextFrame;
    Mat currentFrameGrayscale, nextFrameGrayscale;
    Mat frameDifference;
    Mat frameDifferenceThresholded;
    Mat frameDifferenceBlurred;

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
        primaryObjectLastPosition[0] = -1;
        primaryObjectLastPosition[1] = -1;
        primaryObjectLastBoundingRectangle = Rect(-1, -1, -1, -1);
        primaryObjectLastDetectedFrame = -1;
        trail = vector<Point>();


        // Analyze frames
        while (liveStream || stream.get(CV_CAP_PROP_POS_FRAMES) < stream.get(CV_CAP_PROP_FRAME_COUNT) - 1) {
            chrono::high_resolution_clock::time_point frameProcessingStartTime = chrono::high_resolution_clock::now();

            // Get current frame as grayscale
            stream.read(currentFrame);
            cvtColor(currentFrame, currentFrameGrayscale, COLOR_BGR2GRAY);

            // Get next frame as grayscale
            stream.read(nextFrame);
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


            // Track objects
            if (trackingEnabled) {
                trackObjects(frameDifferenceBlurred, currentFrame);
            }


            // Calculate wait time
            auto frameProcessingDuration = chrono::duration_cast<chrono::milliseconds>(
                chrono::high_resolution_clock::now() - frameProcessingStartTime).count();

            // TODO: Reuse second frame as first frame next time
            int waitTime = 2 * 1000 / framerate - frameProcessingDuration;
            if (waitTime <= 0) {
                drawText(currentFrame, Scalar(0, 0, 255), FRAME_WIDTH / 2, 40,
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
