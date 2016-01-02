#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;

//const string FILENAME = "";
const string FILENAME = "input/video2.mp4";
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1080;
const float TRAIL_DURATION = 3.0;

const string RAW_FRAME_WINDOW_NAME = "Raw Input";
const string DIFFERENCE_FRAME_WINDOW_NAME = "Raw Difference";
const string THRESHOLDED_FRAME_WINDOW_NAME = "Thresholded Difference";
const string BLURRED_FRAME_WINDOW_NAME = "Blurred Difference (Final)";
const string SETTINGS_WINDOW_NAME = "Settings";

int thresholdSensitivity = 32;
int blurSize = 20;
int framerate = 30;

int primaryObjectLastPosition[2] = { 0,0 };
Rect primaryObjectLastBoundingRectangle = Rect(0, 0, 0, 0);


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

void drawText(Mat &frame, Scalar color, int x, int y, string text) {
    const float TEXT_OFFSET_X_FACTOR = 9.0;
    const int FONT_FACE = 1;
    const int FONT_SCALE = 1;
    const int FONT_THICKNESS = 1;
    const int LINE_TYPE = 8;

    putText(frame, text, Point(x - (text.length() / 2 * TEXT_OFFSET_X_FACTOR), y),
        FONT_FACE, FONT_SCALE, color, FONT_THICKNESS, LINE_TYPE);
}

void drawTrail(Mat &frame, Scalar color, vector<pair<int, int>> points) {
    const int MINIMUM_THICKNESS = 1;
    const int MAXIMUM_THICKNESS = 3;
    const int LINE_TYPE = 8;

    for (int pointIndex = 0; pointIndex < points.size() - 2; pointIndex++) {
        if (points[pointIndex].first == -1 || points[pointIndex + 1].first == -1) {
            break;
        }

        Point startingPoint = Point(points[pointIndex].first, points[pointIndex].second);
        Point endPoint = Point(points[pointIndex + 1].first, points[pointIndex + 1].second);
        int thickness = round(MINIMUM_THICKNESS + (MAXIMUM_THICKNESS - MINIMUM_THICKNESS) / (points.size() - 1) * pointIndex); // Linear progression

        line(frame, startingPoint, endPoint, color, thickness, LINE_TYPE);
    }
}

vector<Point> identifyPrimaryObject(vector<vector<Point>> contours) {
    if (contours.empty()) {
        return vector<Point>();
    }

    //the largest contour is found at the end of the contours vector
    //we will simply assume that the biggest contour is the object we are looking for.
    return contours.at(contours.size() - 1);
}

void trackObjects(Mat frameDifference, Mat &outputFrame) {
    const Scalar PRIMARY_OBJECT_TRACKING_COLOR = Scalar(0, 255, 0);
    const Scalar PRIMARY_OBJECT_LOST_COLOR = Scalar(0, 0, 255);
    const Scalar SECONDARY_OBJECTS_TRACKING_COLOR = Scalar(255, 255, 255);
    const bool DRAW_SECONDARY_OBJECT_MARKERS = true;
    const int TEXT_OFFSET_Y = 50;

    if (true) { // Using contour detection
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

            primaryObjectLastPosition[0] = primaryObjectLastBoundingRectangle.x + primaryObjectLastBoundingRectangle.width / 2;
            primaryObjectLastPosition[1] = primaryObjectLastBoundingRectangle.y + primaryObjectLastBoundingRectangle.height / 2;
            int x = primaryObjectLastPosition[0];
            int y = primaryObjectLastPosition[1];

            drawRectangle(outputFrame, PRIMARY_OBJECT_TRACKING_COLOR, primaryObjectLastBoundingRectangle);
            drawCrosshair(outputFrame, PRIMARY_OBJECT_TRACKING_COLOR, x, y);
        }
        else if (primaryObjectLastPosition[0] != -1) {
            int x = primaryObjectLastPosition[0];
            int y = primaryObjectLastPosition[1];

            drawText(outputFrame, PRIMARY_OBJECT_LOST_COLOR, x, y + TEXT_OFFSET_Y, "Object lost at (" + to_string(x) + "," + to_string(y) + ")");

            drawRectangle(outputFrame, PRIMARY_OBJECT_LOST_COLOR, primaryObjectLastBoundingRectangle);
            drawCrosshair(outputFrame, PRIMARY_OBJECT_LOST_COLOR, x, y);
        }
    }
    else { // Using blob detection
        const float PROCESSING_RESOLUTION_FACTOR = 0.2;

        // Setup blob detection parameters
        cv::SimpleBlobDetector::Params params;
        params.filterByConvexity = false;
        params.filterByColor = false;

        params.filterByArea = true;
        params.minArea = 25.0f;
        params.maxArea = 400.0f;

        params.filterByCircularity = true;
        params.minCircularity = 0.4;
        params.maxCircularity = 1.0;

        params.filterByInertia = true;
        params.minInertiaRatio = 0.25;
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
    bool debugMode = false;
    bool pause = false;

    VideoCapture stream;
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


        // Analyze frames
        while (liveStream || stream.get(CV_CAP_PROP_POS_FRAMES) < stream.get(CV_CAP_PROP_FRAME_COUNT) - 1) {
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


            // Display output
            createNamedWindow(RAW_FRAME_WINDOW_NAME);
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

            int waitTime = 1000 / framerate;
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
