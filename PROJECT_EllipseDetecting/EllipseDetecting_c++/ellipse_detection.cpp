#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("input.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file\n";
        return -1;
    }

    int res_fac = 1;

    // Declare variables for trackbar positions
    int lh = 37, ls = 76, lv = 0;
    int hh = 255, hs = 255, hv = 255;

    namedWindow("custom window", WINDOW_KEEPRATIO);
    createTrackbar("lh", "custom window", &lh, 255, nullptr);
    createTrackbar("ls", "custom window", &ls, 255, nullptr);
    createTrackbar("lv", "custom window", &lv, 255, nullptr);
    createTrackbar("hh", "custom window", &hh, 255, nullptr);
    createTrackbar("hs", "custom window", &hs, 255, nullptr);
    createTrackbar("hv", "custom window", &hv, 255, nullptr);

    while (true) {
        Mat frame, img_resized, blurred, hsv, mask, morph, filtered;
        cap >> frame;
        if (frame.empty())
            break;

        resize(frame, img_resized, Size(frame.cols / res_fac, frame.rows / res_fac));
        blur(img_resized, blurred, Size(11, 11));
        cvtColor(blurred, hsv, COLOR_BGR2HSV);

        // Get current positions of trackbars
        lh = getTrackbarPos("lh", "custom window");
        ls = getTrackbarPos("ls", "custom window");
        lv = getTrackbarPos("lv", "custom window");
        hh = getTrackbarPos("hh", "custom window");
        hs = getTrackbarPos("hs", "custom window");
        hv = getTrackbarPos("hv", "custom window");

        inRange(hsv, Scalar(lh, ls, lv), Scalar(hh, hs, hv), mask);
        mask = 255 - mask;

        Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(mask, morph, MORPH_CLOSE, kernel);

        Mat labels, stats, centroids;
        int num_labels = connectedComponentsWithStats(morph, labels, stats, centroids, 4, CV_32S);

        filtered = Mat::zeros(morph.size(), CV_8U);
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, CC_STAT_AREA);
            if (area > 5000) {
                Mat label_mask = (labels == i);
                label_mask.copyTo(filtered, label_mask);
            }
        }

        vector<Vec3f> circles;
        HoughCircles(filtered, circles, HOUGH_GRADIENT_ALT, 4, 20, 100, 0.5, 50);

        if (!circles.empty()) {
            float max_radius = 0;
            Vec3f max_circle;

            for (const auto& circle : circles) {
                float radius = circle[2];
                if (radius > max_radius) {
                    max_radius = radius;
                    max_circle = circle;
                }
            }

            int center_x = cvRound(max_circle[0]);
            int center_y = cvRound(max_circle[1]);
            int radius = cvRound(max_circle[2]);

            circle(img_resized, Point(center_x, center_y), radius, Scalar(0, 255, 0), 2);
            circle(img_resized, Point(center_x, center_y), 2, Scalar(0, 0, 255), 3);

            stringstream ss;
            ss << "Center: (" << center_x << ", " << center_y << "), Radius: " << radius;
            putText(img_resized, ss.str(), Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 2);
        } else {
            cout << "No circles found\n";
        }

        imshow("custom window", img_resized);
        resizeWindow("custom window", frame.cols / 2, frame.rows / 2);

        imshow("custom mask window", mask);
        resizeWindow("custom mask window", frame.cols / 2, frame.rows / 2);

        imshow("custom filt mask window", filtered);
        resizeWindow("custom filt mask window", frame.cols / 2, frame.rows / 2);

        if (waitKey(30) == 'q')
            break;
    }

    return 0;
}
