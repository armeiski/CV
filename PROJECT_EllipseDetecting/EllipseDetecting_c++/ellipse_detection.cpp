#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("input.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file\n";
        return -1;
    }

    int res_fac = 1;

    int lh = 37, ls = 76, lv = 0;
    int hh = 255, hs = 255, hv = 255;

    cv::namedWindow("custom window", cv::WINDOW_KEEPRATIO);

    while (true) {
        cv::Mat frame, img_resized, blurred, hsv, mask, morph, filtered;
        cap >> frame;
        if (frame.empty())
            break;

        cv::resize(frame, img_resized, cv::Size(frame.cols / res_fac, frame.rows / res_fac));
        cv::blur(img_resized, blurred, cv::Size(11, 11));
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

        cv::inRange(hsv, cv::Scalar(lh, ls, lv), cv::Scalar(hh, hs, hv), mask);
        mask = 255 - mask;

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(mask, morph, cv::MORPH_CLOSE, kernel);

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(morph, labels, stats, centroids, 4, CV_32S);

        filtered = cv::Mat::zeros(morph.size(), CV_8U);
        for (int i = 1; i < num_labels; ++i) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area > 5000) {
                cv::Mat label_mask = (labels == i);
                label_mask.copyTo(filtered, label_mask);
            }
        }

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(filtered, circles, cv::HOUGH_GRADIENT_ALT, 4, 20, 100, 0.5, 50);

        if (!circles.empty()) {
            float max_radius = 0;
            cv::Vec3f max_circle;

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

            cv::circle(img_resized, cv::Point(center_x, center_y), radius, cv::Scalar(0, 255, 0), 2);
            cv::circle(img_resized, cv::Point(center_x, center_y), 2, cv::Scalar(0, 0, 255), 3);

            std::stringstream ss;
            ss << "Center: (" << center_x << ", " << center_y << "), Radius: " << radius;
            cv::putText(img_resized, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 2);
        } else {
            std::cout << "No circles found\n";
        }

        cv::imshow("custom window", img_resized);
        cv::resizeWindow("custom window", frame.cols / 2, frame.rows / 2);

        cv::imshow("custom mask window", mask);
        cv::resizeWindow("custom mask window", frame.cols / 2, frame.rows / 2);

        cv::imshow("custom filt mask window", filtered);
        cv::resizeWindow("custom filt mask window", frame.cols / 2, frame.rows / 2);

        if (cv::waitKey(30) == 'q')
            break;
    }

    return 0;
}
