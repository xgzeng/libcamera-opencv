#include <libcamera_cv.hpp>
#include <opencv2/highgui.hpp>

int main() {
    cv::CameraCapture cap(1);
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::imshow("Camera", frame);
        if (cv::pollKey() == 'q') break;
    }
    return 0;
}