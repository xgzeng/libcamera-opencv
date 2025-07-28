#ifndef LIBCAMERA_OPENCV_HPP
#define LIBCAMERA_OPENCV_HPP

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <memory>
#include <vector>

namespace cv {

// Forward declarations
class CameraCaptureImpl;

class CameraCapture
{
public:
    // Constructor/Destructor
    CameraCapture();
    explicit CameraCapture(int index);
    explicit CameraCapture(const std::string& device);
    virtual ~CameraCapture();

    // OpenCV VideoCapture interface
    virtual bool open(int index);
    virtual bool open(const std::string& device);
    virtual bool isOpened() const;
    virtual void release();
    virtual bool read(OutputArray image);
    virtual CameraCapture& operator>>(Mat& image);
    virtual bool grab();
    virtual bool retrieve(OutputArray image, int flag = 0);
    virtual double get(int propId) const;
    virtual bool set(int propId, double value);

    // Libcamera specific interface
    std::string id() const;
    void setOrientation(bool hflip, bool vflip);

private:
    // Internal implementation
    std::unique_ptr<CameraCaptureImpl> pImpl;
};

}  // namespace cv

#endif  // LIBCAMERA_OPENCV_HPP