#include "libcamera_cv.hpp"
#include "camera_capture_impl.hpp"
#include <libcamera/orientation.h>
#include <opencv2/core/utils/logger.hpp>
#include <mutex>

namespace cv {

libcamera::CameraManager gCameraManager;
int gCameraManagerRefCount = 0;
std::mutex gCameraManagerMutex;

// Helper functions for camera manager lifecycle management
static void startCameraManager()
{
    std::lock_guard<std::mutex> lock(gCameraManagerMutex);
    if (gCameraManagerRefCount == 0)
    {
        gCameraManager.start();
        CV_LOG_INFO(NULL, "Camera manager started");
    }
    gCameraManagerRefCount++;
}

static void stopCameraManager()
{
    std::lock_guard<std::mutex> lock(gCameraManagerMutex);
    gCameraManagerRefCount--;
    if (gCameraManagerRefCount == 0)
    {
        gCameraManager.stop();
        CV_LOG_INFO(NULL, "Camera manager stopped");
    }
}

// CameraCapture public interface implementation
CameraCapture::CameraCapture() : pImpl(std::make_unique<CameraCaptureImpl>())
{
    startCameraManager();
}

CameraCapture::CameraCapture(int index) : pImpl(std::make_unique<CameraCaptureImpl>())
{
    startCameraManager();
    open(index);
}

CameraCapture::CameraCapture(const std::string& device) :
    pImpl(std::make_unique<CameraCaptureImpl>())
{
    startCameraManager();
    open(device);
}

CameraCapture::~CameraCapture()
{
    release();
    stopCameraManager();
}

libcamera::Camera* CameraCapture::camera() const
{
    return pImpl->camera();
}

bool CameraCapture::open(int index)
{
    if (pImpl->isOpened())
    {
        return false;
    }

    auto cameras = gCameraManager.cameras();
    if (index <= 0 || index > cameras.size())
    {
        CV_LOG_ERROR(NULL, "Camera index " << index << " out of range");
        return false;
    }

    return pImpl->open(cameras[index - 1]);
}

bool CameraCapture::open(const std::string& device)
{
    if (pImpl->isOpened())
    {
        return false;
    }

    auto camera = gCameraManager.get(device);
    if (!camera)
    {
        CV_LOG_ERROR(NULL, "Failed to get camera: " << device);
        return false;
    }

    return pImpl->open(camera);
}

bool CameraCapture::isOpened() const
{
    return pImpl->isOpened();
}

void CameraCapture::release()
{
    pImpl->close();
}

std::string CameraCapture::id() const
{
    return pImpl->id();
}

bool CameraCapture::read(OutputArray image)
{
    return grab() && retrieve(image);
}

CameraCapture& CameraCapture::operator>>(Mat& image)
{
    read(image);
    return *this;
}

bool CameraCapture::grab()
{
    return pImpl->grabFrame();
}

bool CameraCapture::retrieve(OutputArray image, int flag)
{
    return pImpl->retrieveFrame(flag, image);
}

double CameraCapture::get(int propId) const
{
    return pImpl->getProperty(propId);
}

bool CameraCapture::set(int propId, double value)
{
    return pImpl->setProperty(propId, value);
}

void CameraCapture::setOrientation(bool hflip, bool vflip)
{
    libcamera::Orientation orientation;

    if (hflip && vflip)
    {
        // Both horizontal and vertical flip = 180 degree rotation
        orientation = libcamera::Orientation::Rotate180;
    }
    else if (hflip)
    {
        // Horizontal flip only = mirror
        orientation = libcamera::Orientation::Rotate0Mirror;
    }
    else if (vflip)
    {
        // Vertical flip only = 180 degree rotation + mirror
        orientation = libcamera::Orientation::Rotate180Mirror;
    }
    else
    {
        // No flip = normal orientation
        orientation = libcamera::Orientation::Rotate0;
    }

    pImpl->setOrientation(orientation);
}

}  // namespace cv
