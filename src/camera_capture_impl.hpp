#ifndef CAMERA_CAPTURE_IMPL_HPP
#define CAMERA_CAPTURE_IMPL_HPP

#include <opencv2/core.hpp>
#include <libcamera/libcamera.h>
#include <memory>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <optional>

namespace cv {

class CameraCaptureImpl
{
public:
    CameraCaptureImpl();
    ~CameraCaptureImpl();

    bool open(std::shared_ptr<libcamera::Camera> cam);
    void close();

    std::string id() const;

    libcamera::Camera* camera() const
    {
        return camera_.get();
    }

    // IVideoCapture
    double getProperty(int propId) const;
    bool setProperty(int propId, double value);
    bool grabFrame();
    bool retrieveFrame(int flag, OutputArray image);
    bool isOpened() const;

    // libcamera specific interface
    void setOrientation(libcamera::Orientation value);

    // Extended interface(internal use)
    int grabFrame(int n);
    /// retrieve first completed request(use with causion)
    libcamera::Request* retrieveRequest(int timeout_ms);
    bool retrieveFrameFromRequest(libcamera::Request* request, OutputArray image);
    /// finish first completed request(use with causion)
    void finishRequest(libcamera::Request*);

private:
    void onRequestCompleted(libcamera::Request* request);
    void onDisconnected();

    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> camera_config_;
    void DumpCameraConfig();

    libcamera::Stream* stream_ = nullptr;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;

    std::optional<double> frame_rate_;

    // mutex to protect three request queues
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    int total_request_count_ = 0;
    std::queue<std::unique_ptr<libcamera::Request>> free_requests_;
    std::queue<std::unique_ptr<libcamera::Request>> pending_requests_;  // request that is submmited
    std::queue<std::unique_ptr<libcamera::Request>> completed_requests_;

    bool prepareControlList(libcamera::ControlList& ctrl_list);
    bool startCapture();
    void stopCapture();
    bool running_ = false;

    struct DmaBufferMapping
    {
        void* address = nullptr;  // dmabuf mapped address
        size_t length = 0;  // dmabuf length
    };

    // index is fd
    std::unordered_map<int, DmaBufferMapping> mapped_dmabuf_;
};

}  // namespace cv

#endif  // CAMERA_CAPTURE_IMPL_HPP
