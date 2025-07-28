#include "camera_capture_impl.hpp"
#include <libcamera/base/span.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <unistd.h>  // lseek

// CV_FOURCC from new version of <opencv2/core/cvdef.h>
// #include <opencv2/core/cvdef.h>
/** @brief Constructs the 'fourcc' code, used in video codecs and many other
   places. Simply call it with 4 chars like `CV_FOURCC('I', 'Y', 'U', 'V')`
*/
CV_INLINE int CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

#define LIBCAMERA_CHECK(result) \
    if (result != 0) \
    { \
        CV_LOG_ERROR(NULL, "libcamera operation failed, error " << result); \
        CV_Assert(false); \
    }

namespace cv {

namespace controls = libcamera::controls;

CameraCaptureImpl::CameraCaptureImpl() {}

CameraCaptureImpl::~CameraCaptureImpl()
{
    close();
}

std::string CameraCaptureImpl::id() const
{
    if (camera_)
    {
        return camera_->id();
    }
    else
    {
        return "";
    }
}

bool CameraCaptureImpl::open(std::shared_ptr<libcamera::Camera> camera)
{
    if (camera->acquire() != 0)
    {
        CV_LOG_ERROR(NULL, "Failed to acquire camera");
        return false;
    }

    auto camera_config = camera->generateConfiguration({libcamera::StreamRole::VideoRecording});
    if (!camera_config)
    {
        CV_LOG_ERROR(NULL, "Failed to generate camera configuration");
        camera->release();
        return false;
    }
    if (camera_config->empty())
    {
        CV_LOG_ERROR(NULL, "Camera configuration is empty");
        camera->release();
        return false;
    }

    camera_config->at(0).pixelFormat = libcamera::formats::RGB888;

    camera_ = camera;
    camera_config_ = std::move(camera_config);
    return true;
}

bool CameraCaptureImpl::isOpened() const
{
    return camera_ != nullptr;
}

void CameraCaptureImpl::close()
{
    if (!camera_)
        return;

    stopCapture();
    int err = camera_->release();
    LIBCAMERA_CHECK(err);

    camera_.reset();
    camera_config_.reset();
}

bool CameraCaptureImpl::grabFrame()
{
    if (!camera_)
        return false;

    if (!running_ && !startCapture())
    {
        return false;
    }

    // submit one request to capture frame
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (free_requests_.empty())
    {
        CV_LOG_WARNING(NULL, "no more free buffer/request");
        return false;
    }

    if (camera_->queueRequest(free_requests_.front().get()))
    {
        CV_LOG_ERROR(NULL, "queueRequest failed");
        return false;
    }

    pending_requests_.push(std::move(free_requests_.front()));
    free_requests_.pop();
    return true;
}

bool CameraCaptureImpl::retrieveFrame(int flag, OutputArray image)
{
    if (!camera_)
    {
        // camera not opened
        return false;
    }

    // Wait for a completed request to be available
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (!queue_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this] {
            return !completed_requests_.empty();
        }))
    {
        CV_LOG_WARNING(NULL, "wait request completion timeout");
        return false;
    }

    // Get the completed request
    auto request = std::move(completed_requests_.front());
    completed_requests_.pop();
    lock.unlock();

    // process request without lock
    bool result = retrieveFrameFromRequest(request.get(), image);
    CV_LOG_IF_ERROR(NULL, !result, "retrieve frame from completed request failed");

    // Move request back to free queue
    lock.lock();
    request->reuse(libcamera::Request::ReuseBuffers);
    free_requests_.push(std::move(request));

    return result;
}

void CameraCaptureImpl::setOrientation(libcamera::Orientation value)
{
    if (!camera_config_ || running_)
    {
        CV_LOG_ERROR(NULL, "Camera is not opened or is running");
        return;
    }

    camera_config_->orientation = value;
}

double CameraCaptureImpl::getProperty(int propId) const
{
    if (!camera_config_)
    {
        CV_LOG_ERROR(NULL, "Camera configuration not available");
        return -1;
    }

    const auto& stream_config = camera_config_->at(0);

    switch (propId)
    {
        case CAP_PROP_FRAME_WIDTH: return static_cast<double>(stream_config.size.width);

        case CAP_PROP_FRAME_HEIGHT: return static_cast<double>(stream_config.size.height);

        case CAP_PROP_FOURCC:
            // Convert libcamera::PixelFormat to FourCC
            return static_cast<double>(stream_config.pixelFormat.fourcc());
        case CAP_PROP_FPS:
            if (frame_rate_)
            {
                return frame_rate_.value();
            }
            else
            {
                return -1;
            }
        case CAP_PROP_BUFFERSIZE: return stream_config.bufferCount;
        default: return -1;
    }
    // unknown parameter or value not available
    return -1;
}

bool CameraCaptureImpl::setProperty(int propId, double value)
{
    if (!camera_config_ || running_)
    {
        CV_LOG_ERROR(NULL, "Camera is not opened or is running");
        return false;
    }

    auto& stream_config = camera_config_->at(0);
    switch (propId)
    {
        case CAP_PROP_FRAME_WIDTH:
            stream_config.size.width = static_cast<unsigned int>(value);
            break;
        case CAP_PROP_FRAME_HEIGHT:
            stream_config.size.height = static_cast<unsigned int>(value);
            break;
        case CAP_PROP_FOURCC:
        {
            libcamera::PixelFormat pixel_format(static_cast<uint32_t>(value));
            std::vector<libcamera::PixelFormat> supported_formats =
                stream_config.formats().pixelformats();
            if (std::find(supported_formats.begin(), supported_formats.end(), pixel_format) ==
                supported_formats.end())
            {
                CV_LOG_ERROR(NULL,
                             "pixel format " << pixel_format << " is not supported by camera");
                return false;
            }
            stream_config.pixelFormat = pixel_format;
        }
        break;
        case CAP_PROP_BUFFERSIZE: stream_config.bufferCount = static_cast<uint32_t>(value); break;
        case CAP_PROP_FPS: frame_rate_ = value; break;
        default: return false;
    }

    libcamera::CameraConfiguration::Status status = camera_config_->validate();
    if (status == libcamera::CameraConfiguration::Invalid)
    {
        // TODO: restore value
        return false;
    }
    else if (status == libcamera::CameraConfiguration::Adjusted)
    {
        CV_LOG_WARNING(NULL, "Camera configuration adjusted");
        return true;
    }
    return true;
}

void CameraCaptureImpl::DumpCameraConfig()
{
    CV_Assert(camera_config_);

    CV_LOG_INFO(NULL, "Camera Configuration: " << camera_config_->orientation);
    if (camera_config_->sensorConfig)
    {
        CV_LOG_INFO(NULL, "Sensor: ");
    }

    for (int i = 0; i < camera_config_->size(); ++i)
    {
        auto& stream_config = camera_config_->at(i);

        CV_LOG_INFO(NULL,
                    "Stream " << i << ": "  //
                              << stream_config.size.width << "x" << stream_config.size.height  //
                              << ", " << stream_config.pixelFormat);
        CV_LOG_INFO(NULL,
                    "  stride=" << stream_config.stride << ", frameSize=" << stream_config.frameSize
                                << ", bufferCount=" << stream_config.bufferCount);
    }
}

bool CameraCaptureImpl::startCapture()
{
    CV_Assert(camera_);
    CV_LOG_INFO(NULL, "Start capture of camera " << camera_->id());

    // apply configuration
    auto& stream_config = camera_config_->at(0);
    if (camera_->configure(camera_config_.get()))
    {
        CV_LOG_ERROR(NULL, "Failed to configure camera");
        return false;
    }
    DumpCameraConfig();

    // Store stream pointer for later use
    stream_ = stream_config.stream();

    // Allocate buffers
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    int n_buffer = allocator_->allocate(stream_);
    if (n_buffer < 0)
    {
        CV_LOG_ERROR(NULL, "Failed to allocate buffers");
        return false;
    }
    CV_LOG_INFO(NULL, "Allocated " << n_buffer << " buffers");

    // Create all requests according to available buffer count
    for (int i = 0; i < n_buffer; ++i)
    {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request)
        {
            CV_LOG_ERROR(NULL, "Failed to create request " << i << " in initializeCamera");
            break;
        }

        if (request->addBuffer(stream_, allocator_->buffers(stream_)[i].get()))
        {
            CV_LOG_ERROR(NULL, "Failed to add buffer to request " << i << " in initializeCamera");
            break;
        }

        free_requests_.push(std::move(request));
    }
    total_request_count_ = free_requests_.size();

    // Connect signal
    camera_->requestCompleted.connect(this, &CameraCaptureImpl::onRequestCompleted);

    // Start camera
    libcamera::ControlList ctrl_list(camera_->controls());
    prepareControlList(ctrl_list);
    if (!prepareControlList(ctrl_list) || camera_->start(&ctrl_list) != 0)
    {
        CV_LOG_ERROR(NULL, "Failed to start camera");
        // TODO: clean up
        return false;
    }

    running_ = true;
    return true;
}

bool CameraCaptureImpl::prepareControlList(libcamera::ControlList& ctrl_list)
{
    if (frame_rate_)
    {
        // calculate frame duration from frame rate
        int64_t frame_duration = 1000000 / frame_rate_.value();
        ctrl_list.set(controls::FrameDurationLimits,
                      libcamera::Span<const int64_t, 2>({frame_duration, frame_duration}));
    }
    return true;
}

void CameraCaptureImpl::stopCapture()
{
    if (!running_)
    {
        return;
    }

    int ret = camera_->stop();
    LIBCAMERA_CHECK(ret);

    // wait for no pending request
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] {
        return pending_requests_.empty();
    });

    // All requests should be either in free or completed queues
    CV_Assert(free_requests_.size() + completed_requests_.size() == total_request_count_);
    // free all requests
    free_requests_ = {};
    completed_requests_ = {};
    // unmap dmabufs
    for (auto& [fd, buf] : mapped_dmabuf_)
    {
        if (buf.address)
            munmap(buf.address, buf.length);
    }
    // free buffer
    allocator_->free(stream_);
    allocator_.reset();
    stream_ = nullptr;

    running_ = false;
}

void CameraCaptureImpl::onRequestCompleted(libcamera::Request* request)
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    CV_Assert(request == pending_requests_.front().get());

    auto completed_request = std::move(pending_requests_.front());
    pending_requests_.pop();

    if (completed_request->status() == libcamera::Request::RequestCancelled)
    {
        // move to free queue
        CV_LOG_INFO(NULL, "Request cancelled");
        completed_request->reuse(libcamera::Request::ReuseBuffers);
        free_requests_.push(std::move(completed_request));
    }
    else
    {
        // move from pending to completed queue
        completed_requests_.push(std::move(completed_request));
    }

    queue_cv_.notify_one();
}

bool CameraCaptureImpl::retrieveFrameFromRequest(libcamera::Request* request,
                                                 OutputArray output_image)
{
    libcamera::FrameBuffer* buffer = request->findBuffer(stream_);
    if (buffer == nullptr)
    {
        return false;
    }

    // Get configuration values
    CV_Assert(camera_config_);
    const auto& stream_config = camera_config_->at(0);

    const auto& planes = buffer->planes();
    if (planes.empty())
        return false;

    if (planes.size() > 1)
    {
        CV_LOG_WARNING(NULL, "multiplane framebuffer not supported yet");
        return false;
    }

    const int fd = planes[0].fd.get();

    DmaBufferMapping& mapped_buf = mapped_dmabuf_[fd];
    if (mapped_buf.address == 0)
    {
        // dmabuf not mapped yet
        const auto dmabuf_length = lseek(fd, 0, SEEK_END);
        if (dmabuf_length == -1)
        {
            CV_LOG_ERROR(NULL, "lseek dambuf failed");
            return false;
        }
        void* dmabuf_address = mmap(nullptr, dmabuf_length, PROT_READ, MAP_SHARED, fd, 0);
        if (dmabuf_address == MAP_FAILED)
        {
            CV_LOG_ERROR(NULL, "mmap dambuf failed");
        }
        mapped_buf.address = dmabuf_address;
        mapped_buf.length = dmabuf_length;
        CV_LOG_DEBUG(NULL, "New dmabuf mapped, length=" << dmabuf_length);
    }

    // copy/convert pixel data to OpenCV Matrix
    void* memory = (uint8_t*)mapped_buf.address + planes[0].offset;
    const unsigned int w = stream_config.size.width;
    const unsigned int h = stream_config.size.height;
    const unsigned int stride = stream_config.stride;
    CV_Assert(stride * h == planes[0].length);

    const libcamera::PixelFormat pixel_fmt = stream_config.pixelFormat;

    if (pixel_fmt == libcamera::formats::RGB888 || pixel_fmt == libcamera::formats::BGR888)
    {
        Mat tmp = Mat(h, w, CV_8UC3, memory, stride).clone();
        output_image.move(tmp);
    }
    else if (pixel_fmt == libcamera::formats::R8)
    {
        Mat tmp = Mat(h, w, CV_8UC1, memory, stride).clone();
        output_image.move(tmp);
    }
    else if (pixel_fmt == libcamera::formats::YUYV)
    {
        Mat tempFrame(h, w, CV_8UC2, memory, stride);
        cvtColor(tempFrame, output_image, COLOR_YUV2BGR_YUYV);
    }
    else
    {
        CV_LOG_ERROR(NULL, "Unsupported pixel format " << pixel_fmt);
        return false;
    }

    return true;
}

}  // namespace cv
